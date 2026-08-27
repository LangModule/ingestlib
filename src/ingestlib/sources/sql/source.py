"""SqlSource — answer a question from a SQL database (generated, or verified).

Default path: GENERATE read-only SQL from the schema (introspected + the
source's `tables` hints), guard it, execute it, self-correct once on a DB
error. If the source declares `verified:` queries, a matching question runs
that reviewed SQL instead — guaranteed correct. Every path is bounded by the
read-only role, the statement allowlist, LIMIT, and a timeout (safety.py).
Uses foundations/llm (generate / param-fill / match) exactly like the
operations do — no new engine.
"""
import asyncio
import math
from typing import Any

from pydantic import BaseModel, Field, create_model

from ingestlib.config import SourceSpec
from ingestlib.foundations.llm import achat_structured, aembed_text
from ingestlib.sources.base import Source, SourceResult
from ingestlib.sources.sql import engine
from ingestlib.sources.sql.safety import guard, with_limit
from ingestlib.sources.sql.schema import SchemaIndex
from ingestlib.utils.logger import get_logger


_RENDER_ROW_CAP = 50        # rows rendered into the prompt-ready content string
_WIDEN_FACTOR = 4           # on a self-correct retry, retrieve this many× more tables

# Cosine to accept a verified-query match — calibrated PER embedding provider,
# because the score scale depends on whether the embeddings are asymmetric. The
# question embeds as a query (GENERIC_RETRIEVAL) and the verified description as a
# document (GENERIC_INDEX):
#   - Bedrock HONORS the purpose → asymmetric scale: exact match ~0.48, confusable
#     but wrong ~0.24 → 0.35 sits cleanly in the gap.
#   - OpenAI / Ollama IGNORE the purpose → symmetric scale: scores run higher and
#     the confusable gap is tighter. Measured on text-embedding-3: should-match
#     ≥0.68, confusable-but-wrong ≤0.55 → 0.62 sits in the gap (0.35 there over-fires,
#     wrongly matching "how many nations" to a "how many regions" verified query).
# A single constant can't serve both, so the threshold follows the provider.
_VERIFIED_THRESHOLD_ASYMMETRIC = 0.35   # bedrock (purpose-aware embeddings)
_VERIFIED_THRESHOLD_SYMMETRIC = 0.62    # openai / ollama (purpose ignored)


def _verified_threshold() -> float:
    """The verified-match cosine floor for the configured embedding provider."""
    from ingestlib.config import get_config

    if get_config().embedding_provider == "bedrock":
        return _VERIFIED_THRESHOLD_ASYMMETRIC
    return _VERIFIED_THRESHOLD_SYMMETRIC


class _GeneratedSQL(BaseModel):
    """The model's SQL answer — a single read-only statement, no prose."""
    sql: str = Field(description="one read-only SQL query answering the question")


class SqlSource(Source):
    """A read-only SQL database exposed as a retrieval Source."""

    def __init__(self, spec: SourceSpec) -> None:
        self.name = spec.name
        self._spec = spec
        self._logger = get_logger(f"ingestlib.sources.sql.{spec.name}")
        self._index: SchemaIndex | None = None
        self._verified_embeddings: dict[str, list[float]] | None = None

    # ---- public contract ----

    async def answer(self, question: str, *, top_k: int = 5) -> list[SourceResult]:
        spec = self._spec
        verified = await self._match_verified(question)
        if verified is not None:
            params = await self._fill_params(question, verified)
            sql, is_verified = verified["sql"], True
        else:
            sql, params, is_verified = await self._generate(question), {}, False

        sql = with_limit(guard(sql, allow=spec.allow), spec.row_limit)
        try:
            columns, rows = await self._execute(sql, params)
        except Exception as exc:
            if is_verified:
                raise
            self._logger.warning("generated SQL failed, self-correcting once: %s", exc)
            # Widen the schema on the retry: a missing table/column is the common
            # cause, and retrieval may simply have omitted it (the unrecoverable
            # failure mode). Pulling in more tables is the recall safety net.
            retry = await self._generate(question, prior_sql=sql, error=str(exc), widen=True)
            sql = with_limit(guard(retry, allow=spec.allow), spec.row_limit)
            columns, rows = await self._execute(sql, {})
            params = {}

        return [SourceResult(
            content=self._render(columns, rows),
            source=self.name,
            source_type="structured",
            provenance={"sql": sql, "params": params, "verified": is_verified},
            score=None,
            raw={"columns": columns, "rows": rows},
        )]

    async def health(self) -> tuple[str, str]:
        try:
            await self._execute("SELECT 1", {})
        except Exception as exc:
            return "fail", f"source {self.name}: {exc}"
        return "ok", f"source {self.name} ({self._spec.type}): reachable"

    # ---- execution ----

    async def _execute(self, sql: str, params: dict[str, Any]) -> tuple[list[str], list[tuple]]:
        spec = self._spec
        return await asyncio.to_thread(
            engine.run_query, spec.dsn, spec.type, sql, params,
            row_limit=spec.row_limit, timeout=spec.timeout,
        )

    # ---- generation ----

    def _schema_index(self) -> SchemaIndex:
        if self._index is None:
            spec = self._spec
            self._index = SchemaIndex(
                spec.dsn, table_hints=spec.tables,
                cache_dir=_schema_cache_dir(), cache_tag=_embedding_tag(),
            )
        return self._index

    async def _schema(self, question: str, *, widen: bool = False) -> str:
        """The schema block for generation. On a small schema (or schema_rag=off)
        the whole thing is dumped; on a wide one only the tables relevant to the
        question, plus their foreign-key bridges, are retrieved (schema-RAG). See
        sources/sql/schema.py. Falls back to the full dump if retrieval is empty."""
        spec = self._spec
        index = self._schema_index()
        await index.ensure_cards()
        if spec.schema_rag == "off" or (
            spec.schema_rag == "auto" and index.table_count <= spec.schema_rag_min_tables
        ):
            return index.serialize_all()
        top_k = spec.schema_rag_top_k * (_WIDEN_FACTOR if widen else 1)
        tables = await index.retrieve(question, top_k=top_k)
        if not tables:
            return index.serialize_all()
        self._logger.info(
            "schema-RAG: %d/%d tables for question", len(tables), index.table_count
        )
        return index.serialize(tables)

    async def _generate(
        self, question: str, *, prior_sql: str = "", error: str = "", widen: bool = False
    ) -> str:
        schema = await self._schema(question, widen=widen)
        allow = ", ".join(self._spec.allow).upper()
        desc = f"\nThis database holds: {self._spec.description}" if self._spec.description else ""
        prompt = (
            f"Write ONE {allow} SQL query answering the question, for a "
            f"{self._spec.type} database. Use only the tables and columns below. "
            f"Return SQL only, no prose.{desc}\n\nSCHEMA:\n{schema}\n\n"
            f"QUESTION: {question}"
        )
        if error:
            prompt += (
                f"\n\nYour previous query failed — fix it:\nSQL: {prior_sql}\nERROR: {error}"
            )
        return (await achat_structured(prompt, _GeneratedSQL)).sql

    # ---- verified queries ----

    async def _match_verified(self, question: str) -> dict[str, Any] | None:
        verified = self._spec.verified
        if not verified:
            return None
        embeddings = await self._verified_embeds(verified)
        q = await aembed_text(question, purpose="GENERIC_RETRIEVAL")
        best_name, best_score = None, 0.0
        for name, emb in embeddings.items():
            score = _cosine(q, emb)
            if score > best_score:
                best_name, best_score = name, score
        if best_name is None or best_score < _verified_threshold():
            return None
        self._logger.info("verified-query match: %s (%.2f)", best_name, best_score)
        return {"name": best_name, **verified[best_name]}

    async def _verified_embeds(self, verified: dict[str, Any]) -> dict[str, list[float]]:
        if self._verified_embeddings is None:
            names = list(verified.keys())
            embs = await asyncio.gather(
                *[aembed_text(str(verified[n].get("description") or n)) for n in names]
            )
            self._verified_embeddings = dict(zip(names, embs))
        return self._verified_embeddings

    async def _fill_params(self, question: str, verified: dict[str, Any]) -> dict[str, Any]:
        names = list(verified.get("params") or [])
        if not names:
            return {}
        model = create_model(
            "VerifiedParams",
            **{n: (str, Field(description=f"the value of {n} from the question")) for n in names},
        )
        filled = await achat_structured(
            f"Extract these parameter values from the question: {', '.join(names)}.\n"
            f"QUESTION: {question}", model,
        )
        return {n: getattr(filled, n) for n in names}

    # ---- rendering ----

    def _render(self, columns: list[str], rows: list[tuple]) -> str:
        if not rows:
            return "(no rows)"
        header = " | ".join(columns)
        body = "\n".join(
            " | ".join("" if v is None else str(v) for v in row)
            for row in rows[:_RENDER_ROW_CAP]
        )
        extra = len(rows) - _RENDER_ROW_CAP
        more = f"\n… ({extra} more rows)" if extra > 0 else ""
        return f"{header}\n{body}{more}"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _schema_cache_dir() -> str | None:
    """Where the schema-RAG index persists — a user cache dir, so a wide schema is
    embedded once ever, not per process. Best-effort; None disables persistence."""
    try:
        from pathlib import Path

        return str(Path.home() / ".cache" / "ingestlib" / "schema")
    except Exception:
        return None


def _embedding_tag() -> str:
    """The embedding model's identity — provider AND model id — so the persisted
    index invalidates if either changes. Model matters as much as provider: a
    different model means a different vector space (often a different dimension),
    which would make cached vectors meaningless (and _cosine would silently
    compare truncated vectors)."""
    try:
        from ingestlib.config import get_config

        cfg = get_config()
        model = getattr(getattr(cfg, cfg.embedding_provider, None), "embedding_model_id", "")
        return f"{cfg.embedding_provider}:{model}"
    except Exception:
        return ""
