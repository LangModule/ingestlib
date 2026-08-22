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
from ingestlib.utils.logger import get_logger


# Cosine to accept a verified-query match. The question embeds as a query
# (GENERIC_RETRIEVAL) and the verified description as a document (GENERIC_INDEX),
# so scores run on the asymmetric scale: on Bedrock Nova an exact-text match
# lands ~0.48 and a confusable but wrong question ~0.24, so 0.35 sits cleanly in
# the gap. (Symmetric embedding scores higher but separates the confusable case
# far worse — 0.65 real vs 0.61 wrong — so the asymmetric pairing is deliberate.)
_VERIFIED_THRESHOLD = 0.35
_RENDER_ROW_CAP = 50        # rows rendered into the prompt-ready content string


class _GeneratedSQL(BaseModel):
    """The model's SQL answer — a single read-only statement, no prose."""
    sql: str = Field(description="one read-only SQL query answering the question")


class SqlSource(Source):
    """A read-only SQL database exposed as a retrieval Source."""

    def __init__(self, spec: SourceSpec) -> None:
        self.name = spec.name
        self._spec = spec
        self._logger = get_logger(f"ingestlib.sources.sql.{spec.name}")
        self._schema_cache: str | None = None
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
            sql = with_limit(guard(await self._generate(question, prior_sql=sql, error=str(exc)),
                                   allow=spec.allow), spec.row_limit)
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

    async def _schema(self) -> str:
        if self._schema_cache is None:
            self._schema_cache = await asyncio.to_thread(self._introspect)
        return self._schema_cache

    def _introspect(self) -> str:
        from sqlalchemy import inspect

        insp = inspect(engine.get_engine(self._spec.dsn))
        lines = []
        for table in insp.get_table_names():
            cols = ", ".join(f"{c['name']} {c['type']}" for c in insp.get_columns(table))
            hint = self._spec.tables.get(table, "")
            lines.append(f"TABLE {table} ({cols})" + (f"  -- {hint}" if hint else ""))
        return "\n".join(lines) or "(no tables)"

    async def _generate(self, question: str, *, prior_sql: str = "", error: str = "") -> str:
        schema = await self._schema()
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
        if best_name is None or best_score < _VERIFIED_THRESHOLD:
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
