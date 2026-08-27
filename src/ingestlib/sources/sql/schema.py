"""Schema-RAG — retrieve the relevant slice of a wide schema before generating SQL.

On a small, clean schema, dumping every table into the prompt is fine (and, per
E-SQL/Maamari, pruning can even hurt a strong model), so SqlSource keeps doing
that. On a wide schema it breaks two ways: the prompt grows without bound, and
ambiguous columns spread across dozens of tables lead the model to the wrong join
(measured — a "total revenue" query used the right price column at 6 tables and a
spurious one at 70). This module is the fix. It builds one embedded "card" per
table — name, typed columns, PK/FK markers, sampled values, and any plain-English
hint — then for a question retrieves the most similar tables and walks the
foreign-key graph to pull in the bridge tables that make the joins possible.

Recall-first by design: omitting a needed table is unrecoverable, while extra
tables are noise the model can ignore (SchemaGraphSQL, LinkAlign). Two stages —
cheap embedding retrieval, then FK-graph closure — are exactly the shape the 2025
literature converged on.

DB-agnostic: introspection is SQLAlchemy `inspect()`, so every dialect the engine
speaks is covered; sampling quotes identifiers via the dialect's preparer. Cards
embed as documents (the aembed_text default, GENERIC_INDEX) and the question as a
query (GENERIC_RETRIEVAL) — the same asymmetric pairing the verified-match path in
source.py is calibrated on. No vector store: the card vectors live in memory on the
index, built once per source and reused (mirrors SqlSource's schema cache).

Two hardening measures make the build survive a wide schema on a rate-limited
provider (a 120-card burst throttles Bedrock outright):
  - Persistence — a fingerprint-keyed on-disk cache (cache_dir) so the one-time
    embed cost is paid once ever, not per process; the fingerprint invalidates it
    when the schema (or embedding provider) changes.
  - Incremental, paced, graceful build — embed only the cards not already cached,
    paced on a large schema, persisting after each batch; a card whose embed fails
    is skipped (it can still be pulled in as an FK bridge) and retried next build,
    so throttling degrades to "slower / one table short", never a crash.
And inferred join edges — when a schema declares no foreign keys (common in the
enterprise), edges are inferred from column/PK naming so FK closure still connects
tables. See _infer_fks.
"""
import asyncio
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, Iterable

from ingestlib.foundations.llm import aembed_text
from ingestlib.sources.sql import engine
from ingestlib.utils.logger import get_logger

logger = get_logger(__name__)

# An embedding function: (text, purpose) -> vector. Injectable so tests can stub
# it deterministically without a model; defaults to foundations aembed_text.
EmbedFn = Callable[..., Awaitable[list[float]]]

# Card-embed concurrency. Small schemas burst; a wide schema (> _PACE_THRESHOLD
# tables) drops to a gentle rate with a pause between calls, because a large burst
# throttles rate-limited providers (measured on Bedrock). Persistence + graceful
# skip mean even a throttled build makes progress and self-heals on the next run.
_EMBED_CONCURRENCY = 8
_PACE_THRESHOLD = 32
_BIG_CONCURRENCY = 2
_EMBED_PACE = 0.5          # seconds between embeds on a wide schema
_CACHE_BATCH = 16          # persist the index every N newly-embedded cards


@dataclass
class TableCard:
    """One table's schema-RAG card — the unit that is embedded and serialized.

    render() is the M-Schema-style text used for BOTH retrieval (what we embed)
    and generation (what we feed the model): the table name, each column with its
    type and PK/FK markers and a few example values, plus the plain-English hint.
    XiYan-SQL found this semi-structured form (types + keys + sample values) beats
    a raw DDL dump for LLM understanding.
    """
    name: str
    columns: list[tuple[str, str]]                        # (column name, type)
    pk: set[str] = field(default_factory=set)             # primary-key column names
    fks: list[tuple[str, str, str]] = field(default_factory=list)  # (local, ref_table, ref_col)
    inferred_fks: list[tuple[str, str, str]] = field(default_factory=list)  # inferred by naming
    description: str = ""                                  # the sources.yaml `tables` hint
    samples: dict[str, list[str]] = field(default_factory=dict)    # {column: example values}

    def render(self) -> str:
        lines = [f"TABLE {self.name}"]
        # declared FKs win; inferred ones fill in only where no FK is declared
        fk_by_col: dict[str, tuple[str, str, bool]] = {
            local: (rt, rc, False) for local, rt, rc in self.fks
        }
        for local, rt, rc in self.inferred_fks:
            fk_by_col.setdefault(local, (rt, rc, True))
        for col, typ in self.columns:
            marks = []
            if col in self.pk:
                marks.append("PK")
            if col in fk_by_col:
                rt, rc, inferred = fk_by_col[col]
                marks.append(f"-> {rt}.{rc}" + (" (inferred)" if inferred else ""))
            suffix = ("  " + " ".join(marks)) if marks else ""
            ex = self.samples.get(col)
            if ex:
                suffix += "  e.g. " + ", ".join(f"'{v}'" for v in ex)
            lines.append(f"  {col} {typ}{suffix}")
        if self.description:
            lines.append(f"  -- {self.description}")
        return "\n".join(lines)


class SchemaIndex:
    """An in-memory, embedded index of a SQL source's tables for schema-RAG.

    build() introspects the schema into cards and embeds each once; retrieve()
    ranks cards against a question and returns the connected table set (top-k +
    FK closure); serialize() renders that set as the schema block for the prompt.
    Built lazily and cached on the owning SqlSource, disposed when engines reset.
    """

    def __init__(
        self,
        dsn: str,
        *,
        table_hints: dict[str, str] | None = None,
        embed: EmbedFn | None = None,
        sample_rows: int = 3,
        max_bridge_hops: int = 2,
        cache_dir: str | Path | None = None,
        cache_tag: str = "",
        embed_pace: float = _EMBED_PACE,
    ) -> None:
        self._dsn = dsn
        self._hints = table_hints or {}
        self._embed = embed or aembed_text
        self._sample_rows = sample_rows
        self._max_bridge_hops = max_bridge_hops
        self._embed_pace = embed_pace         # inter-embed delay on a wide schema; 0 for local/stub
        self._cache_dir = Path(cache_dir) if cache_dir else None  # None → no persistence
        self._cache_tag = cache_tag                               # embedding-model identity
        self._cards: dict[str, TableCard] = {}
        self._vectors: dict[str, list[float]] = {}
        self._graph: dict[str, set[str]] = {}
        self._built = False

    # ---- build ----

    async def ensure_cards(self) -> None:
        """Introspect the schema into cards once, WITHOUT embedding. Cheap enough
        to run on every path (the "off" and small-schema cases never embed), and
        idempotent. table_count and serialize_all() are usable after this."""
        if self._cards:
            return
        self._cards = await asyncio.to_thread(self._introspect_cards)
        self._graph = _fk_graph(self._cards)

    async def build(self) -> None:
        """ensure_cards + embed each card once, then cache to disk. Incremental:
        loads whatever is already cached for this schema fingerprint and embeds
        only the rest, persisting after each batch — so a throttled build resumes
        instead of restarting. Paced + graceful on a wide schema: a card whose
        embed fails is skipped (it can still arrive via FK closure) and retried on
        the next build. A no-op once fully built in this process."""
        if self._built:
            return
        await self.ensure_cards()
        fingerprint = self._fingerprint()
        self._vectors = self._load_cache(fingerprint)
        missing = [n for n in self._cards if n not in self._vectors]
        if missing:
            big = len(missing) > _PACE_THRESHOLD
            sem = asyncio.Semaphore(_BIG_CONCURRENCY if big else _EMBED_CONCURRENCY)
            pace = self._embed_pace if big else 0.0

            async def embed(name: str) -> tuple[str, list[float] | None]:
                async with sem:
                    try:
                        vec = await self._embed(self._cards[name].render())  # documents
                        if pace:
                            await asyncio.sleep(pace)
                        return name, vec
                    except Exception as exc:  # throttle/transient — skip, retry next build
                        logger.warning(
                            "schema card embed failed for %r (skipped, retried next "
                            "build): %s", name, exc,
                        )
                        return name, None

            for start in range(0, len(missing), _CACHE_BATCH):
                chunk = missing[start:start + _CACHE_BATCH]
                for name, vec in await asyncio.gather(*(embed(n) for n in chunk)):
                    if vec is not None:
                        self._vectors[name] = vec
                self._save_cache(fingerprint)
        self._built = True

    @property
    def table_count(self) -> int:
        return len(self._cards)

    def serialize_all(self) -> str:
        """The full schema as M-Schema — every table, in introspection order.
        Requires ensure_cards()."""
        return self.serialize(self._cards.keys())

    def _introspect_cards(self) -> dict[str, TableCard]:
        from sqlalchemy import inspect

        eng = engine.get_engine(self._dsn)
        insp = inspect(eng)
        cards: dict[str, TableCard] = {}
        for table in insp.get_table_names():
            cols = [(c["name"], str(c["type"])) for c in insp.get_columns(table)]
            try:
                pk = set(insp.get_pk_constraint(table).get("constrained_columns") or [])
            except Exception:  # not all dialects report a PK constraint uniformly
                pk = set()
            fks: list[tuple[str, str, str]] = []
            try:
                for fk in insp.get_foreign_keys(table):
                    ref = fk.get("referred_table")
                    for lc, rc in zip(
                        fk.get("constrained_columns") or [], fk.get("referred_columns") or []
                    ):
                        if ref:
                            fks.append((lc, ref, rc))
            except Exception:  # FK reflection is best-effort; retrieval still works without it
                pass
            samples = self._sample(eng, table, [c for c, _ in cols]) if self._sample_rows else {}
            cards[table] = TableCard(
                name=table, columns=cols, pk=pk, fks=fks,
                description=self._hints.get(table, ""), samples=samples,
            )
        # infer join edges from naming where FKs are undeclared (common in the wild)
        for table, edges in _infer_fks(cards).items():
            cards[table].inferred_fks = edges
        return cards

    def _sample(self, eng, table: str, columns: list[str]) -> dict[str, list[str]]:
        """A few example values per column, best-effort. Dialect-safe: the
        identifier is quoted by the dialect's preparer and every supported backend
        (postgres/mysql/sqlite/duckdb/snowflake) understands `LIMIT n`. Any failure
        degrades to no samples — cards stay useful, just without example values."""
        from sqlalchemy import text

        qname = eng.dialect.identifier_preparer.quote(table)
        try:
            with eng.connect() as conn:
                rows = conn.execute(
                    text(f"SELECT * FROM {qname} LIMIT {int(self._sample_rows)}")
                ).fetchall()
        except Exception:
            return {}
        out: dict[str, list[str]] = {c: [] for c in columns}
        for row in rows:
            for col, val in zip(columns, row):
                if val is None or len(out[col]) >= self._sample_rows:
                    continue
                s = str(val)
                if len(s) > 40:
                    s = s[:37] + "..."
                if s not in out[col]:
                    out[col].append(s)
        return {c: vs for c, vs in out.items() if vs}

    # ---- retrieve ----

    async def retrieve(self, question: str, *, top_k: int) -> set[str]:
        """The table set for a question: the top-k most similar by embedding, then
        FK-graph closure so the result is join-complete. Recall-first — closure
        only ever adds bridge tables, never drops a retrieved one."""
        await self.build()
        if not self._vectors:
            return set()
        q = await self._embed(question, purpose="GENERIC_RETRIEVAL")
        ranked = sorted(
            self._vectors, key=lambda n: _cosine(q, self._vectors[n]), reverse=True
        )
        return self.fk_closure(set(ranked[:top_k]))

    def fk_closure(self, selected: Iterable[str]) -> set[str]:
        """Add the bridge tables that connect the selected tables along the FK
        graph, so any two picked tables can actually be joined. Bridges longer
        than max_bridge_hops intermediate tables are skipped, bounding how far a
        hub table can drag unrelated tables into the prompt."""
        picks = [t for t in selected if t in self._graph]
        added: set[str] = set()
        for i in range(len(picks)):
            for j in range(i + 1, len(picks)):
                bridge = _bridge(self._graph, picks[i], picks[j], self._max_bridge_hops)
                if bridge:
                    added.update(bridge)
        return set(selected) | added

    # ---- serialize ----

    def serialize(self, tables: Iterable[str]) -> str:
        """Render the given tables as the schema block, in stable introspection
        order (not retrieval-score order, so the prompt is deterministic)."""
        want = set(tables)
        picked = [c for name, c in self._cards.items() if name in want]
        return "\n".join(c.render() for c in picked) or "(no tables)"

    # ---- persistence ----

    def _fingerprint(self) -> str:
        """Identity of the current schema + embedding model. Changing a table,
        column, or the embedding provider changes it, invalidating the cache."""
        parts = []
        for table in sorted(self._cards):
            cols = ",".join(f"{c}:{t}" for c, t in self._cards[table].columns)
            parts.append(f"{table}({cols})")
        raw = "|".join(parts) + f"|tag={self._cache_tag}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _cache_path(self) -> Path | None:
        if not self._cache_dir:
            return None
        # the DSN carries credentials — hash it, never put it in a filename
        stem = hashlib.sha256(self._dsn.encode()).hexdigest()[:16]
        return self._cache_dir / f"schema-{stem}.json"

    def _load_cache(self, fingerprint: str) -> dict[str, list[float]]:
        """Cached vectors for the current fingerprint (possibly a partial set from
        an earlier throttled build); empty on any miss/mismatch/error."""
        path = self._cache_path()
        if not path or not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception:
            return {}
        if data.get("fingerprint") != fingerprint:
            return {}
        return {k: v for k, v in (data.get("vectors") or {}).items() if k in self._cards}

    def _save_cache(self, fingerprint: str) -> None:
        path = self._cache_path()
        if not path:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"fingerprint": fingerprint, "vectors": self._vectors}))
        except Exception as exc:  # a read-only cache dir must never break retrieval
            logger.debug("could not persist schema index to %s: %s", path, exc)


# ---- graph helpers ----

def _fk_graph(cards: dict[str, TableCard]) -> dict[str, set[str]]:
    """Undirected adjacency over tables, one edge per foreign key (declared or
    inferred), both ways — so closure connects tables even on a schema that
    declares no foreign keys at all."""
    graph: dict[str, set[str]] = {name: set() for name in cards}
    for name, card in cards.items():
        for _, ref_table, _ in (*card.fks, *card.inferred_fks):
            if ref_table in graph:
                graph[name].add(ref_table)
                graph[ref_table].add(name)
    return graph


def _infer_fks(cards: dict[str, TableCard]) -> dict[str, list[tuple[str, str, str]]]:
    """Infer join edges from naming when foreign keys are undeclared. Two
    conservative, low-false-positive conventions:

      1. a column whose name equals another table's (unique) primary-key column —
         e.g. orders.customer_id when customers' PK is customer_id;
      2. a `<prefix>_id` column whose prefix names a table (singular or plural)
         with a single-column PK — e.g. orders.customer_id → customers(id).

    Only columns without a declared FK (and that are not the table's own PK) are
    considered, so declared foreign keys always take precedence.
    """
    # unique PK-column-name → owning table (drop names that are a PK in >1 table)
    pk_owner: dict[str, str] = {}
    ambiguous: set[str] = set()
    for table, card in cards.items():
        for pk_col in card.pk:
            if pk_col in pk_owner:
                ambiguous.add(pk_col)
            pk_owner[pk_col] = table
    for name in ambiguous:
        pk_owner.pop(name, None)

    names = set(cards)

    def table_for_prefix(prefix: str) -> str | None:
        for cand in (prefix, prefix + "s", prefix + "es",
                     (prefix[:-1] + "ies") if prefix.endswith("y") else ""):
            if cand and cand in names:
                return cand
        return None

    inferred: dict[str, list[tuple[str, str, str]]] = {t: [] for t in cards}
    for table, card in cards.items():
        declared = {local for local, _, _ in card.fks}
        for col, _ in card.columns:
            if col in card.pk or col in declared:
                continue
            target = pk_owner.get(col)                       # convention 1
            ref_col = col
            if target is None and col.endswith("_id"):       # convention 2
                cand = table_for_prefix(col[:-3])
                if cand and cand != table:
                    cand_pk = cards[cand].pk
                    if len(cand_pk) == 1:
                        target, ref_col = cand, next(iter(cand_pk))
            if target and target != table:
                inferred[table].append((col, target, ref_col))
    return inferred


def _bridge(
    graph: dict[str, set[str]], a: str, b: str, max_hops: int
) -> list[str] | None:
    """Intermediate tables on a shortest a-b path (at most max_hops of them), or
    None if unreachable within the bound. [] means a and b are the same or already
    directly joinable — no bridge needed. BFS, so the first hit is shortest."""
    if a == b or b in graph.get(a, ()):
        return []
    seen = {a}
    queue: deque[tuple[str, list[str]]] = deque([(a, [])])
    while queue:
        node, mids = queue.popleft()
        for neighbor in graph.get(node, ()):
            if neighbor == b:
                return mids
            if neighbor not in seen and len(mids) < max_hops:
                seen.add(neighbor)
                queue.append((neighbor, mids + [neighbor]))
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
