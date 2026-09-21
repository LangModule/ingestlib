"""Corpus management commands — the day-1+ surface.

`init` and `doctor` get a stack to green; these manage the documents in it:
ingest / sync files into the corpus, browse it (list / show / collections),
re-sort it after a rules change (recollect), audit durability (verify),
rebuild the index from the registry (reindex), and remove documents. Thin
wrappers over the library services — the work lives there, the CLI adds
argument parsing, a table, progress lines, and exit codes.
"""
from pathlib import Path

from ingestlib.operations.parse.detector import SUPPORTED_EXTENSIONS
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)


def _quiet_library_logging() -> None:
    """The command's printed lines ARE the output — hush INFO chatter unless
    the user asked for it (an explicit INGESTLIB_LOG_LEVEL still wins)."""
    import os

    from ingestlib.utils.logger import configure

    if not os.environ.get("INGESTLIB_LOG_LEVEL"):
        configure(level="WARNING")


def _stage_printer():
    """An on_stage callback that prints one dotted progress line per stage."""
    def on_stage(stage: str, event: str) -> None:
        if event == "done":
            print(f"    · {stage}")
    return on_stage


def _expand(paths: list[str]) -> list[Path]:
    """Files as given; a directory expands to its ingestible files (recursive)."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(
                f for f in p.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ))
        else:
            out.append(p)
    return out


def run_ingest(paths: list[str], namespace: str = "") -> int:
    """`ingestlib ingest <path...>` — index files or folders."""
    _quiet_library_logging()
    from ingestlib.services import ingest

    files = _expand(paths)
    if not files:
        print("no ingestible files found (PDF/DOCX/PPTX/PNG/JPEG/WebP)")
        return 1

    failed = 0
    for f in files:
        print(f"ingest {f}")
        if not f.exists():
            print("  ✗ file not found — check the path")
            failed += 1
            continue
        try:
            result = ingest(f, namespace=namespace, on_stage=_stage_printer())
        except Exception as exc:
            print(f"  ✗ {exc}")
            failed += 1
            continue
        summary = f"  {result.status}: {result.chunks} chunk(s)"
        if result.status == "replaced":
            summary += f" (replaced {result.replaced_doc_id[:12]})"
        print(summary)
    print(f"\n{len(files) - failed}/{len(files)} succeeded")
    return 1 if failed else 0


def run_sync(directory: str, *, prune: bool, dry_run: bool, namespace: str = "") -> int:
    """`ingestlib sync <dir> [--prune] [--dry-run]` — reconcile a folder."""
    _quiet_library_logging()
    from ingestlib.services import sync

    try:
        result = sync(
            directory, namespace=namespace, prune=prune, dry_run=dry_run,
            on_stage=None if dry_run else _stage_printer(),
        )
    except ValueError as exc:
        print(f"✗ {exc}")
        return 1

    verb = "would" if dry_run else ""
    for action in result.actions:
        name = Path(action.path).name
        tail = f" ({action.detail})" if action.detail else ""
        print(f"  {verb + ' ' if verb else ''}{action.action}: {name}{tail}")

    counts = ", ".join(f"{k}={v}" for k, v in sorted(result.counts.items()))
    print(f"\n{'plan' if dry_run else 'sync'}: {counts or 'nothing to do'}")
    if result.errors:
        print(f"{len(result.errors)} file(s) failed — see the error lines above")
        return 1
    return 0


def run_list(namespace: str | None = None) -> int:
    """`ingestlib list` — the registry as a table."""
    _quiet_library_logging()
    from ingestlib.storage import artifacts

    docs = artifacts.list_documents()
    if namespace is not None:
        docs = [d for d in docs if d.namespace == namespace]
    if not docs:
        print("no documents stored")
        return 0

    docs.sort(key=lambda d: (d.namespace, d.filename))
    rows = [("DOC ID", "PAGES", "CHUNKS", "CATEGORY", "NAMESPACE", "SOURCE")]
    for d in docs:
        rows.append((
            d.doc_id[:12], str(d.page_count), str(d.chunks),
            d.category or "-", d.namespace or "-",
            d.source_path or d.filename or "-",
        ))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for i, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)).rstrip())
        if i == 0:
            print("  ".join("-" * w for w in widths))
    print(f"\n{len(docs)} document(s)")
    return 0


def run_show(target: str) -> int:
    """`ingestlib show <path|doc_id>` — everything stored about one document."""
    _quiet_library_logging()
    from ingestlib.services import get_document
    from ingestlib.storage import artifacts

    by_path = artifacts.find_by_path(target)
    doc = get_document(by_path.doc_id) if by_path is not None else get_document(target)
    if doc is None:  # try a doc_id prefix (as `ingestlib list` prints them)
        matches = [m for m in artifacts.list_documents() if m.doc_id.startswith(target)]
        if len(matches) == 1:
            doc = get_document(matches[0].doc_id)
    if doc is None:
        print(f"✗ no document matching {target!r} — `ingestlib list` shows what's stored")
        return 1

    conf = f" ({doc.classify_confidence:.2f})" if doc.classify_confidence is not None else ""
    collection = f"  ·  collection {doc.collection}" if doc.collection else ""
    print(f"doc {doc.doc_id[:12]}  {doc.filename or '?'}")
    print(f"  source     {doc.source_path or '?'}")
    print(f"  namespace  {doc.namespace or '(none)'}")
    print(f"  category   {doc.category or '?'}{conf}{collection}")
    print(f"  status     {doc.status or '?'}")
    print(f"  pages {doc.page_count} · sections {doc.section_count} · chunks {doc.chunk_count}")
    if doc.sections:
        print(f"  sections   {', '.join(s['name'] for s in doc.sections)}")
    for key, value in (doc.source_metadata or {}).items():
        print(f"    {key}: {value}")
    if doc.extractions:
        by_schema: dict[str, int] = {}
        for e in doc.extractions:
            by_schema[e.get("schema_name") or "?"] = by_schema.get(e.get("schema_name") or "?", 0) + 1
        print("  extractions")
        for schema_name, count in sorted(by_schema.items()):
            print(f"    {schema_name}: {count} item(s)")
            for e in doc.extractions:
                if (e.get("schema_name") or "?") == schema_name:
                    print(f"      - {e.get('value')}")
    return 0


def run_collections(namespace: str | None = None) -> int:
    """`ingestlib collections` — the collections in the corpus and their counts."""
    _quiet_library_logging()
    from ingestlib.storage import registry

    rows = registry.list_collections(namespace)
    if not rows:
        print("no collections")
        return 0
    declared = {c["name"]: c for c in registry.collection_rules()}
    for row in rows:
        name = row["collection"] or ""
        rule = declared.get(name)
        badge = ""
        if rule is not None and rule.get("extract_schema"):
            badge = "  [auto-extract]" if rule.get("auto_extract") else "  [schema]"
        print(f"  {(name or '(uncategorized)'):<24} {row['count']}{badge}")
    print(f"\n{len(rows)} collection(s)")
    return 0


def run_verify(namespace: str | None = None, repair: bool = False) -> int:
    """`ingestlib verify` — audit durability (registry vs vector store vs blob store)."""
    _quiet_library_logging()
    from ingestlib.services import verify

    result = verify(namespace=namespace, repair=repair)
    if result.checked == 0:
        print("no documents to verify")
        return 0
    for item in result.drifted:
        name = item.filename or item.doc_id[:12]
        problems = []
        if not item.vectors_ok:
            problems.append(f"vectors: expected {item.expected}, store has {item.actual}")
        if item.missing_blobs:
            problems.append(f"missing blobs: {', '.join(item.missing_blobs)}")
        print(f"  ✗ {name}: {'; '.join(problems)}")
    repaired = [i for i in result.items if i.repaired and i.vectors_ok]
    if repaired:
        print(f"  ↻ re-embedded {len(repaired)} drifted document(s)")
    if result.ok:
        print(f"✓ {result.checked} document(s) verified — durable across all stores")
        return 0
    print(f"\n✗ {len(result.drifted)} of {result.checked} document(s) failed durability")
    return 1


def run_recollect(namespace: str | None = None) -> int:
    """`ingestlib recollect` — re-classify the corpus after rules.yaml changed."""
    _quiet_library_logging()
    from ingestlib.services import recollect

    result = recollect(namespace=namespace)
    if result.recollected == 0:
        print("no documents to recollect")
        return 0
    for item in result.changed:
        name = item.filename or item.doc_id[:12]
        print(f"  {name}: {item.from_category or '(none)'} → {item.to_category or '(none)'}")
    print(
        f"\nrecollected {result.recollected} document(s), "
        f"{len(result.changed)} re-sorted in {result.duration_seconds}s"
    )
    return 0


def run_remove(target: str, namespace: str = "") -> int:
    """`ingestlib remove <path|doc_id>` — erase one document from both stores."""
    _quiet_library_logging()
    from ingestlib.services import remove

    try:
        result = remove(target, namespace=namespace)
    except ValueError as exc:
        print(f"✗ {exc}")
        return 1
    print(
        f"removed {result.filename or result.doc_id[:12]}: "
        f"{result.vectors_deleted} vector(s), {result.artifacts_deleted} artifact object(s)"
    )
    return 0


def run_reindex(namespace: str = "") -> int:
    """`ingestlib reindex` — rebuild the vector store from the registry."""
    _quiet_library_logging()
    from ingestlib.services import reindex

    result = reindex(namespace=namespace)
    print(
        f"reindexed {result.documents} document(s), {result.chunks} chunk(s) "
        f"in {result.duration_seconds}s"
    )
    if result.skipped:
        print(f"{len(result.skipped)} document(s) skipped (no stored split — "
              f"need a real ingest)")
    return 0
