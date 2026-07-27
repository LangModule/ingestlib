"""Corpus management commands — the day-1+ surface.

`init` and `doctor` get a stack to green; these manage the documents in it:
ingest files or folders, sync a folder against the corpus, list what's
stored, and remove documents. Thin wrappers over the library services — the
work lives there, the CLI adds argument parsing, a table, progress lines,
and exit codes.
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


def run_backfill(namespace: str = "") -> int:
    """`ingestlib backfill` — rebuild the vector store from stored artifacts."""
    _quiet_library_logging()
    from ingestlib.services import backfill

    result = backfill(namespace=namespace)
    print(
        f"backfilled {result.documents} document(s), {result.chunks} chunk(s) "
        f"in {result.duration_seconds}s"
    )
    if result.skipped:
        print(f"{len(result.skipped)} document(s) skipped (no split artifact — "
              f"need a real ingest)")
    return 0
