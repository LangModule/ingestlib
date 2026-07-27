"""sync() / async_sync() — reconcile a folder with the corpus.

The folder is the truth; sync makes the stores match it. Phase order is
load-bearing:

    1. scan     — hash every ingestible file under the root (pure, local)
    2. plan     — diff against the registry: what's new, changed, moved,
                  unchanged (pure; this IS the dry_run output)
    3. execute  — per file, through aingest(), which already owns the
                  ingest / replace / move / skip semantics
    4. repair   — two documents claiming one (namespace, path) is the trace
                  of a crashed replace: the newest created_at wins
    5. prune    — LAST, against the post-move registry, and only when asked

Prune is guarded three ways: opt-in (prune=False default); root-scoped —
only documents whose RECORDED path lies under the synced root and whose
file is genuinely gone (a file excluded by the glob still exists, so it is
never pruned); and the empty-scan refusal — a scan that finds nothing while
prune would delete documents aborts instead of obeying (wrong directory,
unmounted volume). Per-file failures become "error" actions and sync
continues — one bad PDF must not abandon the reconcile.

`async` is a Python keyword, so the async form is async_sync() — the one
exception to the a-prefix naming.
"""
import asyncio
import os
import time
from collections import defaultdict
from pathlib import Path

from ingestlib.operations.parse.detector import SUPPORTED_EXTENSIONS
from ingestlib.services.ingest.ingestor import StageCallback, aingest
from ingestlib.services.lifecycle.models import SyncAction, SyncResult
from ingestlib.services.lifecycle.remover import aremove
from ingestlib.storage import VectorStore, artifacts
from ingestlib.storage.artifacts import DocumentMeta
from ingestlib.utils.files import sha256_of_file
from ingestlib.utils.logger import get_logger
from ingestlib.utils.sync import run_sync


logger = get_logger(__name__)

_STATUS_TO_ACTION = {
    "ingested": "ingest",
    "replaced": "replace",
    "moved": "move",
    "skipped": "skip",
}


def _scan(root: Path, glob: str) -> list[Path]:
    """Every ingestible file under root matching the glob, sorted."""
    return sorted(
        p for p in root.glob(glob)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _registry(namespace: str) -> list[DocumentMeta]:
    """Documents in this namespace that carry a logical identity."""
    return [
        m for m in artifacts.list_documents()
        if m.namespace == namespace and m.source_path
    ]


def _under(root: Path, path_str: str) -> bool:
    return path_str.startswith(str(root.resolve()) + os.sep)


def _duplicate_losers(registry: list[DocumentMeta]) -> list[tuple[DocumentMeta, str]]:
    """(loser, winner_doc_id) for every path claimed more than once —
    the newest created_at keeps the path."""
    groups: dict[str, list[DocumentMeta]] = defaultdict(list)
    for m in registry:
        groups[m.source_path].append(m)
    losers: list[tuple[DocumentMeta, str]] = []
    for group in groups.values():
        if len(group) > 1:
            group.sort(key=lambda m: m.created_at)
            losers.extend((m, group[-1].doc_id) for m in group[:-1])
    return losers


def _prune_candidates(
    registry: list[DocumentMeta], root: Path, exclude_ids: frozenset[str] = frozenset()
) -> list[DocumentMeta]:
    """Documents recorded under the root whose file is genuinely gone.

    The is_file() check — not scan membership — is what makes a narrow glob
    safe: a file the glob skipped still exists and is never pruned.
    """
    return [
        m for m in registry
        if _under(root, m.source_path)
        and not Path(m.source_path).is_file()
        and m.doc_id not in exclude_ids
    ]


def _plan(
    files: list[Path], namespace: str
) -> tuple[list[SyncAction], list[SyncAction], frozenset[str]]:
    """Pure diff of folder vs registry → (per-file plan, repair plan,
    doc_ids accounted for by moves/repairs — excluded from prune)."""
    registry = _registry(namespace)
    by_id = {m.doc_id: m for m in registry}
    newest_by_path: dict[str, DocumentMeta] = {}
    for m in registry:
        held = newest_by_path.get(m.source_path)
        if held is None or m.created_at > held.created_at:
            newest_by_path[m.source_path] = m

    plans: list[SyncAction] = []
    accounted: set[str] = set()
    for f in files:
        resolved = str(f.resolve())
        checksum = sha256_of_file(f)
        stored = by_id.get(checksum)
        if stored is not None:
            if stored.source_path == resolved:
                if artifacts.ingest_complete(checksum):
                    action, detail = "skip", ""
                else:
                    action, detail = "ingest", "retrying incomplete ingest"
            else:
                action, detail = "move", f"from {stored.source_path}"
                accounted.add(checksum)
        else:
            prior = newest_by_path.get(resolved)
            if prior is not None:
                action, detail = "replace", prior.doc_id
            else:
                action, detail = "ingest", ""
        plans.append(SyncAction(path=str(f), action=action, doc_id=checksum, detail=detail))

    repairs = [
        SyncAction(path=loser.source_path, action="repair", doc_id=loser.doc_id,
                   detail=f"older duplicate of {winner[:12]}")
        for loser, winner in _duplicate_losers(registry)
    ]
    accounted.update(a.doc_id for a in repairs)
    return plans, repairs, frozenset(accounted)


async def async_sync(
    directory: Path | str,
    *,
    namespace: str = "",
    prune: bool = False,
    dry_run: bool = False,
    glob: str = "**/*",
    store: VectorStore | None = None,
    on_stage: StageCallback | None = None,
) -> SyncResult:
    """Make the corpus match a folder (async).

    directory — the folder that is the source of truth
    namespace — the corpus partition to reconcile against
    prune     — also DELETE documents whose recorded file is gone (opt-in,
                root-scoped; see the module docstring for the guardrails)
    dry_run   — return the full plan, execute nothing
    glob      — which files count, relative to directory ("**/*.pdf", ...)
    store     — vector store connector; defaults to config.yaml's selection
    on_stage  — forwarded to each document's ingest for per-stage progress
    """
    t0 = time.perf_counter()
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"sync target {root} is not a directory")

    files = await asyncio.to_thread(_scan, root, glob)
    plans, repairs, accounted = await asyncio.to_thread(_plan, files, namespace)

    if prune and not files and not dry_run:  # dry_run may inspect the plan
        doomed = await asyncio.to_thread(
            _prune_candidates, _registry(namespace), root, accounted
        )
        if doomed:
            raise ValueError(
                f"sync found NO ingestible files under {root} while prune=True "
                f"would delete {len(doomed)} document(s) — refusing. Wrong "
                f"directory, unmounted volume, or a too-narrow glob? Run with "
                f"dry_run=True to inspect the plan."
            )

    actions: list[SyncAction] = []

    if dry_run:
        actions.extend(plans)
        actions.extend(repairs)
        if prune:
            registry = await asyncio.to_thread(_registry, namespace)
            actions.extend(
                SyncAction(path=m.source_path, action="prune", doc_id=m.doc_id,
                           detail=m.filename)
                for m in _prune_candidates(registry, root, accounted)
            )
        return SyncResult(
            directory=str(root), namespace=namespace, dry_run=True,
            actions=actions, duration_seconds=round(time.perf_counter() - t0, 2),
        )

    # 3. execute — aingest owns the per-document semantics
    for planned in plans:
        if planned.action == "skip":
            actions.append(planned)
            continue
        try:
            result = await aingest(
                Path(planned.path), store=store, namespace=namespace,
                on_stage=on_stage,
            )
            actions.append(SyncAction(
                path=planned.path,
                action=_STATUS_TO_ACTION[result.status],
                doc_id=result.doc_id,
                detail=result.replaced_doc_id,
            ))
        except Exception as exc:
            logger.warning("sync: %s failed — continuing: %s", planned.path, exc)
            actions.append(SyncAction(path=planned.path, action="error", detail=str(exc)))

    # 4. repair — recomputed on the post-execution registry
    registry = await asyncio.to_thread(_registry, namespace)
    for loser, winner in _duplicate_losers(registry):
        try:
            await aremove(loser.doc_id, namespace=namespace, store=store)
            actions.append(SyncAction(
                path=loser.source_path, action="repair", doc_id=loser.doc_id,
                detail=f"older duplicate of {winner[:12]}",
            ))
        except Exception as exc:
            logger.warning("sync: repair of %s failed: %s", loser.doc_id[:12], exc)
            actions.append(SyncAction(
                path=loser.source_path, action="error", doc_id=loser.doc_id,
                detail=str(exc),
            ))

    # 5. prune — LAST, so moves and repairs have already settled the registry
    if prune:
        registry = await asyncio.to_thread(_registry, namespace)
        for meta in _prune_candidates(registry, root):
            try:
                await aremove(meta.doc_id, namespace=namespace, store=store)
                actions.append(SyncAction(
                    path=meta.source_path, action="prune", doc_id=meta.doc_id,
                    detail=meta.filename,
                ))
            except Exception as exc:
                logger.warning("sync: prune of %s failed: %s", meta.doc_id[:12], exc)
                actions.append(SyncAction(
                    path=meta.source_path, action="error", doc_id=meta.doc_id,
                    detail=str(exc),
                ))

    result = SyncResult(
        directory=str(root), namespace=namespace, dry_run=False,
        actions=actions, duration_seconds=round(time.perf_counter() - t0, 2),
    )
    logger.info("sync done: %s — %s", root, result.counts or "nothing to do")
    return result


def sync(
    directory: Path | str,
    *,
    namespace: str = "",
    prune: bool = False,
    dry_run: bool = False,
    glob: str = "**/*",
    store: VectorStore | None = None,
    on_stage: StageCallback | None = None,
) -> SyncResult:
    """Make the corpus match a folder. Sync wrapper — use async_sync()
    inside an event loop (`async` is a keyword, hence the one naming
    exception to the a-prefix convention)."""
    return run_sync(
        async_sync(
            directory, namespace=namespace, prune=prune, dry_run=dry_run,
            glob=glob, store=store, on_stage=on_stage,
        ),
        "async_sync",
    )
