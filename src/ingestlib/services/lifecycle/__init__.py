"""Lifecycle services — manage the corpus as files change.

    from ingestlib.services import remove, sync

    remove("report.pdf")            # erase one document: vectors + artifacts
    sync("corpus/", prune=True)     # make the corpus match the folder

remove() erases a document from both stores; sync() reconciles a folder
against the corpus (new → ingest, changed → replace, renamed → move,
gone → prune when asked); backfill() rebuilds a vector store from stored
artifacts — embedding time, not pipeline time.
"""
from ingestlib.services.lifecycle.backfiller import abackfill, backfill
from ingestlib.services.lifecycle.models import (
    BackfillResult,
    RemoveResult,
    SyncAction,
    SyncResult,
)
from ingestlib.services.lifecycle.remover import aremove, remove
from ingestlib.services.lifecycle.syncer import async_sync, sync

__all__ = [
    "remove",
    "aremove",
    "RemoveResult",
    "sync",
    "async_sync",
    "SyncResult",
    "SyncAction",
    "backfill",
    "abackfill",
    "BackfillResult",
]
