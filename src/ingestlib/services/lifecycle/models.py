"""Data models returned by the lifecycle services: RemoveResult, SyncResult."""
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field


class RemoveResult(BaseModel):
    """Outcome of remove() — one document erased from both stores.

    vectors_deleted   — vectors removed from the vector store (0 when the
                        document was parsed but never ingested)
    artifacts_deleted — objects removed from the artifact store
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str = ""
    vectors_deleted: int = 0
    artifacts_deleted: int = 0


class ReindexResult(BaseModel):
    """Outcome of reindex() — a vector store rebuilt from the registry.

    skipped — doc_ids that had no stored split (parsed but never split;
              they need a real ingest, not a reindex)
    """

    model_config = ConfigDict(frozen=True)

    documents: int = 0
    chunks: int = 0
    skipped: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0


class RecollectItem(BaseModel):
    """One document whose collection changed when the rules were re-applied."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str = ""
    from_category: str = ""
    to_category: str = ""


class RecollectResult(BaseModel):
    """Outcome of recollect() — the corpus re-sorted into collections (no OCR).

    recollected — documents re-classified; changed — those whose category moved.
    """

    model_config = ConfigDict(frozen=True)

    recollected: int = 0
    changed: list["RecollectItem"] = Field(default_factory=list)
    duration_seconds: float = 0.0


class SyncAction(BaseModel):
    """One decision sync() made (or, under dry_run, would make) for one
    path or document.

    action — ingest | replace | move | skip | prune | repair | error
    detail — extra context: the replaced doc_id, the error message, ...
    """

    model_config = ConfigDict(frozen=True)

    path: str
    action: str
    doc_id: str = ""
    detail: str = ""


class SyncResult(BaseModel):
    """Outcome of sync() — the folder and the corpus reconciled.

    dry_run=True means actions is the PLAN: nothing was executed.
    """

    model_config = ConfigDict(frozen=True)

    directory: str
    namespace: str = ""
    dry_run: bool = False
    actions: list[SyncAction] = Field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def counts(self) -> dict[str, int]:
        """Actions tallied by kind, e.g. {'ingest': 3, 'skip': 12}."""
        return dict(Counter(a.action for a in self.actions))

    @property
    def errors(self) -> list[SyncAction]:
        """The per-file failures (sync continues past them)."""
        return [a for a in self.actions if a.action == "error"]
