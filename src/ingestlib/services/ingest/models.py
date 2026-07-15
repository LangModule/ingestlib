"""Data model returned by ingest(): IngestResult."""
from pydantic import BaseModel, ConfigDict, Field


class IngestResult(BaseModel):
    """Outcome of one document's journey through the full pipeline.

    status    — "ingested" (fresh run) or "skipped" (this checksum already
                completed the full pipeline and skip_existing was True)
    doc_id    — the document's content checksum; keys every artifact and vector
    durations — per-stage wall-clock seconds (parse/classify/split/embed/upsert)
    """

    model_config = ConfigDict(frozen=True)

    status: str
    doc_id: str
    filename: str = ""
    category: str = ""
    confidence: float = 0.0
    pages: int = 0
    sections: int = 0
    chunks: int = 0
    vectors: int = 0
    durations: dict[str, float] = Field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        """Wall-clock total across all stages."""
        return sum(self.durations.values())
