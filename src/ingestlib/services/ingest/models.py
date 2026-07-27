"""Data model returned by ingest(): IngestResult."""
from pydantic import BaseModel, ConfigDict, Field


class IngestResult(BaseModel):
    """Outcome of one document's journey through the full pipeline.

    status    — "ingested"  fresh run
                "skipped"   this checksum already completed the full pipeline
                            (skip_existing was True)
                "moved"     same checksum arrived from a new path — only the
                            registry's source_path was re-pointed, nothing ran
                "replaced"  a previous version held this source path; it was
                            fully deleted (vectors + artifacts) after the new
                            version went live — see replaced_doc_id
    doc_id    — the document's content checksum; keys every artifact and vector
    replaced_doc_id — the old version's doc_id when status is "replaced"
    durations — per-stage wall-clock seconds (parse/classify/split/embed/
                upsert, plus replace when an old version was deleted)
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
    replaced_doc_id: str = ""
    durations: dict[str, float] = Field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        """Wall-clock total across all stages."""
        return sum(self.durations.values())
