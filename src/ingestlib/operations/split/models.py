"""Data models returned by split(): Chunk, Section, VocabEntry, and SplitResult.

Frozen Pydantic v2 models, matching the parse and classify conventions.
"""
from pydantic import BaseModel, ConfigDict, Field, computed_field


class Chunk(BaseModel):
    """One natural retrieval unit — the thing the embedding phase embeds.

    chunk_id       — document-wide index in reading order
    section        — name of the section this chunk belongs to
    heading        — topic label for this chunk (from the boundary pass)
    text           — plain-text content
    markdown       — markdown content (tables as HTML, figures as references)
    embedding_text — markdown prefixed with its context breadcrumb
                     "[category › section › heading]" — embed THIS field
    pages          — 1-indexed page numbers this chunk spans
    region_ids     — {page_num: [region_id, ...]} provenance back to parse
                     regions (empty when split ran standalone without a parse)
    kind           — dominant content type: text | table | figure | mixed
    token_estimate — rough size (chars/4) for embedding-batch planning
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: int
    section: str
    heading: str = ""
    text: str
    markdown: str
    embedding_text: str
    pages: list[int]
    region_ids: dict[int, list[int]] = Field(default_factory=dict)
    kind: str = "text"
    token_estimate: int = 0


class VocabEntry(BaseModel):
    """One discovered section category."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""


class Section(BaseModel):
    """Consecutive pages sharing one category, containing its natural chunks."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    pages: list[int]
    text: str = ""
    markdown: str = ""
    chunks: list[Chunk] = Field(default_factory=list)


class SplitResult(BaseModel):
    """Full split output — sections in document order, each with its chunks.

    vocabulary — the section categories Pass 1 discovered for this document
    pages_used — pages actually processed (caps at 500)
    """

    model_config = ConfigDict(frozen=True)

    sections: list[Section]
    vocabulary: list[VocabEntry] = Field(default_factory=list)
    pages_used: int = 0

    @computed_field
    @property
    def chunks(self) -> list[Chunk]:
        """Every chunk in document order — the list the embedding phase iterates."""
        return [c for s in self.sections for c in s.chunks]

    @computed_field
    @property
    def section_names(self) -> list[str]:
        return [s.name for s in self.sections]

    def section_by_name(self, name: str) -> Section:
        """First section with this name. Raises KeyError if absent."""
        for s in self.sections:
            if s.name == name:
                return s
        raise KeyError(f"section {name!r} not found (have {self.section_names})")
