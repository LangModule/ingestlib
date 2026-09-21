"""get_document(doc_id) — read one stored document back out of the corpus.

The registry holds the structure (pages/regions/sections/chunks/extractions);
the blob store holds the bytes (source, page renders, figure crops, whole-doc
markdown). This assembles the registry view and offers lazy blob access, so a
search hit's citation (doc·page·section) can be resolved to the real content.
"""
import asyncio
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ingestlib.storage import artifacts, registry


class StoredDocument(BaseModel):
    """One document as the registry knows it, with lazy access to its blobs.

    pages/sections/chunks/regions/extractions are the registry rows (plain
    dicts). markdown() and page_image() fetch bytes from the blob store on
    demand — they are NOT loaded eagerly.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str = ""
    source_path: str = ""
    namespace: str = ""
    source_format: str = ""
    category: str = ""
    collection: str = ""
    status: str = ""
    classify_confidence: float | None = None
    page_count: int = 0
    section_count: int = 0
    chunk_count: int = 0
    created_at: str | None = None
    source_metadata: dict[str, Any] | None = None
    pages: list[dict[str, Any]] = Field(default_factory=list)
    sections: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    regions: list[dict[str, Any]] = Field(default_factory=list)
    extractions: list[dict[str, Any]] = Field(default_factory=list)

    def markdown(self) -> str | None:
        """The whole-document markdown from the blob store (None if absent)."""
        return artifacts.document_markdown(self.doc_id)

    def page_image(self, page_num: int) -> bytes:
        """The rendered PNG for one page (raw bytes) from the blob store."""
        return artifacts.read_blob(artifacts.page_image_key(self.doc_id, page_num))


def _assemble(data: dict[str, Any]) -> StoredDocument:
    doc = data["document"]
    created = doc.get("created_at")
    return StoredDocument(
        doc_id=doc["doc_id"],
        filename=doc.get("filename", ""),
        source_path=doc.get("source_path", ""),
        namespace=doc.get("namespace", ""),
        source_format=doc.get("source_format", ""),
        category=doc.get("category", ""),
        collection=doc.get("collection", ""),
        status=doc.get("status", ""),
        classify_confidence=doc.get("classify_confidence"),
        page_count=doc.get("page_count", 0),
        section_count=doc.get("section_count", 0),
        chunk_count=doc.get("chunk_count", 0),
        created_at=created.isoformat() if created is not None else None,
        source_metadata=doc.get("source_metadata"),
        pages=data["pages"],
        sections=data["sections"],
        chunks=data["chunks"],
        regions=data["regions"],
        extractions=data["extractions"],
    )


def get_document(doc_id: str) -> StoredDocument | None:
    """The live stored document for doc_id, or None if unknown / a tombstoned
    replacement (whose bytes are gone; the row survives only for lineage)."""
    data = registry.get_document(doc_id)
    if data is None or data["document"].get("status") == "replaced":
        return None
    return _assemble(data)


async def aget_document(doc_id: str) -> StoredDocument | None:
    """Async get_document() — runs the registry read in a worker thread."""
    return await asyncio.to_thread(get_document, doc_id)
