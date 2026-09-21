"""Artifact store — a document's BYTES, keyed by content checksum.

Lives on the backend `artifact_store` selects in config.yaml: an S3 bucket
(durable, shareable) or a plain local folder (zero cloud). Same layout on
both, everything under one prefix per document:

    documents/{doc_id}/
    ├── source/{filename}                     original file, exact bytes
    ├── parse/document.md                     whole-document markdown
    ├── parse/pages/page_0001.png ...         page renders
    └── parse/figures/{fig.filename} ...      figure/chart crops

The blob store holds bytes only; every queryable field — parse structure,
classify, split, extractions, lifecycle — lives in the Postgres registry (see
storage.registry). doc_id is the parse checksum, so re-saving the same file
overwrites in place. The citation chain needs no extra lookup here: a vector
hit's {doc_id, pages} resolves to page images straight from this layout.
"""
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ingestlib.operations.parse.models import ParseResult
from ingestlib.operations.split.models import SplitResult
from ingestlib.storage.blobs import get_blob_store
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_PREFIX = "documents"


class DocumentMeta(BaseModel):
    """Lightweight per-document view — the registry documents-row projected onto
    the shape lifecycle code expects (logical identity + counts).

    Assembled from the registry by _meta_from_row; not persisted itself.
    source_path + namespace are the document's LOGICAL identity — the file the
    content came from and the corpus partition it lives in — and drive lifecycle:
    replace-on-reingest, move detection, sync(), prune.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str
    filename: str = ""
    source_format: str = ""
    page_count: int = 0
    created_at: str = ""
    category: str = ""
    sections: int = 0
    chunks: int = 0
    source_path: str = ""
    namespace: str = ""


def _key(doc_id: str, *parts: str) -> str:
    return "/".join((_PREFIX, doc_id, *parts))


def _put(key: str, body: bytes, content_type: str) -> None:
    get_blob_store().put(key, body, content_type)


def _get_or_none(key: str) -> bytes | None:
    return get_blob_store().get_or_none(key)


def _page_key(doc_id: str, page_num: int) -> str:
    return _key(doc_id, "parse", "pages", f"page_{page_num:04d}.png")


def missing_blobs(doc_id: str, filename: str) -> list[str]:
    """Essential blobs that should exist for a stored document but don't — the
    source bytes (the ultimate truth: the full pipeline is re-runnable from them)
    and the assembled document.md. The audit half of verify for the blob store."""
    store = get_blob_store()
    essential = {"document.md": _key(doc_id, "parse", "document.md")}
    if filename:  # can only locate the source blob when its name is known
        essential["source"] = _key(doc_id, "source", filename)
    return [name for name, key in essential.items() if not store.exists(key)]


# ---------- parse (bytes) ----------


def save_parse(result: ParseResult) -> str:
    """Persist a ParseResult's BYTES — source file, page renders, figure crops,
    and the whole-document markdown. Structure + metadata go to the registry."""
    if not result.source_checksum:
        raise ValueError("ParseResult has no source_checksum — cannot derive doc_id")
    doc_id = result.source_checksum

    # original file
    source = result.source_path
    if source.exists():
        _put(
            _key(doc_id, "source", source.name),
            source.read_bytes(),
            "application/octet-stream",
        )

    # binary artifacts
    n_figures = 0
    for page in result.pages:
        if page.image_bytes is not None:
            _put(_page_key(doc_id, page.page_num), page.image_bytes, "image/png")
        for fig in page.figures:
            _put(
                _key(doc_id, "parse", "figures", fig.filename(page.page_num)),
                fig.image_bytes,
                "image/png",
            )
            n_figures += 1

    _put(_key(doc_id, "parse", "document.md"), result.markdown.encode(), "text/markdown")
    logger.info(
        "saved parse bytes: doc_id=%s pages=%d figures=%d",
        doc_id[:12], result.page_count, n_figures,
    )
    return doc_id


def load_split(doc_id: str) -> SplitResult:
    """Load a document's split output, reconstructed from the registry."""
    from ingestlib.storage import registry

    result = registry.split_result(doc_id)
    if result is None:
        raise FileNotFoundError(
            f"no split data in the registry for doc_id {doc_id[:12]!r}… — run "
            f"split() (or ingest()) on the document first"
        )
    return result


# ---------- corpus registry — reads served by the Postgres registry ----------


def _meta_from_row(row: dict[str, Any]) -> DocumentMeta:
    """A registry documents-row dict → the DocumentMeta shape callers expect."""
    created = row.get("created_at")
    return DocumentMeta(
        doc_id=row["doc_id"],
        filename=row.get("filename", ""),
        source_format=row.get("source_format", ""),
        page_count=row.get("page_count", 0),
        created_at=created.isoformat() if created else "",
        category=row.get("category", ""),
        sections=row.get("section_count", 0),
        chunks=row.get("chunk_count", 0),
        source_path=row.get("source_path", ""),
        namespace=row.get("namespace", ""),
    )


def document_exists(doc_id: str) -> bool:
    """True when this document has a registry row (dedup check)."""
    from ingestlib.storage import registry

    return registry.document_exists(doc_id)


def ingest_complete(doc_id: str) -> bool:
    """True when the FULL pipeline finished (the vector/embed step ran)."""
    from ingestlib.storage import registry

    return registry.ingest_complete(doc_id)


def get_document_meta(doc_id: str) -> DocumentMeta:
    """Registry entry for one document (empty meta when unknown)."""
    from ingestlib.storage import registry

    row = registry.meta_row(doc_id)
    return _meta_from_row(row) if row is not None else DocumentMeta(doc_id=doc_id)


def list_documents() -> list[DocumentMeta]:
    """Every live document in the registry — id, filename, pages, category, counts."""
    from ingestlib.storage import registry

    return [_meta_from_row(r) for r in registry.meta_rows()]


def find_by_path(path: Path | str, namespace: str = "") -> DocumentMeta | None:
    """The live document currently claiming this source path — logical identity.

    Matched on the resolved absolute path AND namespace via an indexed registry
    query (replaced/tombstoned rows excluded); newest created_at wins.
    """
    from ingestlib.storage import registry

    row = registry.meta_by_path(str(Path(path).resolve()), namespace)
    return _meta_from_row(row) if row is not None else None


def set_source_path(doc_id: str, path: Path | str) -> None:
    """Re-point a document's logical identity after its file moved.

    The content — and so the doc_id — is unchanged: only source_path moves,
    updated in the registry (the read source).
    """
    from ingestlib.storage import registry

    resolved = str(Path(path).resolve())
    registry.set_source_path(doc_id, resolved)
    logger.info("moved: doc_id=%s now at %s", doc_id[:12], path)


# ---------- blob reads (page renders, figure crops, whole-doc markdown) ----------


def document_markdown(doc_id: str) -> str | None:
    """The whole-document markdown blob (parse/document.md), or None if absent."""
    body = _get_or_none(_key(doc_id, "parse", "document.md"))
    return body.decode() if body is not None else None


def page_image_key(doc_id: str, page_num: int) -> str:
    """Artifact key of a page render — read_blob() serves it on any backend
    (on s3, get_s3_client().generate_presigned_url can serve it as a URL)."""
    return _page_key(doc_id, page_num)


def read_blob(key: str) -> bytes:
    """Raw bytes at an artifact key, whichever backend holds them.

    The backend-agnostic way for a UI to serve page renders and figure crops
    when a presigned URL is not available (artifact_store: local)."""
    return get_blob_store().get(key)


def delete_document(doc_id: str) -> int:
    """Remove every object under the document's prefix. Returns count deleted."""
    deleted = get_blob_store().delete_prefix(_key(doc_id) + "/")
    logger.info("deleted document %s (%d objects)", doc_id[:12], deleted)
    return deleted
