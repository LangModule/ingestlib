"""S3 artifact store — persists every operation's output, keyed by document checksum.

Layout (everything under one prefix per document):

    s3://{bucket}/documents/{doc_id}/
    ├── source/{filename}                     original file, exact bytes
    ├── parse/result.json                     ParseResult (image bytes stripped)
    ├── parse/document.md                     whole-document markdown
    ├── parse/pages/page_0001.png ...         page renders
    ├── parse/figures/{fig.filename} ...      figure/chart crops
    ├── classify/result.json                  ClassifyResult
    ├── split/result.json                     SplitResult (chunks with provenance)
    └── split/ingest_manifest.json            vector-store sync record

doc_id is the parse checksum, so re-saving the same file overwrites in place and
"already ingested?" is a single existence check. The citation chain needs no
database: a vector hit's {doc_id, pages, region_ids} resolves to page images and
bboxes straight from this layout.
"""
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.classify.models import ClassifyResult
from ingestlib.operations.parse.models import FigureImage, PageResult, ParseResult
from ingestlib.operations.split.models import SplitResult
from ingestlib.storage.s3.client import ensure_bucket, get_s3_client
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_PREFIX = "documents"


class DocumentMeta(BaseModel):
    """Lightweight per-document registry entry (stored as meta.json).

    Written at save_parse; category / section counts are patched in by
    save_classify and save_split. Self-heals from parse/result.json when a
    document predates this file.
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


def _key(doc_id: str, *parts: str) -> str:
    return "/".join((_PREFIX, doc_id, *parts))


def _put(key: str, body: bytes, content_type: str) -> None:
    get_s3_client().put_object(
        Bucket=ensure_bucket(), Key=key, Body=body, ContentType=content_type
    )


def _put_json(key: str, payload: dict[str, Any]) -> None:
    _put(key, json.dumps(payload, ensure_ascii=False).encode(), "application/json")


def _get(key: str) -> bytes:
    response = get_s3_client().get_object(Bucket=ensure_bucket(), Key=key)
    return response["Body"].read()


def _page_key(doc_id: str, page_num: int) -> str:
    return _key(doc_id, "parse", "pages", f"page_{page_num:04d}.png")


def _meta_key(doc_id: str) -> str:
    return _key(doc_id, "meta.json")


def _patch_meta(doc_id: str, **fields: Any) -> None:
    """Merge fields into the document's meta.json (created if absent)."""
    try:
        current = json.loads(_get(_meta_key(doc_id)))
    except Exception:
        current = {"doc_id": doc_id}
    current.update(fields)
    _put_json(_meta_key(doc_id), current)


def _load_meta(doc_id: str) -> DocumentMeta:
    """Load meta.json; rebuild it from parse/result.json for pre-meta documents."""
    try:
        return DocumentMeta.model_validate(json.loads(_get(_meta_key(doc_id))))
    except Exception:
        pass
    try:  # self-heal: derive from the parse result, then persist
        payload = json.loads(_get(_key(doc_id, "parse", "result.json")))
        meta = DocumentMeta(
            doc_id=doc_id,
            filename=Path(payload["source_path"]).name,
            source_format=payload["source_format"],
            page_count=len(payload["pages"]),
            created_at=payload["created_at"],
        )
        _put_json(_meta_key(doc_id), meta.model_dump())
        return meta
    except Exception:
        return DocumentMeta(doc_id=doc_id)


# ---------- parse ----------


def save_parse(result: ParseResult) -> str:
    """Persist a ParseResult and all its binary artifacts. Returns the doc_id.

    The JSON carries every structural field (regions, bboxes, markdown, ...);
    page renders and figure crops are written as separate PNG objects.
    """
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

    # structure (bytes stripped — they live as the objects above)
    payload = result.model_dump(
        mode="json",
        exclude={
            "pages": {
                "__all__": {
                    "image_bytes": True,
                    "figures": {"__all__": {"image_bytes"}},
                }
            }
        },
    )
    _put_json(_key(doc_id, "parse", "result.json"), payload)
    _put(_key(doc_id, "parse", "document.md"), result.markdown.encode(), "text/markdown")
    _patch_meta(
        doc_id,
        filename=source.name,
        source_format=result.source_format,
        page_count=result.page_count,
        created_at=result.created_at.isoformat(),
    )

    logger.info(
        "saved parse artifacts: doc_id=%s pages=%d figures=%d",
        doc_id[:12], result.page_count, n_figures,
    )
    return doc_id


def _region_from_dict(data: dict[str, Any]) -> Region:
    bbox = data["bbox"]
    return Region(
        region_type=data["region_type"],
        bbox=BoundingBox(
            x=bbox["x"], y=bbox["y"], width=bbox["width"], height=bbox["height"]
        ),
        region_id=data["region_id"],
        text=data["text"],
        content=data["content"],
        confidence=data["confidence"],
    )


def load_parse(doc_id: str, *, include_images: bool = False) -> ParseResult:
    """Load a persisted ParseResult.

    include_images=False (default) returns pages with image_bytes=None and
    figure crops as empty bytes — cheap, structure-only. include_images=True
    fetches every PNG back into the result.
    """
    payload = json.loads(_get(_key(doc_id, "parse", "result.json")))

    pages: list[PageResult] = []
    for p in payload["pages"]:
        regions = [_region_from_dict(r) for r in p["regions"]]
        figures = []
        for f in p["figures"]:
            fig = FigureImage(
                region_id=f["region_id"],
                region_type=f["region_type"],
                image_bytes=b"",
                caption=f["caption"],
                description=f["description"],
            )
            if include_images:  # fetch via the model's canonical filename — single source of truth
                fig = fig.model_copy(update={"image_bytes": _get(
                    _key(doc_id, "parse", "figures", fig.filename(p["page_num"]))
                )})
            figures.append(fig)
        image_bytes = (
            _get(_page_key(doc_id, p["page_num"])) if include_images else None
        )
        pages.append(PageResult(
            page_num=p["page_num"],
            text=p["text"],
            markdown=p["markdown"],
            regions=regions,
            figures=figures,
            native_text=p["native_text"],
            image_bytes=image_bytes,
            image_format=p["image_format"],
            image_dpi=p["image_dpi"],
            page_width=p["page_width"],
            page_height=p["page_height"],
        ))

    return ParseResult(
        pages=pages,
        source_path=payload["source_path"],
        source_format=payload["source_format"],
        was_converted=payload["was_converted"],
        source_metadata=payload["source_metadata"],
        source_checksum=payload["source_checksum"],
        created_at=payload["created_at"],
        parse_duration_seconds=payload["parse_duration_seconds"],
    )


# ---------- classify / split ----------


def save_classify(doc_id: str, result: ClassifyResult) -> None:
    """Persist a ClassifyResult under the document's prefix."""
    _put_json(_key(doc_id, "classify", "result.json"), result.model_dump(mode="json"))
    _patch_meta(doc_id, category=result.category)
    logger.info("saved classify artifact: doc_id=%s category=%s", doc_id[:12], result.category)


def load_classify(doc_id: str) -> ClassifyResult:
    """Load a persisted ClassifyResult."""
    return ClassifyResult.model_validate(json.loads(_get(_key(doc_id, "classify", "result.json"))))


def save_split(doc_id: str, result: SplitResult) -> None:
    """Persist a SplitResult (sections + chunks with full provenance)."""
    _put_json(_key(doc_id, "split", "result.json"), result.model_dump(mode="json"))
    _patch_meta(doc_id, sections=len(result.sections), chunks=len(result.chunks))
    logger.info(
        "saved split artifact: doc_id=%s sections=%d chunks=%d",
        doc_id[:12], len(result.sections), len(result.chunks),
    )


def load_split(doc_id: str) -> SplitResult:
    """Load a persisted SplitResult."""
    return SplitResult.model_validate(json.loads(_get(_key(doc_id, "split", "result.json"))))


def save_ingest_manifest(doc_id: str, manifest: dict[str, Any]) -> None:
    """Record what was pushed to the vector store (index, namespace, vector IDs)."""
    _put_json(_key(doc_id, "split", "ingest_manifest.json"), manifest)


def load_ingest_manifest(doc_id: str) -> dict[str, Any]:
    """Load the vector-store sync record written by save_ingest_manifest."""
    return json.loads(_get(_key(doc_id, "split", "ingest_manifest.json")))


# ---------- registry ----------


def document_exists(doc_id: str) -> bool:
    """True when this document was parsed and saved before (dedup check)."""
    response = get_s3_client().list_objects_v2(
        Bucket=ensure_bucket(), Prefix=_key(doc_id, "parse", "result.json"), MaxKeys=1
    )
    return response.get("KeyCount", 0) > 0


def get_document_meta(doc_id: str) -> DocumentMeta:
    """Registry entry for one document (self-healing, like list_documents)."""
    return _load_meta(doc_id)


def list_documents() -> list[DocumentMeta]:
    """Registry of every persisted document — id, filename, pages, category, counts."""
    client = get_s3_client()
    bucket = ensure_bucket()
    doc_ids: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{_PREFIX}/", Delimiter="/"
    ):
        for cp in page.get("CommonPrefixes", []):
            doc_ids.append(cp["Prefix"].split("/")[1])
    return [_load_meta(d) for d in doc_ids]


def page_image_key(doc_id: str, page_num: int) -> str:
    """S3 key of a page render — for presigned URLs in a UI."""
    return _page_key(doc_id, page_num)


def delete_document(doc_id: str) -> int:
    """Remove every object under the document's prefix. Returns count deleted."""
    client = get_s3_client()
    bucket = ensure_bucket()
    deleted = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_key(doc_id) + "/"):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
            deleted += len(keys)
    logger.info("deleted document %s (%d objects)", doc_id[:12], deleted)
    return deleted
