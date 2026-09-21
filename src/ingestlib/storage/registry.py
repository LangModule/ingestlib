"""Write ingest metadata INTO the registry, and read one document back OUT.

Plain in-process functions (not a web API) that map the operation outputs —
ParseResult / ClassifyResult / SplitResult / ExtractResult — onto the
ingestlib_registry ORM rows and persist them. The blob store keeps the bytes
(source, page PNGs, figure crops); everything queryable lives here.

Depends on ingestlib_registry (one way); the registry package never imports
ingestlib. The Document row is upserted field-by-field across the pipeline
stages (parse sets identity, classify sets the category columns, …) and the
child tables are replaced whole per doc_id. doc_id is a content hash, so a
re-ingest overwrites in place.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from sqlalchemy import delete, inspect as sa_inspect, select

from ingestlib_registry.db import session_scope
from ingestlib_registry.models import (
    Chunk,
    Collection,
    Document,
    Extraction,
    Page,
    Region,
    Section,
)

if TYPE_CHECKING:  # annotations only — no runtime import of the operations layer
    from sqlalchemy.orm import Session

    from ingestlib.operations.classify.models import ClassifyResult
    from ingestlib.operations.extract.models import ExtractResult
    from ingestlib.operations.parse.models import ParseResult
    from ingestlib.operations.split.models import SplitResult


_MIN_DT = datetime.min.replace(tzinfo=timezone.utc)  # sort key for rows with no created_at


def _update_document(session: "Session", doc_id: str, **fields: Any) -> Document:
    """Upsert the Document row, applying only the given fields (others keep their
    stored value / server default). Guarantees the parent row exists for children."""
    doc = session.get(Document, doc_id)
    if doc is None:
        doc = Document(doc_id=doc_id)
        session.add(doc)
    for key, value in fields.items():
        setattr(doc, key, value)
    return doc


def _bbox(bbox: Any) -> dict[str, float]:
    return {"x": bbox.x, "y": bbox.y, "width": bbox.width, "height": bbox.height}


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if isinstance(value, BaseModel) else value


# ── write side (ingest) ──────────────────────────────────────────────────────


def save_parse(
    doc_id: str,
    parse: "ParseResult",
    *,
    namespace: str = "",
    source_path: str | Path | None = None,
) -> None:
    """documents (identity + source) · pages · regions."""
    path = Path(source_path) if source_path is not None else Path(parse.source_path)
    with session_scope() as s:
        _update_document(
            s, doc_id,
            namespace=namespace,
            source_path=str(path.resolve()),  # resolved, so find_by_path (resolved) matches
            filename=path.name,
            source_format=parse.source_format,
            source_metadata=parse.source_metadata or None,
            page_count=len(parse.pages),
            created_at=parse.created_at,
        )
        s.flush()  # parent before children (no ORM relationships to order for us)
        s.execute(delete(Region).where(Region.doc_id == doc_id))
        s.execute(delete(Page).where(Page.doc_id == doc_id))
        for page in parse.pages:
            s.add(Page(
                doc_id=doc_id,
                page_num=page.page_num,
                width=page.page_width or 0,
                height=page.page_height or 0,
            ))
            for region in page.regions:
                s.add(Region(
                    doc_id=doc_id,
                    page_num=page.page_num,
                    region_id=region.region_id,
                    region_type=region.region_type,
                    bbox=_bbox(region.bbox),
                    text=region.text,
                    content=region.content,
                ))


def save_classify(doc_id: str, classify: "ClassifyResult") -> None:
    """documents: category, confidence, reasoning, alternatives, collection.

    collection is the SOFT label = the classify category (the routing bin the
    document lands in); recollect() re-sorts it when the rules change.
    """
    alternatives = [
        {"label": a.label, "score": a.score} for a in classify.alternatives
    ] or None
    with session_scope() as s:
        _update_document(
            s, doc_id,
            category=classify.category,
            classify_confidence=classify.confidence,
            classify_reasoning=classify.reasoning,
            classify_alternatives=alternatives,
            collection=classify.category,
        )


def save_split(doc_id: str, split: "SplitResult") -> None:
    """sections · chunks · documents.section_count/chunk_count."""
    chunks = split.chunks
    with session_scope() as s:
        _update_document(
            s, doc_id,
            section_count=len(split.sections),
            chunk_count=len(chunks),
        )
        s.flush()
        s.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        s.execute(delete(Section).where(Section.doc_id == doc_id))
        for ordinal, section in enumerate(split.sections):
            s.add(Section(
                doc_id=doc_id,
                ord=ordinal,
                name=section.name,
                description=section.description,
                pages=section.pages,
            ))
        for chunk in chunks:
            s.add(Chunk(
                doc_id=doc_id,
                chunk_id=chunk.chunk_id,
                section=chunk.section,
                heading=chunk.heading,
                kind=chunk.kind,
                markdown=chunk.markdown,
                embedding_text=chunk.embedding_text,
                pages=chunk.pages,
                region_ids=chunk.region_ids or None,
                token_estimate=chunk.token_estimate,
            ))


def save_embed(
    doc_id: str,
    *,
    vector_store: str,
    vector_dim: int,
    vector_count: int,
    embedded_at: Any,
) -> None:
    """documents: vector_store, vector_dim, vector_count, embedded_at."""
    with session_scope() as s:
        _update_document(
            s, doc_id,
            vector_store=vector_store,
            vector_dim=vector_dim,
            vector_count=vector_count,
            embedded_at=embedded_at,
        )


def save_status(doc_id: str, *, status: str, replaced_doc_id: str = "") -> None:
    """documents: status, replaced_doc_id (the lifecycle columns)."""
    with session_scope() as s:
        _update_document(s, doc_id, status=status, replaced_doc_id=replaced_doc_id)


def save_extraction(doc_id: str, extract: "ExtractResult") -> None:
    """extractions — one row per extracted item; replaces this schema's prior rows."""
    with session_scope() as s:
        s.execute(
            delete(Extraction).where(
                Extraction.doc_id == doc_id,
                Extraction.schema_name == extract.schema_name,
            )
        )
        for index, item in enumerate(extract.items):
            s.add(Extraction(
                doc_id=doc_id,
                schema_name=extract.schema_name,
                item_index=index,
                value=_dump(item.value),
                fields={name: fv.model_dump(mode="json") for name, fv in item.fields.items()},
                pages=item.pages,
            ))


# ── read side ────────────────────────────────────────────────────────────────


def _row(obj: Any) -> dict[str, Any]:
    """Mapped columns of one ORM row as a plain, detached dict."""
    return {c.key: getattr(obj, c.key) for c in sa_inspect(obj).mapper.column_attrs}


def _int_keys(mapping: Any) -> Any:
    """JSONB stringifies int keys on write; restore them for region_ids on read."""
    return {int(k): v for k, v in mapping.items()} if isinstance(mapping, dict) else mapping


def document_exists(doc_id: str) -> bool:
    """True when a LIVE document row exists (tombstoned replacements read as gone)."""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        return doc is not None and doc.status != "replaced"


def ingest_complete(doc_id: str) -> bool:
    """True when a live document went through the full pipeline (the vector step ran)."""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        return doc is not None and doc.status != "replaced" and doc.embedded_at is not None


def meta_row(doc_id: str) -> dict[str, Any] | None:
    """The documents row as a plain dict, or None."""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        return _row(doc) if doc is not None else None


def meta_rows(namespace: str | None = None, *, include_replaced: bool = False) -> list[dict[str, Any]]:
    """Every live document row (tombstoned replacements excluded by default)."""
    with session_scope() as s:
        stmt = select(Document)
        if namespace is not None:
            stmt = stmt.where(Document.namespace == namespace)
        if not include_replaced:
            stmt = stmt.where(Document.status != "replaced")
        return [_row(d) for d in s.scalars(stmt).all()]


def meta_by_path(source_path: str, namespace: str = "", *, include_replaced: bool = False) -> dict[str, Any] | None:
    """The live document claiming (source_path, namespace); newest created_at wins."""
    with session_scope() as s:
        stmt = select(Document).where(
            Document.source_path == source_path, Document.namespace == namespace
        )
        if not include_replaced:
            stmt = stmt.where(Document.status != "replaced")
        rows = s.scalars(stmt).all()
        if not rows:
            return None
        return _row(max(rows, key=lambda d: d.created_at or _MIN_DT))


def count_documents(*, include_replaced: bool = False) -> int:
    """How many documents the registry holds (tombstones excluded by default)."""
    from sqlalchemy import func

    with session_scope() as s:
        stmt = select(func.count()).select_from(Document)
        if not include_replaced:
            stmt = stmt.where(Document.status != "replaced")
        return int(s.scalar(stmt) or 0)


def list_collections(namespace: str | None = None) -> list[dict[str, Any]]:
    """The live collections and their document counts (tombstones excluded)."""
    from sqlalchemy import func

    with session_scope() as s:
        stmt = select(Document.collection, func.count()).where(Document.status != "replaced")
        if namespace is not None:
            stmt = stmt.where(Document.namespace == namespace)
        stmt = stmt.group_by(Document.collection).order_by(Document.collection)
        return [{"collection": c or "", "count": n} for c, n in s.execute(stmt).all()]


def sync_collections(rules: dict[str, dict[str, Any]]) -> None:
    """Materialize the declared collections (name → {description, extract_schema,
    auto_extract}) into the collections table — upsert each, leaving rows for
    collections no longer declared untouched (a doc may still carry that label)."""
    if not rules:
        return
    with session_scope() as s:
        for name, rule in rules.items():
            coll = s.get(Collection, name)
            if coll is None:
                coll = Collection(name=name)
                s.add(coll)
            coll.description = rule.get("description") or ""
            coll.extract_schema = rule.get("extract_schema") or None
            coll.auto_extract = bool(rule.get("auto_extract", False))


def get_collection(name: str) -> dict[str, Any] | None:
    """One declared collection's row (name · description · extract_schema · auto_extract)."""
    with session_scope() as s:
        coll = s.get(Collection, name)
        return _row(coll) if coll is not None else None


def collection_rules() -> list[dict[str, Any]]:
    """Every declared collection row, name-ordered."""
    with session_scope() as s:
        rows = s.scalars(select(Collection).order_by(Collection.name)).all()
        return [_row(c) for c in rows]


def document_attrs(doc_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Per-document filter attributes (category · collection · classify_confidence)
    for a set of doc_ids — powers the registry-backed retrieve filters."""
    if not doc_ids:
        return {}
    with session_scope() as s:
        rows = s.scalars(select(Document).where(Document.doc_id.in_(doc_ids))).all()
        return {
            d.doc_id: {
                "category": d.category,
                "collection": d.collection,
                "classify_confidence": d.classify_confidence,
            }
            for d in rows
        }


def set_source_path(doc_id: str, source_path: str) -> None:
    with session_scope() as s:
        _update_document(s, doc_id, source_path=source_path)


def set_status(doc_id: str, status: str) -> None:
    with session_scope() as s:
        _update_document(s, doc_id, status=status)


def reconstruct_parse(doc_id: str) -> Any:
    """A TEXT-ONLY ParseResult rebuilt from the registry (no OCR) — enough to
    RE-CLASSIFY a stored document. Regions/figures/page images are not restored;
    each page carries its regions' text. Returns None when the doc has no pages."""
    from ingestlib.operations.parse.models import PageResult, ParseResult

    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return None
        page_rows = s.scalars(
            select(Page).where(Page.doc_id == doc_id).order_by(Page.page_num)
        ).all()
        region_rows = s.scalars(
            select(Region).where(Region.doc_id == doc_id)
            .order_by(Region.page_num, Region.region_id)
        ).all()
        source_path, source_format, filename = doc.source_path, doc.source_format, doc.filename

    text_by_page: dict[int, list[str]] = {}
    for r in region_rows:
        body = (r.content or r.text or "").strip()
        if body:
            text_by_page.setdefault(r.page_num, []).append(body)
    pages = [
        PageResult(
            page_num=p.page_num,
            text="\n".join(text_by_page.get(p.page_num, [])),
            markdown="\n".join(text_by_page.get(p.page_num, [])),
        )
        for p in page_rows
    ]
    if not pages:
        return None
    fmt = source_format if source_format in ("pdf", "docx", "pptx", "png", "jpeg", "webp") else "pdf"
    return ParseResult(
        pages=pages,
        source_path=Path(source_path or filename or f"{doc_id}.pdf"),
        source_format=fmt,
        source_checksum=doc_id,
    )


def split_result(doc_id: str) -> Any:
    """Reconstruct a SplitResult from the registry (chunk `text` derived from markdown,
    which is not stored). Returns None when the document has no sections/chunks."""
    from ingestlib.operations.split.models import (
        Chunk as _Chunk,
        Section as _Section,
        SplitResult as _SplitResult,
    )

    with session_scope() as s:
        sections = s.scalars(
            select(Section).where(Section.doc_id == doc_id).order_by(Section.ord)
        ).all()
        chunk_rows = s.scalars(
            select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_id)
        ).all()
        if not sections and not chunk_rows:
            return None
        by_section: dict[str, list] = {}
        for c in chunk_rows:
            by_section.setdefault(c.section, []).append(_Chunk(
                chunk_id=c.chunk_id, section=c.section, heading=c.heading,
                text=c.markdown, markdown=c.markdown, embedding_text=c.embedding_text,
                pages=c.pages or [], region_ids=_int_keys(c.region_ids) if c.region_ids else {},
                kind=c.kind, token_estimate=c.token_estimate,
            ))
        result_sections = [
            _Section(name=sec.name, description=sec.description, pages=sec.pages or [],
                     chunks=by_section.get(sec.name, []))
            for sec in sections
        ]
        return _SplitResult(sections=result_sections)


def get_document(doc_id: str) -> dict[str, Any] | None:
    """The whole document as plain dicts: the core row + its children in order.
    Returns None when doc_id is unknown. Blobs are fetched separately, by doc_id."""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return None
        pages = s.scalars(
            select(Page).where(Page.doc_id == doc_id).order_by(Page.page_num)
        ).all()
        regions = s.scalars(
            select(Region).where(Region.doc_id == doc_id)
            .order_by(Region.page_num, Region.region_id)
        ).all()
        sections = s.scalars(
            select(Section).where(Section.doc_id == doc_id).order_by(Section.ord)
        ).all()
        chunks = s.scalars(
            select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_id)
        ).all()
        extractions = s.scalars(
            select(Extraction).where(Extraction.doc_id == doc_id).order_by(Extraction.id)
        ).all()

        chunk_rows = []
        for chunk in chunks:
            row = _row(chunk)
            row["region_ids"] = _int_keys(row["region_ids"])
            chunk_rows.append(row)

        return {
            "document": _row(doc),
            "pages": [_row(p) for p in pages],
            "regions": [_row(r) for r in regions],
            "sections": [_row(sec) for sec in sections],
            "chunks": chunk_rows,
            "extractions": [_row(e) for e in extractions],
        }


def delete_document(doc_id: str) -> bool:
    """Delete the document and (via ON DELETE CASCADE) all its children."""
    with session_scope() as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return False
        s.delete(doc)
        return True
