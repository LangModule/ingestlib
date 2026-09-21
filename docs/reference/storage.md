# Storage API

The corpus lives in two stores that share the `doc_id` (content checksum) as
their join key: the **registry** (Postgres) holds everything queryable —
structure, classification, chunks, extractions, lifecycle — and the **artifact
store** (`s3` | `local`) holds the bytes.

## Registry — read a stored document

The registry is authoritative for metadata. Read a whole document back — its
structure from the registry, its bytes lazily from the artifact store.

```python
from ingestlib.services import get_document
```

::: ingestlib.services.document.get_document

::: ingestlib.services.document.StoredDocument

## Artifacts — the bytes

The artifact store holds a document's **bytes only**: the source file, page
renders, figure crops, and the whole-document markdown. Everything queryable
lives in the registry.

```python
from ingestlib.storage import artifacts
```

::: ingestlib.storage.artifacts.save_parse

::: ingestlib.storage.artifacts.load_split

::: ingestlib.storage.artifacts.document_markdown

::: ingestlib.storage.artifacts.page_image_key

::: ingestlib.storage.artifacts.read_blob

::: ingestlib.storage.artifacts.missing_blobs

::: ingestlib.storage.artifacts.document_exists

::: ingestlib.storage.artifacts.ingest_complete

::: ingestlib.storage.artifacts.list_documents

::: ingestlib.storage.artifacts.get_document_meta

::: ingestlib.storage.artifacts.find_by_path

::: ingestlib.storage.artifacts.delete_document

## The VectorStore contract

Every connector implements this interface — code written against it runs
on any backend.

::: ingestlib.storage.base.VectorStore

::: ingestlib.storage.base.RetrievedChunk

## Connectors

```python
from ingestlib.storage import (
    SqliteStore, PineconeStore, QdrantStore, PgvectorStore,
    MongodbStore, MilvusStore, OpensearchStore, WeaviateStore,
    default_store,
)
```

::: ingestlib.storage.default_store

All eight constructors take `hybrid: bool = True` — pass `hybrid=False`
for dense-only behavior. Connection details come from configuration, never
constructor arguments — see
[Connect a vector store](../how-to/vector-stores.md).
