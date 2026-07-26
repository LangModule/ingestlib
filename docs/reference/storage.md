# Storage API

## Artifacts

Persist and reload every stage's output, keyed by document checksum, on
the configured backend (`s3` | `local`).

```python
from ingestlib.storage import artifacts
```

::: ingestlib.storage.artifacts.save_parse

::: ingestlib.storage.artifacts.load_parse

::: ingestlib.storage.artifacts.load_classify

::: ingestlib.storage.artifacts.load_split

::: ingestlib.storage.artifacts.load_ingest_manifest

::: ingestlib.storage.artifacts.document_exists

::: ingestlib.storage.artifacts.ingest_complete

::: ingestlib.storage.artifacts.list_documents

::: ingestlib.storage.artifacts.get_document_meta

::: ingestlib.storage.artifacts.page_image_key

::: ingestlib.storage.artifacts.read_blob

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
