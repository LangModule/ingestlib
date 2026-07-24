# Storage

The vector-store contract, its eight connectors, and the artifact store.

## The contract

::: ingestlib.storage.base.VectorStore

::: ingestlib.storage.base.RetrievedChunk

::: ingestlib.storage.default_store

## Connectors

Every connector implements the same contract; pick one with the
`vector_store` key in config.yaml or instantiate directly. See the
[vector stores guide](../guides/vector-stores.md) for choosing.

::: ingestlib.storage.sqlite.SqliteStore

::: ingestlib.storage.pinecone.PineconeStore

::: ingestlib.storage.qdrant.QdrantStore

::: ingestlib.storage.pgvector.PgvectorStore

::: ingestlib.storage.mongodb.MongodbStore

::: ingestlib.storage.milvus.MilvusStore

::: ingestlib.storage.opensearch.OpensearchStore

::: ingestlib.storage.weaviate.WeaviateStore

## Artifact store

::: ingestlib.storage.artifacts
