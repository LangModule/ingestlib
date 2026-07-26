# Retrieve & filter

`retrieve()` turns a question into ranked, cited chunks: embed → vector
search (hybrid on every store) → rerank → hits.

## The basic call

```python
from ingestlib.services import retrieve

result = retrieve("what does the report conclude?", top_k=5)

for hit in result.hits:
    print(hit.rerank_score, hit.vector_score, hit.citation)
    print(hit.chunk.markdown[:120])
```

Two scores per hit:

- `vector_score` — the store's signal (cosine similarity, or an RRF rank
  score when the hybrid branches fused)
- `rerank_score` — the cross-encoder's relevance (`None` when reranking
  is off); when present, it's the order the hits arrive in

## Filters

Equality constraints on the chunk payload — combine freely:

```python
retrieve("revenue growth",
         filters={"category": "sec_filing", "section": "financial_statements"})
```

Filterable fields on every store: `document_id`, `category`, `section`,
`kind` (`text` | `table` | `mixed`). An unknown field raises with the
valid list — filters never fail silently.

Scoping to one document is just a filter:

```python
retrieve("what were the risks?", filters={"document_id": doc_id})
```

## Namespaces

Query only one corpus partition — must match the namespace used at ingest:

```python
retrieve("question", namespace="tenant-a")
```

## Hybrid search happens automatically

Every store runs the dense query; hybrid stores also run a lexical BM25
search with the raw question text and fuse the two. You don't opt in —
exact tokens (invoice numbers, error codes, names) just work:

```python
retrieve("payment terms for INV-20114")   # the lexical side catches 'INV-20114'
```

## Reranking

On by default, selected by config.yaml's `reranker` key
(`jina | aws | none`). With reranking on, a 4× candidate pool is fetched
and the reranker orders it by reading full text. Turn it off per call:

```python
retrieve("question", rerank=False)        # vector order, no reranker call
```

If the reranker errors mid-query, retrieve logs a warning and returns
vector order — a flaky reranker never breaks retrieval.

## Bring your own store instance

Both services accept an explicit connector, bypassing config selection:

```python
from ingestlib.storage import QdrantStore

retrieve("question", store=QdrantStore())
```

## No hits?

`result.hits == []` usually means one of:

- nothing ingested into this store/namespace yet — check
  `artifacts.list_documents()` and your `namespace`
- a filter excluded everything — try without filters
- the corpus was embedded with a **different embedding provider** than the
  query — re-ingest or switch back
  ([why](../concepts/providers.md#switching-the-embedding-provider))

---

Next: [Build cited answers](cited-answers.md) — from hits to an answer
that points at pages.
