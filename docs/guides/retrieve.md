# Retrieve

Ask a question, get ranked chunks that cite exactly where they came
from.

```python
from ingestlib.services import retrieve

res = retrieve("what were total revenues?", top_k=5)
for hit in res.hits:
    print(hit.rerank_score, hit.citation)
    # 0.93  doc 7b6b95d79149 · p.12 · financial_statements

print(res.context)
# [1] (doc 7b6b95d79149 · p.12 · financial_statements)
# Total revenues for the quarter were ...
```

Async: `aretrieve`.

## How a query runs

```
embed the question
   ↓
hybrid search           dense vectors AND lexical/BM25, over a 4× candidate
                        pool, fused with reciprocal-rank fusion (RRF)
   ↓
rerank                  a cross-encoder reads the full text of every
                        candidate and produces the final order
   ↓
hits with citations
```

**Why hybrid?** Dense vectors find meaning ("earnings" matches
"revenue"); lexical search finds exact tokens (part numbers, names,
"+360%"). Each catches what the other misses. Every one of the eight
connectors implements both halves — some fuse server-side (Qdrant,
Milvus, Weaviate), the rest client-side; the retrieval behavior is the
same.

**Why rerank?** It is the single biggest quality lever we measured
(+5 to +14 points hit@1 over raw vector order) — vector and lexical
scores are not comparable across systems, and the reranker, which
actually reads the text, is the final arbiter. Configured by
`reranker: jina | aws | none` in config.yaml; if the reranker fails,
retrieval degrades gracefully to vector order.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `top_k` | `5` | Hits returned (the candidate pool is 4× this) |
| `filters` | `None` | Metadata equality filters, e.g. `{"category": "sec_filing"}` — applied inside the store, before ranking |
| `namespace` | `""` | Query only this partition |
| `rerank` | `True` | Set `False` to skip reranking for this call |
| `store` | `None` → config's `vector_store` | Bring your own connector instance |

Filterable fields: `document_id`, `category`, `section`, `kind`.

## What a hit contains

Each `Hit` carries the stored chunk (markdown, text, heading, pages,
`region_ids` provenance), `vector_score`, `rerank_score` (when
reranked), and `citation` — the human-readable
`doc <id> · p.<page> · <section>` string.

`result.context` assembles the hits into a numbered, cited block ready
to paste into an LLM prompt — the fastest path from retrieval to a
grounded answer.

Full signatures: [API reference → Services](../reference/services.md).
