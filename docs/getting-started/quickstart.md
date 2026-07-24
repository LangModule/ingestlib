# Quickstart

Two lines to a searchable, cited corpus. This page assumes
[installation](installation.md) is done and the OCR server is running.

## Ingest and retrieve

```python
from ingestlib.services import ingest, retrieve

r = ingest("report.pdf")
print(r.status, r.category, r.chunks, r.durations)
# ingested financial_report 14 {'parse': 26.3, 'classify': 2.1, ...}

res = retrieve("what does the report conclude?", top_k=5)
for hit in res.hits:
    print(hit.rerank_score, hit.citation)
    # 0.91 doc 7b6b95d79149 · p.4 · conclusions

print(res.context)   # numbered, cited, prompt-ready block for your LLM
```

`ingest` runs parse → classify → split → embed → upsert, persisting
every stage to the artifact store. Re-ingesting the same file is
detected by content hash and skipped. `retrieve` embeds the question,
runs hybrid (dense + lexical) search, reranks, and returns hits that
each know their source document, page, and section.

## The operations standalone

Each stage also works on its own — this is how you verify the pipeline
on a document before committing anything:

```python
from ingestlib.operations import parse, classify, split

result = parse("report.pdf")           # OCR + enrichment → ParseResult
print(result.markdown)                 # the whole document as markdown

label = classify(result)               # or classify("report.pdf") — no OCR needed
print(label.category, label.confidence, label.reasoning)

chunks = split(result, category=label.category)
for section in chunks.sections:
    print(section.name, section.pages)
```

Async variants (`aparse`, `aclassify`, `asplit`, `aingest`,
`aretrieve`) exist for every call — use them when you are already
inside an event loop; the sync forms are wrappers that run one.

## The zero-AWS path

The whole pipeline can run without an AWS account:

```yaml
# config.yaml
llm_provider: openai
embedding_provider: openai
artifact_store: local        # artifacts in a plain folder beside config.yaml
vector_store: sqlite         # vectors in a local file, no server, no keys
reranker: jina               # free key — or `none`
```

With `OPENAI_API_KEY` (and optionally `JINA_API_KEY`) in `.env`, the
only remaining piece is the local OCR server. Everything — parsed
pages, embeddings, search — stays on your machine.

## Where to go next

- How each stage actually works: [Parse](../guides/parse.md) ·
  [Classify](../guides/classify.md) · [Split](../guides/split.md)
- Constrain the pipeline to your document types:
  [Content rules](../guides/content-rules.md)
- Pick your database: [Vector stores](../guides/vector-stores.md)
- Review documents visually: [The studio](../guides/studio.md)
