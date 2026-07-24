# ingestlib

Self-hosted document intelligence for RAG. ingestlib parses, classifies,
and splits PDF/DOCX/PPTX documents into searchable, cited,
retrieval-ready chunks — the capability of cloud parsers like LlamaParse,
running entirely on your own stack.

```python
from ingestlib.services import ingest, retrieve

ingest("report.pdf")                # parse → classify → split → embed → store
result = retrieve("what was Q1 revenue?")
print(result.context)               # ranked chunks, each cited to its source page
```

No document ever leaves your infrastructure: OCR runs on a local
inference server, the LLM calls go to your AWS Bedrock or OpenAI
account, and storage is your bucket (or a plain local folder) plus any
of eight vector databases — including SQLite, which needs no server at
all.

## What you get

- **Parse** — layout-aware OCR that reads tables as HTML (merged cells
  intact), formulas as LaTeX, and charts as data tables; the output is
  clean markdown where every block is traceable to a bounding box on
  its page.
- **Classify** — the document's type, open-ended or against your own
  rules, with confidence and reasoning.
- **Split** — role-based sections and natural, RAG-quality chunks with
  a `[category › section › heading]` breadcrumb baked into the
  embedding text.
- **Ingest & retrieve** — the pipeline composed end to end: every stage
  persisted, hybrid (dense + lexical) search, reranking, and answers
  that cite `document · page · section`.

## Where to start

| You want to… | Go to |
|---|---|
| Install and run the pipeline once | [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) |
| Understand every config key | [Configuration](getting-started/configuration.md) |
| Learn how a stage works | The [guides](guides/parse.md) — one per operation |
| Look up a function signature | The [API reference](reference/operations.md) |
| Pick a vector database | [Vector stores](guides/vector-stores.md) |
| Review documents visually | [The studio](guides/studio.md) |

## The shape of the library

```
services      ingest · retrieve            the composed flows
operations    parse · classify · split     each usable standalone
storage       artifact store + 8 vector-store connectors
foundations   LLM providers (Bedrock | OpenAI) + OCR engine
```

Every layer only calls downward, and everything above `foundations` is
provider-neutral: one config key switches the LLM, the embeddings, the
artifact store, or the vector database without touching code.
