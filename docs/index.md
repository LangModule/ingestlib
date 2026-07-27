# ingestlib

**Self-hosted document intelligence for RAG.** One library takes a raw
document — PDF, DOCX, PPTX, or an image — and produces searchable, **cited**,
retrieval-ready chunks. The territory of LlamaParse, Reducto, and
Unstructured.io, running entirely on your own stack.

```python
from ingestlib.services import ingest, retrieve

ingest("finance-10k.pdf")        # parse → classify → split → embed → vector store
result = retrieve("what were the total revenues?")

for hit in result.hits:
    print(hit.citation, "→", hit.chunk.heading)
# doc 3f9c2ab81e04 · p.42 · financial_statements → Consolidated Revenues
```

Every answer knows exactly where it came from: the document, the page, and
the bounding-box regions on that page. That provenance chain — not just
parsing quality — is what ingestlib is built around.

## Where to go

<div class="grid cards" markdown>

- **New here?**

    Install, run one document through the pipeline, and get a cited answer
    in about five minutes.

    [:octicons-arrow-right-24: Quickstart](get-started/quickstart.md)

- **Keeping data in-house?**

    LLM, embeddings, vectors, and artifacts all on your machine — no API
    keys, nothing leaves your network.

    [:octicons-arrow-right-24: Run fully local](how-to/local-stack.md)

- **Building on top?**

    Task-focused guides: your own categories, namespaces and filters,
    progress callbacks, and building a citations UI.

    [:octicons-arrow-right-24: How-to guides](how-to/parse-documents.md)

- **Looking something up?**

    Every function, every config key, every CLI flag — with defaults.

    [:octicons-arrow-right-24: Reference](reference/configuration.md)

</div>

## What you get

| Stage | Output |
|---|---|
| **Parse** | Layout-aware markdown per page: tables as HTML (merged cells intact), formulas as LaTeX, charts converted to data tables, figures as PNG crops with AI descriptions — every block traceable to a bounding box |
| **Classify** | A document-type label (`invoice`, `research_paper`, …) — open-ended or constrained to your own categories — with confidence and ranked alternatives |
| **Split** | Sections (pages grouped by role) containing natural chunks: boundaries follow the content, tables never split, every chunk carries a `[category › section › heading]` breadcrumb |
| **Extract** | Your Pydantic schema filled from the document — one instance or every instance in a batch — each field citing its page and regions, grounded against the source text, with honest confidence |
| **Ingest** | The whole pipeline in one call — every stage persisted to the artifact store, vectors upserted, documents deduplicated by content checksum |
| **Retrieve** | Question → hybrid search (dense + lexical) → rerank → hits with scores, citations, and a prompt-ready context block |

## Your stack, your choices

Everything pluggable, selected in one config file:

- **AI providers** — Amazon Bedrock (Nova), OpenAI (GPT-5), or a local
  [Ollama](https://ollama.com) server. Mix them: one for chat, another for
  embeddings.
- **Eight vector stores** — sqlite (zero setup, the default), Pinecone,
  Qdrant, Postgres/pgvector, MongoDB, Milvus, OpenSearch, Weaviate — all
  with hybrid dense + lexical search.
- **Artifacts** — S3 or a plain local folder.
- **OCR** — PaddleOCR-VL (0.9B), served from your own GPU.

Ready? Start with [Installation](get-started/installation.md).
