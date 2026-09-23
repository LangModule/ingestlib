---
template: home.html
title: ingestlib — self-hosted document intelligence for RAG
hide:
  - navigation
  - toc
---

<p class="hp-lead" markdown>
The capabilities of LlamaParse, Reducto, and Unstructured.io — layout-aware parsing,
schema-driven extraction, hybrid retrieval, and text-to-SQL — as one open-source
Python library that runs **entirely on your own stack**. No document ever leaves
your network.
</p>

<p class="section-eyebrow">The pipeline</p>
<h2 class="section-title">Eight capabilities, one call each</h2>

<div class="grid cards" markdown>

-   :material-file-document-outline:{ .lg .middle } __Parse__

    Layout-aware markdown per page — tables as HTML, formulas as LaTeX, charts as
    data tables, figures as PNG crops with AI descriptions. Every block traces to a
    bounding box on the page.

-   :material-tag-outline:{ .lg .middle } __Classify__

    A document-type label — open-ended or constrained to your own categories — with
    confidence and ranked alternatives. Works standalone, no OCR.

-   :material-scissors-cutting:{ .lg .middle } __Split__

    Sections grouped by role, containing natural chunks: boundaries follow the
    content, tables never split, each chunk carries a context breadcrumb.

-   :material-format-list-checks:{ .lg .middle } __Extract__

    Your Pydantic schema, filled and cited — every field grounded against the source
    text with honest, verification-capped confidence.

-   :material-database-import-outline:{ .lg .middle } __Ingest__

    The whole pipeline in one call — queryable output to the registry, bytes to the
    artifact store, vectors upserted, deduplicated by content checksum.

-   :material-text-search:{ .lg .middle } __Retrieve__

    Hybrid search (dense + lexical) → rerank → cited hits with scores and a
    prompt-ready context block.

-   :material-database-search-outline:{ .lg .middle } __Query databases__

    The same `retrieve()` also answers from your SQL databases — read-only generated
    SQL behind a permission boundary, merged with document results.

-   :material-robot-outline:{ .lg .middle } __Serve to agents__

    `ingestlib mcp` exposes the whole loop as MCP tools — point Claude Desktop or
    Cursor at your self-hosted corpus.

</div>

<p class="section-eyebrow">Provenance</p>
<h2 class="section-title">Every answer knows where it came from</h2>

<p class="hp-lead" markdown>
Not just the document — the page, and the bounding-box regions on that page. That
provenance chain, not just parsing quality, is what ingestlib is built around.
</p>

```python
from ingestlib.services import retrieve

result = retrieve("what were the total revenues?")
for hit in result.hits:
    print(hit.citation, "→", hit.chunk.heading)
# doc 3f9c2ab81e04 · p.42 · financial_statements → Consolidated Revenues
```

<p class="section-eyebrow">Your stack, your choices</p>
<h2 class="section-title">Everything pluggable, one config file</h2>

- **AI providers** — Amazon Bedrock (Nova), OpenAI (GPT-5), or a local
  [Ollama](https://ollama.com) server. Mix them: one for chat, another for embeddings.
- **Eight vector stores** — SQLite (zero setup, the default), Pinecone, Qdrant,
  Postgres/pgvector, MongoDB, Milvus, OpenSearch, Weaviate — all hybrid dense + lexical.
- **Artifacts** — AWS S3, a self-hosted MinIO, or a plain local folder. **Registry** — the built-in Postgres
  metadata hub that makes the whole corpus queryable.
- **OCR** — PaddleOCR-VL (0.9B), served from your own GPU.

<div class="logos">
  <span>Bedrock Nova</span>
  <span>OpenAI GPT-5</span>
  <span>Ollama</span>
  <span>Pinecone</span>
  <span>Qdrant</span>
  <span>pgvector</span>
  <span>MongoDB</span>
  <span>Milvus</span>
  <span>OpenSearch</span>
  <span>Weaviate</span>
  <span>SQLite</span>
</div>

<div class="compare" markdown>
**Why self-hosted?** Hosted parsing APIs mean your documents — contracts, filings,
patient records — leave your network and meter by the page. ingestlib runs the same
class of pipeline on infrastructure you control: your GPU for OCR, your provider (or
a fully local Ollama) for the LLM, your vector store, your object storage. ~$0.002 a
page in LLM spend, or nothing at all on the local stack.
</div>

<hr class="brand-rule">

## Where to go

<div class="grid cards" markdown>

-   __New here?__

    Install, run one document through the pipeline, and get a cited answer in about
    five minutes.

    [:octicons-arrow-right-24: Quickstart](get-started/quickstart.md)

-   __Keeping data in-house?__

    LLM, embeddings, vectors, and artifacts all on your machine — no API keys,
    nothing leaves your network.

    [:octicons-arrow-right-24: Run fully local](how-to/local-stack.md)

-   __Building on top?__

    Task-focused guides: your own categories, namespaces and filters, corpus
    lifecycle, and building a citations UI.

    [:octicons-arrow-right-24: How-to guides](how-to/parse-documents.md)

-   __Looking something up?__

    Every function, every config key, every CLI flag — with defaults.

    [:octicons-arrow-right-24: Reference](reference/configuration.md)

</div>
