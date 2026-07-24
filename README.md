# ingestlib

![ingestlib — self-hosted document intelligence for RAG](assets/cover.png)

Self-hosted document intelligence for RAG pipelines. One library that takes a
raw document — PDF, DOCX, PPTX — and produces searchable, **cited**,
retrieval-ready chunks: the territory of LlamaParse / Reducto /
Unstructured.io, running on your own stack.

```python
from ingestlib.services import ingest, retrieve

ingest("finance-10k.pdf")            # parse → classify → split → embed → vector store
result = retrieve("what were the total revenues?")
print(result.context)                # ranked chunks, each citing doc · page · section
```

**Documentation: [langmodule.github.io/ingestlib](https://langmodule.github.io/ingestlib/)** —
guides for every stage, the full configuration reference, and the API docs.

## What it does

| Stage | What you get |
|---|---|
| **Parse** | Layout-aware markdown per page: tables as HTML (merged cells intact), formulas as LaTeX, **charts converted to data tables** (estimated values marked `~`, printed callouts and growth labels captured), figures extracted as PNG crops with captions and AI descriptions — every block traceable to a bounding box on the page |
| **Classify** | Document-type label (`invoice`, `research_paper`, …) — open-ended, or constrained to your rules (per call or preset in `rules.yaml`, with page targeting) — confidence and ranked alternatives included. Works standalone with **no OCR** |
| **Split** | Sections (pages grouped by role: `methods`, `results`, … — LLM-discovered, or **your own categories** via rules) containing **natural chunks** — boundaries follow the content, tables never split, each chunk carries a `[category › section › heading]` breadcrumb in its `embedding_text` |
| **Ingest** | The whole pipeline in one call, every stage persisted to the artifact store (S3 or a local folder), vectors upserted, deduplicated by content checksum |
| **Retrieve** | Question → **hybrid search** (dense embeddings + lexical sparse, merged) → **rerank** (Jina by default; Amazon Rerank or none via `reranker:` in config.yaml) → hits with scores and citations, plus a prompt-ready context block |

Engines: **PaddleOCR-VL-1.6** (0.9B VLM, runs on your GPU) for layout + recognition,
**Amazon Nova 2 Lite** for judgment (chart reading, review, classification,
chunk boundaries), **Nova multimodal embeddings**, **eight vector stores**
(Pinecone, Qdrant, SQLite, Postgres/pgvector, MongoDB, Milvus, OpenSearch,
Weaviate — all hybrid dense + sparse), **S3 or a local folder** for
artifacts (`artifact_store: s3 | local`). ~$0.002/page in LLM spend. An
**OpenAI backend** (GPT-5 vision-capable chat + text-embedding-3) ships
alongside Bedrock — flip `llm_provider: openai` / `embedding_provider:
openai` to run the whole pipeline on it instead. See below.

## Quickstart

### 1. Requirements

- Python 3.12+ and [uv](https://github.com/astral-sh/uv)
- **AWS account** with Bedrock access (`us-east-1`): Nova 2 Lite + Nova 2
  multimodal embeddings — the default provider; the OpenAI backend can run
  the whole pipeline instead (see below)
- **Vector database** — Pinecone account (serverless, free tier works;
  the default), a Qdrant server (local docker or Qdrant Cloud), a Postgres
  with pgvector (RDS/Supabase/Neon or self-hosted), a MongoDB with search
  (Atlas any tier or 8.2+ self-managed), a Milvus (local docker or Zilliz
  Cloud), an OpenSearch (Amazon domain or local docker), a Weaviate (local
  docker or Weaviate Cloud) — each just one connection URL — or none at
  all: the sqlite connector stores vectors in a local file
- **Jina AI account** for reranking (free tier: 100 RPM) — the default; or set
  `reranker: aws` (Amazon Rerank, same AWS credentials) or `reranker: none`
  in config.yaml and skip Jina entirely

### 2. Install

```bash
pip install ingestlib          # or: uv add ingestlib
```

Or work from source:

```bash
git clone https://github.com/LangModule/ingestlib.git
cd ingestlib
uv sync
```

System dependency — LibreOffice (DOCX/PPTX → PDF conversion):

```bash
brew install --cask libreoffice          # macOS (binary is `soffice`)
sudo apt install libreoffice-core libreoffice-writer libreoffice-impress   # Linux
```

### 3. Start the OCR inference server

Parse runs PaddleOCR-VL-1.6 behind an inference server. First launch downloads
~1.8 GB of weights; later launches load from cache in seconds.

```bash
# Apple Silicon (Metal GPU)
uv run python -m mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6

# NVIDIA (then set paddle_vl.backend: vllm-server in config.yaml)
vllm serve PaddlePaddle/PaddleOCR-VL-1.6 --port 8111
```

The layout model (PP-DocLayoutV3, ~126 MB) auto-downloads on the first parse.

### 4. Configure

```bash
cp .env.example .env                 # API keys: Jina, plus your vector store's (sqlite needs none)
cp config.example.yaml config.yaml   # AWS profile + vector store + reranker choice
cp rules.example.yaml rules.yaml     # optional: your classify & split rules (see below)
aws configure --profile your-aws-profile   # Bedrock-enabled credentials
```

Edit `config.yaml`: the `aws` section is the only required part — then pick
your vector store, reranker, and artifact store. Everything else has working
defaults. **The S3 bucket (default `ingestlib-{account_id}`) and the vector
indexes/collections are created automatically on first use** — no manual
setup. Prefer no cloud storage at all? `artifact_store: local` keeps every
parse, page image, and chunk in a plain folder beside your config.yaml —
browsable in a file manager, and moving a corpus between backends is a copy.

Config is discovered at call time, never at import: `INGESTLIB_CONFIG=/path/to/config.yaml`
wins, otherwise the working directory and its parents are searched — so
installed usage works the same as running inside this repo.

### 5. Run

```python
from ingestlib.services import ingest, retrieve

r = ingest("report.pdf")
print(r.status, r.category, r.chunks, r.durations)

res = retrieve("what does the report conclude?", top_k=5)
for hit in res.hits:
    print(hit.rerank_score, hit.citation, hit.chunk.heading)
```

## Using the operations directly

Every operation also works standalone:

```python
from ingestlib.operations import parse, classify, split

result = parse("report.pdf")            # ParseResult: pages, regions, figures
print(result.markdown)                  # whole-document markdown
result.save_images("out/")              # extracted figures/charts as PNGs

label = classify("report.pdf")          # no OCR needed — native text + embedded images
chunks = split(result, category=label.category)
for c in chunks.chunks:
    print(c.token_estimate, c.embedding_text.splitlines()[0])
```

Persistence and vector access are explicit too:

```python
from ingestlib.storage import artifacts, PineconeStore

doc_id = artifacts.save_parse(result)   # artifact store: source, result.json, page PNGs, crops
artifacts.list_documents()              # registry: filename, pages, category, chunks
```

## Classification & split rules

Classify and split are open-ended by default — the LLM decides the document
type and discovers the section vocabulary. Both can instead follow **your**
rules: pass them per call, or preset them once in `rules.yaml` beside your
config.yaml, and every bare call **and the whole ingest pipeline** uses
them automatically:

```python
classify("doc.pdf",
         {"invoice": "Itemized charges, tax info, and payment terms",
          "sec_filing": "10-K/10-Q style regulatory filings"},
         target_pages="1,3,5-7", max_pages=5)   # read only these pages
```

```python
split("report.pdf",
      vocabulary={"financial_statements": "Balance sheets, income statements",
                  "notes": "Footnotes and disclosures"},
      unmatched="other")   # pages fitting nothing: other (default) | require | skip
```

```yaml
# rules.yaml — copy rules.example.yaml; infra stays in config.yaml,
# what your documents MEAN lives here
classify:
  max_pages: 5
  rules:                       # up to 20 — result is one of these or "uncategorized"
    invoice: "Itemized charges, tax info, and payment terms"
    sec_filing: "10-K/10-Q style regulatory filings"
split:
  unmatched: other             # require | other | skip
  categories:                  # up to 50 — YOUR sections; Pass 1 is skipped
    financial_statements: "Balance sheets, income statements, cash flows"
    notes: "Footnotes and disclosures"
```

Classify returns one of your labels or `"uncategorized"`, with confidence,
reasoning, and ranked alternatives. Split labels every page against your
sections — unmatched pages become an honest `other` section (default), get
forced into the nearest category (`require`), or are dropped entirely
(`skip`). Precedence everywhere: explicit arguments beat the preset, and
`{}` forces the open-ended default even when a preset exists.

## OpenAI backend

The same LLM surface Bedrock provides is also available on OpenAI — GPT-5
chat with vision, thinking mode, schema-enforced structured output, and
text-embedding-3 embeddings. Add `OPENAI_API_KEY` to `.env` and pick models
in config.yaml's `openai:` section (defaults: `gpt-5-mini`,
`text-embedding-3-small`).

To run the whole `ingest`/`retrieve` pipeline on it, switch the providers
in config.yaml — every LLM and embedding call routes accordingly:

```yaml
llm_provider: openai          # chart reading, review, classify, chunking
embedding_provider: openai    # chunk + query embeddings
```

Combined with `artifact_store: local`, `vector_store: sqlite`, and
`reranker: jina` (or `none`), the pipeline needs no AWS at all. Two rules:
switching `embedding_provider` changes the vector space, so re-ingest (or
`--backfill`) afterward — vectors from different embedding models never mix
in one index. And text embeddings only: OpenAI has no image-embedding model.

The backend is also importable directly, ignoring the config switch:

```python
from ingestlib.foundations.llm import Image
from ingestlib.foundations.llm.openai import chat, chat_structured, embed_text

chat("Read this chart", images=[Image(png_bytes, "png")])   # vision works
embed_text("a chunk of text")                               # 1024-dim default
```

## Architecture

```
src/ingestlib/
├── services/       ingest · retrieve          — the product
├── operations/     parse · classify · split   — the tools (each standalone)
├── storage/        artifacts (S3 | local) · base (VectorStore contract) · 8 connectors
│                   (pinecone · qdrant · sqlite · pgvector · mongodb · milvus
│                    · opensearch · weaviate)
├── foundations/    llm (Bedrock Nova · OpenAI GPT-5 · Jina) · ocr (PaddleOCR-VL)
├── utils/          logger · files
└── config.py       config.yaml + .env → typed configs
```

Strict downward dependencies. The `VectorStore` contract means backends drop
in as connectors — all eight ship **hybrid search**: **Pinecone** (dense +
hosted sparse model, merged client-side), **Qdrant** (dense + BM25 with
server-side IDF and RRF fusion; local docker or cloud), **SQLite**
(sqlite-vec KNN + built-in FTS5 BM25 with porter stemming, RRF fusion — one
local file, no server, no keys), **Postgres/pgvector** (HNSW cosine +
built-in full-text over a generated weighted tsvector, RRF fusion — the
extension and table bootstrap automatically), **MongoDB** (Atlas Vector
Search + Atlas Search true BM25, RRF fusion — Atlas any tier or self-managed
8.2+; both search indexes bootstrap automatically), **Milvus** (dense
ANN + server-computed BM25 sparse, fused server-side with RRF — local docker
or Zilliz Cloud), **OpenSearch** (faiss HNSW k-NN + Lucene BM25, RRF fused
client-side — an Amazon OpenSearch domain SigV4-signed with your aws
profile, or local docker), and **Weaviate** (HNSW dense + native BM25 fused
server-side in one hybrid call — local docker or Weaviate Cloud). Pick one
with `vector_store: pinecone | qdrant | sqlite | pgvector | mongodb |
milvus | opensearch | weaviate` in config.yaml. Connection secrets sit in
`.env` together (sqlite needs none) — only the selected connector ever
builds a client.

## Logging

```bash
INGESTLIB_LOG_LEVEL=INFO           # DEBUG | INFO | WARNING | ERROR (default INFO)
INGESTLIB_LOG_THIRD_PARTY=1        # also show paddlex/httpx/botocore chatter
INGESTLIB_LOG_COLOR=0              # disable colored output
```

## Testing

Tests hit **real APIs, never mocks**. Pure logic runs always; server-hitting
suites are opt-in via env gates. The sqlite connector's full suite runs
ungated in `make test` — there is no server, so in-process IS the real thing.

```bash
make test                  # fast suite (~340 tests, ~2min; e2e groups skip)
make test-openai           # OpenAI backend       (skips without OPENAI_API_KEY)
make test-parse            # parse e2e            (needs VL server + LLM provider)
make test-classify         # classify e2e         (needs the LLM provider)
make test-split            # split e2e            (needs the LLM provider)
make test-s3               # artifact store e2e   (needs AWS)
make test-pinecone         # vector connector e2e (needs Pinecone + embeddings)
make test-qdrant           # vector connector e2e (needs a Qdrant server + embeddings)
make test-sqlite           # vector connector suite (no gate — nothing to need)
make test-pgvector         # vector connector e2e (needs Postgres at PGVECTOR_URL)
make test-mongodb          # vector connector e2e (needs MongoDB at MONGODB_URL)
make test-milvus           # vector connector e2e (needs Milvus at MILVUS_URL)
make test-opensearch       # vector connector e2e (needs OpenSearch at OPENSEARCH_URL)
make test-weaviate         # vector connector e2e (needs Weaviate at WEAVIATE_URL)
make test-services         # full product e2e     (needs the entire stack)
make test-all              # everything
make eval                  # retrieval quality eval (see below)
make docs                  # live-preview the documentation site
```

Fixture PDFs live in `tests/data/pdf/` — 14 real documents (research papers,
earnings decks, insurance forms, timetables, 10-Ks).

### Retrieval quality

Beyond pass/fail tests, `evals/` measures retrieval quality: 22 ground-truth
questions over the fixture corpus, run through the real `retrieve()` flow
under dense/hybrid × rerank on/off, scored by hit@k and MRR. Measured so far
(consistent across all eight connectors): **with reranking, every answer
lands in the top 3 hits (hit@3 = 1.00)**; hit@1 ranges 0.86–1.00 across runs.
Each run saves a timestamped snapshot to `evals/results/`, so quality changes
are visible over time.

## Disk footprint

| Component | Size | Location |
|---|---|---|
| Python deps | ~3 GB | `.venv/` |
| PaddleOCR-VL-1.6 weights | ~1.8 GB | `~/.cache/huggingface/hub/` |
| PP-DocLayoutV3 | ~126 MB | `~/.paddlex/official_models/` |
| LibreOffice | ~600 MB | system |

## Scope

English documents; PDF / DOCX / PPTX input. Images, charts, and tables
**inside** documents are fully extracted and interpreted; direct image files
and handwriting are out of scope by design.

## The studio

[ingestlib-studio](https://github.com/LangModule/ingestlib-studio) is the
visual companion: a local web UI with a setup wizard, try-before-you-commit
pipeline runs, page-by-page review with hover-to-highlight bounding boxes,
committed ingestion with live progress, a content-rules editor, and a
retrieval playground where every answer points to its source on the page.

## Roadmap

- Extract: schema-driven field extraction with source provenance
- Per-run content rules on `ingest()` (classify/split already accept them)

## License

See [LICENSE](./LICENSE).
