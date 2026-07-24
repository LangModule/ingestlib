# The studio

[ingestlib-studio](https://github.com/LangModule/ingestlib-studio) is
the visual companion: a local web app over this library that lets you
**see exactly what your documents became** — no cloud, single user,
runs beside your config.

```bash
git clone https://github.com/LangModule/ingestlib-studio.git
cd ingestlib-studio
(cd backend && uv sync) && (cd frontend && npm install)
make dev        # backend :8000 + frontend :5173
```

On a machine without ingestlib configuration, a **setup wizard** opens
automatically: it verifies your AWS access, storage, and reranker with
real calls, hands you a pre-filled least-privilege IAM policy, writes
`~/.ingestlib/{config.yaml,.env}`, and activates them without a
restart. (Choosing OpenAI + local artifacts + SQLite skips AWS
entirely.)

## What each screen gives you

- **Try it** — run parse → classify → split on a file entirely in
  memory (nothing stored), then review the result page by page:
  hovering any chunk lights up exactly the regions it came from on the
  page render, and the reverse. A rules panel lets you override
  [content rules](content-rules.md) for that one run — the fastest way
  to iterate on rules against a real document.
- **Ingest** — commit a document with a live five-stage stepper
  (driven by the library's `on_stage` events over SSE).
- **Library** — every ingested document, read straight from the
  artifact store; open any one in the same review shell.
- **Playground** — ask a question, get cited hits, click a citation to
  open the document at that page with the source regions pre-lit.
- **Settings** — edit the whole stack live (applied via
  `reset_config()`, no restart), run health checks, edit the saved
  rules.yaml, and **backfill** — re-embed the stored corpus into a
  newly selected vector store without re-parsing.

## Why it exists

The studio is a pure consumer of this library's public API — it does no
parsing and no model calls of its own. It exists because provenance is
the product: `region_ids` + bounding boxes are only convincing when you
can see the highlight land on the page. It also serves as the reference
implementation for building UIs on top of ingestlib (SSE progress,
artifact-store reads, per-run rules).

Its own architecture docs live in the
[repository's READMEs](https://github.com/LangModule/ingestlib-studio).
