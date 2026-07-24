# Parse

Turn a PDF, DOCX, or PPTX into structured, provenance-carrying markdown.

```python
from ingestlib.operations import parse

result = parse("report.pdf")        # ParseResult
print(result.page_count)
print(result.markdown)              # the whole document
print(result.pages[0].regions[0])   # every block knows its bounding box
```

Office files are converted to PDF first (LibreOffice, in memory), then
follow the same path. There is no page cap. Async: `aparse`.

## How a page is parsed

Five stages, per page:

```
render + native text          pypdfium2 renders at 200 dpi and extracts
                              the PDF's own text layer
        ↓
layout detection              PP-DocLayoutV3 (local CPU) draws a bounding
                              box around every region: paragraphs, titles,
                              tables, charts, formulas, figures, furniture
        ↓
recognition per region        PaddleOCR-VL-1.6 (0.9B VLM on your GPU
                              server) reads each cropped region — text as
                              text, tables as HTML with merged cells,
                              formulas as LaTeX
        ↓
enrich + review (LLM)         chart crops re-read into data tables,
                              figure crops described, captions linked by
                              geometry; a review pass returns per-region
                              corrections — never a page rewrite
        ↓
assembly (pure Python)        regions in reading order become markdown;
                              headers/footers excluded; every block
                              traceable to its region_id
```

### Why two models?

The small OCR model reads text and tables at state-of-the-art accuracy
but **misreads chart values and fabricates numbers from diagrams**. The
frontier LLM (Nova 2 Lite or GPT-5 mini, ~$0.002/page) touches only
what the small model gets wrong: chart and figure crops, plus the
review pass. Estimated chart values are marked with `~` — a bar chart
with unprinted numbers is an estimate and says so.

### Concurrency

OCR is GPU-serialized; the LLM stages run in parallel behind it — page
N is in OCR while page N−1 is being enriched. Wall-clock time ≈ OCR
time, roughly 12 s/page on an M-series Mac.

## What you get back

`ParseResult` holds:

- `markdown` — the assembled document (page furniture excluded).
- `pages` — one `PageResult` per page: its render (`image_bytes`), its
  `regions` (type, text, content, bounding box, `region_id`), its
  `figures` (each a `FigureImage` crop with description and nearest
  caption; `result.save_images("out/")` writes them all), and its own
  markdown.
- `source_checksum` — the SHA-256 of the file bytes; the same bytes
  always hash the same, which powers dedup — the artifact store derives
  the `doc_id` from it (`artifacts.save_parse(result)` returns it).

Bounding boxes carry a `normalized()` form (0–1 coordinates), which is
what UIs use to overlay highlights on the page render — the studio's
hover-provenance is built entirely on `region_id` + bbox.

!!! tip "Persisting a parse"
    `ingest()` saves parse results automatically. To do it yourself:
    `from ingestlib.storage import artifacts` —
    `artifacts.save_parse(result)` / `artifacts.load_parse(doc_id)`.
    See [Artifact store](artifact-store.md).

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `path` | — | PDF/DOCX/PPTX file |
| `dpi` | `200` | Page render resolution — balances OCR accuracy against VLM token cost and memory |

Full signatures: [API reference → Operations](../reference/operations.md).
