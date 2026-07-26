# Parse any format

`parse()` accepts PDF, DOCX, PPTX, and single images (PNG/JPEG/WebP), and
produces the same layout-aware `ParseResult` for all of them. Requires the
[OCR server](../get-started/installation.md#3-the-ocr-inference-server).

## PDFs

```python
from ingestlib.operations import parse

result = parse("report.pdf")
print(result.page_count, "pages in", result.parse_duration_seconds, "s")
```

Pages render at 200 dpi by default — a balance of OCR accuracy against
VLM token cost. Raise it for dense small print:

```python
result = parse("dense-datasheet.pdf", dpi=300)
```

Scanned PDFs need nothing special: parse always OCRs the page *render* and
never trusts the file's embedded text layer.

!!! failure "Password-protected or corrupt PDFs"

    Both fail immediately with a plain-language error — remove the
    password (print/export to a new PDF), or re-export the corrupt file.

## Word & PowerPoint

```python
result = parse("slides.pptx")
print(result.was_converted)   # True — went through PDF on the way
```

DOCX/PPTX convert to PDF via LibreOffice first. If LibreOffice isn't
installed, the error says exactly what to install — PDFs and images keep
working without it.

## Images

A single PNG, JPEG, or WebP parses as a one-page document:

```python
result = parse("whiteboard-photo.png")
page = result.pages[0]
print(page.markdown)
```

Same OCR, same chart reading, same regions and bounding boxes — an image
is simply a page with no text layer.

## Async

Inside an event loop (a notebook, a web handler), use the `a`-prefixed
form — the sync form will tell you so if you forget:

```python
from ingestlib.operations import aparse

result = await aparse("report.pdf")
```

## Cost & performance notes

- Parse is the expensive stage: OCR per page plus LLM enrichment for
  charts/figures plus a review pass — roughly $0.002/page in LLM spend on
  the default stack.
- The same file never parses twice in the pipeline: `ingest()` persists
  the result and dedups by content checksum. When calling `parse()`
  directly, persist it yourself with `artifacts.save_parse(result)` to get
  the same reuse.

---

Next: [Use parse results](parse-results.md) — markdown, tables, figures,
and page images.
