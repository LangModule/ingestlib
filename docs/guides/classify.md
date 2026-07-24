# Classify

Determine a document's type — open-ended, or against your own closed
set of rules.

```python
from ingestlib.operations import classify

label = classify("invoice.pdf")
print(label.category)      # "invoice" — model-invented snake_case label
print(label.confidence)    # 0.97
print(label.reasoning)     # one paragraph of why
```

Async: `aclassify`.

## Two inputs, same result

| Input | What it reads | Needs the OCR server? |
|---|---|---|
| `classify(parse_result)` | The enriched markdown + figure crops from the parse | No (parse already ran) |
| `classify("file.pdf")` | The PDF's native text + embedded images pulled straight from the PDF objects | **No** — no rendering, no models beyond the one classify call |

Standalone mode makes classification cheap enough to run on documents
you may never fully parse: it extracts embedded images directly
(≥300 px, largest first, at most 3 per page, downscaled to 1600 px) and
sends text plus a handful of images to the LLM.

## Open-ended vs closed-set

Without rules, the model invents the best snake_case label. With
`categories`, classification becomes **closed-set**: the model must pick
one of your labels or answer `uncategorized` — it can never invent one.

```python
label = classify("doc.pdf", categories={
    "invoice": "Itemized charges, taxes, payment terms",
    "contract": "Signed agreements with obligations and parties",
})
print(label.category)        # "invoice" | "contract" | "uncategorized"
print(label.alternatives)    # every category, scored and ranked
```

| Parameter | Default | Meaning |
|---|---|---|
| `categories` | `None` | `{label: description}`, max 20. Presence switches to closed-set |
| `target_pages` | `None` | 1-based pages/ranges, e.g. `"1,3,5-7"` — read only these |
| `max_pages` | `None` | Cap after selection; `None`/`0` = no extra cap |

Unset parameters resolve from `rules.yaml`'s `classify:` preset when one
exists — see [Content rules](content-rules.md). Pass `categories={}`
explicitly to force open-ended despite a saved preset.

Page selection stays truthful: pages keep their **original numbers** in
the prompt, so the model's reasoning cites real pages even when it only
saw pages 1 and 3.

## Scale behavior

- **≤ 20 pages** — one structured-output call (text + the first 4
  document images).
- **> 20 pages** — map-reduce: 20-page text chunks classified in
  parallel, then one combine call.
- **Hard cap: 100 pages** (after `target_pages` selection). Front pages
  almost always identify a document's type.

## What you get back

`ClassifyResult`: `category`, `confidence` (0–1), `reasoning`,
`alternatives` (ranked `CategoryScore`s in closed-set mode), and
`pages_used` — how many pages the model actually read.

Full signatures: [API reference → Operations](../reference/operations.md).
