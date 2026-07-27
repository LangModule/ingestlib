# Extract structured data

`extract()` fills a Pydantic schema you define from a document — and tells
you, field by field, **where each value came from and whether it checks
out**. Define the shape; the extraction cites its sources.

## One instance from a document

`mode="one"` (the default) fills a single instance from the whole
document — headline facts, a contract's key terms, a form's fields:

```python
from pydantic import BaseModel
from ingestlib.operations import extract

class Financials(BaseModel):
    company: str
    fiscal_year: int
    total_net_sales: float

report = extract("finance-10k.pdf", schema=Financials)

item = report.items[0]
print(item.value)                    # Financials(company='Apple Inc.', ...)
print(item.citation)                 # p.1
print(item.fields["total_net_sales"].grounded)   # True
```

`item.value` is a real instance of *your* class — validated, typed,
IDE-completable. Field descriptions are part of the prompt, so
`Field(description=...)` is the place to say what a value means:

```python
from pydantic import Field

class Financials(BaseModel):
    total_net_sales: float = Field(
        description="Total net sales in millions of dollars, most recent fiscal year"
    )
```

## Every instance in a batch

`mode="many"` finds all instances across the pages — all the receipts in
a scanned expense bundle, every line item, each policy in a filing:

```python
from ingestlib.operations import parse

class Receipt(BaseModel):
    merchant: str
    total: float
    currency: str

result = parse("expenses.pdf")               # a 16-page scan of receipts
report = extract(result, schema=Receipt, mode="many")

for item in report.items:
    print(item.value.merchant, item.value.total, item.citation)
# BART 20.0 p.10
# Hilton 214.6 p.2
# …
```

Long documents are read in overlapping page windows, and items that appear
in two windows are deduplicated by value — each real-world instance comes
back once.

## Provenance on every field

Each extracted item carries a `FieldValue` per schema field:

```python
total = item.fields["total"]
total.pages        # [10]           — where it was read
total.region_ids   # {10: [3]}      — the parse regions on that page
total.grounded     # True           — the value appears in the cited text
total.confidence   # 0.9            — honest, see below
```

The citations are **verified, not trusted**: a citation naming a region
that doesn't exist in the parse is dropped (and logged), and every value
is checked against its cited source text — numeric forms are normalized,
so `383285.0` grounds against a printed `383,285`.

Confidence follows from that verification:

- A field with **no valid citation** is capped at 0.3
- A field whose value **isn't found in the cited text** is capped at 0.5
  and flagged `grounded=False`
- A cited, grounded field keeps the model's own confidence

Treat `grounded=False` as "verify before use" — a reworded or derived
value can be correct without matching the source text verbatim. See
[Limits](../about/limits.md) for the honest edges.

## Parse first, or not

Extract accepts the same inputs as classify and split, with the same
trade-off:

- **A `ParseResult`** — full OCR quality and **region-level** citations.
  The right choice for scans (their text layers are garbage or missing)
  and whenever you want bounding-box provenance for a UI.
- **A raw path** — reads the native text layer directly, **page-level**
  citations (`region_ids` stays empty), no OCR server involved. The cheap
  path for born-digital documents.

```python
report = extract(parse("scan.pdf"), schema=Receipt, mode="many")  # regions
report = extract("born-digital.pdf", schema=Financials)            # pages only
```

A document with no readable text on either path raises the same clear
error as the other operations: run `parse()` first.

## Target pages

Know where the data lives? Same page targeting as classify:

```python
report = extract("filing.pdf", schema=Financials, target_pages="30-45")
```

Free-text guidance rides along with `instructions`:

```python
report = extract(result, schema=Receipt, mode="many",
                 instructions="Only meal and transport receipts; skip lodging.")
```

Documents are capped at 100 pages per call — use `target_pages` to reach
into anything larger.

## Persist and reload

Extraction results store beside the document's other artifacts, keyed by
schema name:

```python
from ingestlib.storage import artifacts

doc_id = artifacts.save_parse(result)
artifacts.save_extract(doc_id, report)

loaded = artifacts.load_extract(doc_id, Receipt)   # values revalidate into Receipt
```

## Async

Inside an event loop, use `aextract` — the sync form will tell you so if
you forget:

```python
from ingestlib.operations import aextract

report = await aextract(result, schema=Receipt, mode="many")
```

---

Next: [Your categories & sections](content-rules.md) — constrain classify
and split to your domain.
