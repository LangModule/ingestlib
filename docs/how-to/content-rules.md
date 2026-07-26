# Your categories & sections

Classify and split are open-ended by default — the LLM decides the document
type and discovers the section vocabulary. For a real corpus you usually
know better. Content rules constrain both to **your** domain, per call or
as a preset.

## Classify with your categories

Pass up to 20 `{label: description}` rules — the result is one of your
labels or `"uncategorized"`, with confidence and ranked alternatives:

```python
from ingestlib.operations import classify

label = classify("doc.pdf", {
    "invoice":    "Itemized charges, tax info, and payment terms",
    "sec_filing": "10-K/10-Q style regulatory filings",
})
print(label.category, label.confidence)
```

Write descriptions like you'd brief a new hire — they're what the LLM
matches against, and one good sentence beats three vague ones.

### Read only the pages that matter

Long documents usually reveal their type in the first pages. Two knobs:

```python
classify("doc.pdf", target_pages="1,3,5-7", max_pages=5)
```

- `target_pages` — 1-based pages/ranges to read
- `max_pages` — cap applied after selection

## Split with your section vocabulary

Pass up to 50 `{section: description}` categories and the discovery pass
is skipped — every page is labeled against *your* sections:

```python
from ingestlib.operations import split

chunks = split("10k.pdf",
    vocabulary={
        "financial_statements": "Balance sheets, income statements, cash flows",
        "notes":                "Footnotes and accounting-policy disclosures",
    },
    unmatched="other",
)
```

Pages fitting no category follow `unmatched`:

| Mode | Behavior |
|---|---|
| `other` (default) | grouped into an honest `other` section |
| `require` | every page forced into the nearest category |
| `skip` | dropped entirely — no sections, no chunks, no vectors |

`skip` is the targeted-extraction mode: define only the sections you care
about and everything else never reaches the vector store.

## Preset the rules in `rules.yaml`

Put the same rules beside your `config.yaml` and every bare call — and the
whole `ingest()` pipeline — uses them automatically:

```yaml
# rules.yaml — what your documents MEAN; infra stays in config.yaml
classify:
  max_pages: 5
  rules:
    invoice:    "Itemized charges, tax info, and payment terms"
    sec_filing: "10-K/10-Q style regulatory filings"

split:
  unmatched: other
  categories:
    financial_statements: "Balance sheets, income statements, cash flows"
    notes:                "Footnotes and accounting-policy disclosures"
```

## Rules on the pipeline

`ingest()` accepts every rule argument and passes it through with
identical semantics:

```python
from ingestlib.services import ingest

ingest("doc.pdf",
    categories={"invoice": "…", "sec_filing": "…"},
    target_pages="1-3",
    vocabulary={"financial_statements": "…"},
    unmatched="skip",
)
```

!!! warning "Dedup doesn't see rules"

    Documents dedup by **content checksum only** — re-ingesting the same
    file with different rules still returns `skipped`. Pass
    `skip_existing=False` to re-run it under the new rules.

## Precedence

One rule everywhere: **explicit argument → `rules.yaml` preset →
open-ended default.** Passing `{}` explicitly forces the open-ended
behavior even when a preset exists:

```python
classify("doc.pdf", {})        # ignore rules.yaml, classify open-ended
split("doc.pdf", vocabulary={})  # ignore preset, discover sections
```

---

Next: [Ingest documents](ingest.md) — the pipeline end of these rules.
