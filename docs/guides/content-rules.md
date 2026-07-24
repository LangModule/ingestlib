# Content rules

`rules.yaml` is the optional sidecar that tells the pipeline what your
documents *mean* — classification rules and split categories — kept
deliberately separate from `config.yaml`'s infrastructure. Save it once
and every `classify`, `split`, and `ingest` call honors it with zero
code changes.

```yaml
# rules.yaml — beside config.yaml, loaded with it
classify:
  rules:                          # ≤ 20 — presence makes classify closed-set
    invoice: Itemized charges, tax info, and payment terms
    sec_filing: 10-K/10-Q style financial filings
    contract: Signed agreements with parties and obligations
  target_pages: "1,3,5-7"         # optional — read only these pages (1-based)
  max_pages: 5                    # optional — cap after selection; 0 = none

split:
  categories:                     # ≤ 50 — presence skips vocabulary discovery
    financial_statements: Balance sheets, income statements, cash flow
    management_discussion: MD&A narrative sections
    notes: Footnotes and disclosures
  unmatched: other                # require | other | skip   (default: other)
```

## Precedence — who wins

For every rules-shaped parameter, resolution is:

```
explicit call argument  >  rules.yaml preset  >  open-ended / discovered
```

- `classify("doc.pdf")` with a saved preset → closed-set against the
  preset.
- `classify("doc.pdf", categories={"memo": "..."})` → your argument
  wins; the preset is ignored.
- `classify("doc.pdf", categories={})` → the **empty dict is an
  explicit choice**: open-ended, preset bypassed. Same for
  `split(..., vocabulary={})` → discovery.
- No rules.yaml at all → open-ended classification, discovered split
  vocabulary. This is the honest default state.

`ingest()` needs no wiring: its classify and split stages resolve the
preset internally.

## Validation

The library validates rules at call time and refuses presets the
pipeline would choke on:

- More than 20 classification rules or 50 split categories → error.
- `target_pages` grammar: 1-based numbers and ascending ranges
  (`"1,3,5-7"`); `"0"` and `"7-5"` are rejected. Out-of-range pages are
  dropped; a selection that leaves nothing raises.
- `unmatched: require` or `skip` without categories → error (the mode
  only means something against a vocabulary).

## Applying and clearing

The file is read with the configuration, so a long-running process
picks up edits after `reset_config()`. Deleting the file (or clearing
it) returns the pipeline to open-ended everywhere. Rules affect
**future** runs only — already-ingested documents keep their categories
and sections.

!!! tip "Editing visually"
    [The studio](studio.md) has a rules editor on its Settings page
    that writes this exact file, and its Try-it page accepts per-run
    overrides — useful for iterating on rules against a real document
    before saving them.

## Coming from LlamaCloud

The concepts map one-to-one:

| LlamaCloud | ingestlib |
|---|---|
| Classification rules (20 max) | `classify.rules` (20 max) |
| Split categories (50 max) | `split.categories` (50 max) |
| "Uncategorized pages" dropdown | `split.unmatched: require · other · skip` (default group-as-Other) |
| Inline `configuration` on an API call | Per-call arguments |
| Saved `configuration_id` | `rules.yaml` |
