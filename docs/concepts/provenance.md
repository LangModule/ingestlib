# Provenance & citations

The differentiating promise of ingestlib: **every retrieved chunk can point
at the exact place on the exact page it came from.** This page explains the
chain that makes that work.

## The chain

```text
retrieve() hit
   └─ chunk ──────────────  pages: [4]        region_ids: {4: [2, 3]}
        └─ parse/result.json (artifact store)
             └─ page 4, regions 2 & 3 ──────  bounding boxes
                  └─ parse/pages/page_0004.png  (the rendered page)
```

Four links, each stored, none inferred:

1. **A hit carries its chunk** — with `document_id`, `pages`, and
   `region_ids` (a map of page number → the parse region ids the chunk
   covers).
2. **`document_id` keys the artifact store** — `load_parse(doc_id)` returns
   the full parse structure without re-running anything.
3. **Regions have bounding boxes** — every block parse produced (paragraph,
   table, chart, figure) has a `region_id` and a bbox on its page.
4. **Page renders are stored as PNGs** — so a UI can draw the bbox on the
   actual page image.

## Identity: the content checksum

`doc_id` is the SHA-256 of the file's bytes, computed at parse time. It's
the key for everything: artifact layout, vector ids
(`{doc_id}:{chunk_id}`), dedup, and citations. Two identical files are one
document; a one-byte edit is a different document.

## Citations for humans

`Hit.citation` renders the chain as a one-liner:

```text
doc 7b6b95d79149 · p.4 · methods
```

`RetrievalResult.context` assembles all hits into a numbered, cited block
you can hand directly to an LLM — so the *model's* answer can reference
`[2]` and your application can resolve `[2]` back to a page region:

```text
[1] (doc 7b6b95d79149 · p.4 · methods)
Participants were recruited through community centers in Cairo…

[2] (doc 7b6b95d79149 · p.7 · results)
…
```

## Citations for UIs

Everything a click-through viewer needs is two calls away:

```python
from ingestlib.storage import artifacts

parse = artifacts.load_parse(hit.chunk.document_id)
page = next(p for p in parse.pages if p.page_num == 4)
boxes = [r.bbox for r in page.regions if r.region_id in hit.chunk.region_ids[4]]

png = artifacts.read_blob(artifacts.page_image_key(hit.chunk.document_id, 4))
```

Draw `boxes` over `png` and the answer highlights its source on the page.
The full recipe is in [Build cited answers](../how-to/cited-answers.md).

## Why chunks stay traceable

Two design rules protect the chain:

- **Chunks never split tables or figures** — a chunk boundary can't fall
  inside a region, so `region_ids` is always exact, never approximate.
- **Vector payloads are complete** — the chunk's content and provenance
  ride on the vector itself, so retrieval never depends on a second
  database to resolve what it found.

---

Next: [Configuration model](configuration.md) — how choices, secrets, and
content rules are organized.
