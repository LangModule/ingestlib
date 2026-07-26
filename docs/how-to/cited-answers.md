# Build cited answers

Retrieval gives you chunks that know where they came from. This page turns
that into (1) an LLM answer with citations and (2) a viewer that highlights
the source on the page.

## 1. A cited LLM answer

`result.context` is the prompt block — numbered chunks, each tagged with
its citation:

```python
from ingestlib.services import retrieve

result = retrieve("how were participants recruited?", top_k=5)
print(result.context)
```

```text
[1] (doc 7b6b95d79149 · p.4 · methods)
Participants were recruited through community centers in Cairo…

[2] (doc 7b6b95d79149 · p.7 · results)
Of the 214 participants enrolled…
```

Hand it to any LLM with an instruction to cite:

```python
from ingestlib.foundations.llm import chat

answer = chat(
    f"Answer using ONLY the context. Cite sources as [n].\n\n"
    f"Context:\n{result.context}\n\nQuestion: {result.question}"
)
# "Participants were recruited through community centers [1], with 214 enrolled [2]."
```

Your application resolves `[n]` back to `result.hits[n-1]` — and from
there to a page.

## 2. From a hit to a page highlight

Each hit carries `pages` and `region_ids` — the exact parse regions the
chunk covers. Resolve them against the stored parse:

```python
from ingestlib.storage import artifacts

hit = result.hits[0]
doc_id = hit.chunk.document_id

parse = artifacts.load_parse(doc_id)          # structure only — fast

for page_num, region_ids in hit.chunk.region_ids.items():
    page = next(p for p in parse.pages if p.page_num == page_num)
    boxes = [r.bbox for r in page.regions if r.region_id in region_ids]
    print(f"page {page_num}: {[b.as_tuple() for b in boxes]}")
# page 4: [(72.0, 160.4, 523.1, 380.9), (72.0, 390.0, 523.1, 512.6)]
```

## 3. Serve the page image

The rendered page PNGs live in the artifact store:

```python
key = artifacts.page_image_key(doc_id, page_num)
png_bytes = artifacts.read_blob(key)           # works on s3 AND local
```

`read_blob` is backend-agnostic; on S3 you can instead presign the key
(`get_s3_client().generate_presigned_url`) and let the browser fetch it
directly.

Draw the boxes over the PNG — `bbox.as_tuple()` is `(x0, y0, x1, y1)` in
page-render pixel coordinates — and the answer literally points at its
source.

## Why this works reliably

Two invariants protect the chain (details in
[Provenance & citations](../concepts/provenance.md)):

- chunk boundaries never cut through a region, so `region_ids` is exact
- the payload rides on the vector, so no join against a second database
  can drift out of sync

---

Next: [Run fully local](local-stack.md) — the same pipeline with nothing
leaving your machine.
