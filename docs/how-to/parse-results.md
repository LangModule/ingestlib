# Use parse results

A `ParseResult` holds everything the parser saw: markdown, structure,
figures, and page renders. This page shows how to get each out.

## Whole-document markdown

```python
result = parse("report.pdf")
print(result.markdown)          # every page, assembled
```

Tables come out as HTML (merged cells preserved), formulas as LaTeX,
charts as markdown data tables with a caption. This single string is often
all you need if you're feeding another system.

## Per-page structure

```python
for page in result.pages:
    print(page.page_num, len(page.regions), "regions")
    print(page.markdown)        # this page only
```

Every block on a page is a `Region` with a type, its recognized content,
and a bounding box:

```python
for region in result.pages[0].regions:
    print(region.region_id, region.region_type, region.bbox.as_tuple())
    # 0 text     (72.0, 90.5, 523.0, 140.2)
    # 1 table    (72.0, 160.0, 523.0, 380.9)
    # 2 chart    (90.0, 400.0, 500.0, 640.0)
```

Region ids are what chunks cite later — the
[provenance chain](../concepts/provenance.md) hangs off them.

## Figures and charts

Charts and figures are cropped out as PNGs with captions and AI
descriptions:

```python
for page in result.pages:
    for fig in page.figures:
        print(fig.caption, "—", fig.description[:60])

result.save_images("out/")      # writes every crop as a PNG file
```

## Page renders

Each page keeps its rendered image (`page.image_bytes`, PNG) — the canvas
for drawing citation highlights.

## Reuse a parse

Parsing is the expensive stage, so reuse the `ParseResult` object — feed it
straight to the other operations (below), no re-parse. Once a document is
**ingested**, its structure lives in the registry and reads back with
`get_document`:

```python
from ingestlib.services import get_document

doc = get_document(doc_id)          # a stored (ingested) document, from the registry
doc.pages, doc.regions, doc.chunks  # parsed structure, as plain dicts
doc.markdown()                      # whole-document markdown (from the artifact store)
doc.page_image(1)                   # a page render on demand (PNG bytes)
```

`get_document` reads structure from the registry; bytes (page renders,
whole-doc markdown) are fetched lazily from the artifact store. An unknown or
tombstoned `doc_id` returns `None`.

## Feed the other operations

```python
label  = classify(result)                        # no OCR — reuses the parse
chunks = split(result, category=label.category)  # chunks with region provenance
```

Passing the `ParseResult` (rather than the file path) is what gives split
exact region ids — the standalone-path chunks cite pages only.

---

Next: [Extract structured data](extract.md) — your schema, filled and
cited from the parse.
