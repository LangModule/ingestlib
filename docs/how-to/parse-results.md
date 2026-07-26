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

## Persist and reload

Parsing is the expensive stage — store the result once, reuse it forever:

```python
from ingestlib.storage import artifacts

doc_id = artifacts.save_parse(result)          # everything, keyed by checksum

light = artifacts.load_parse(doc_id)                       # structure only
full  = artifacts.load_parse(doc_id, include_images=True)  # + page renders & crops
```

The light form loads with `image_bytes=None` on pages and empty bytes on
figure crops — cheap when you only need text and structure. `ingest()`
does this save automatically; loading an unknown `doc_id` raises a clear
error pointing you at `artifacts.list_documents()`.

## Feed the other operations

```python
label  = classify(result)                        # no OCR — reuses the parse
chunks = split(result, category=label.category)  # chunks with region provenance
```

Passing the `ParseResult` (rather than the file path) is what gives split
exact region ids — the standalone-path chunks cite pages only.

---

Next: [Your categories & sections](content-rules.md) — constrain classify
and split to your domain.
