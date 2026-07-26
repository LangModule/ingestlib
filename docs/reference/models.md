# Result models

The typed results every call returns. All are frozen Pydantic models —
serializable with `.model_dump()` / reconstructable with
`.model_validate()`.

## Parse

::: ingestlib.operations.parse.models.ParseResult

::: ingestlib.operations.parse.models.PageResult

::: ingestlib.operations.parse.models.FigureImage

## Classify

::: ingestlib.operations.classify.models.ClassifyResult

::: ingestlib.operations.classify.models.CategoryScore

## Split

::: ingestlib.operations.split.models.SplitResult

::: ingestlib.operations.split.models.Section

::: ingestlib.operations.split.models.Chunk

## Ingest

::: ingestlib.services.ingest.models.IngestResult

## Retrieve

::: ingestlib.services.retrieve.models.RetrievalResult

::: ingestlib.services.retrieve.models.Hit

## OCR primitives

::: ingestlib.foundations.ocr.models.Region

::: ingestlib.foundations.ocr.models.BoundingBox
