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

## Extract

::: ingestlib.operations.extract.models.ExtractResult

::: ingestlib.operations.extract.models.ExtractedItem

::: ingestlib.operations.extract.models.FieldValue

## Ingest

::: ingestlib.services.ingest.models.IngestResult

## Retrieve

::: ingestlib.services.retrieve.models.RetrievalResult

::: ingestlib.services.retrieve.models.Hit

## Lifecycle

::: ingestlib.services.lifecycle.models.RemoveResult

::: ingestlib.services.lifecycle.models.SyncResult

::: ingestlib.services.lifecycle.models.SyncAction

::: ingestlib.services.lifecycle.models.BackfillResult

## OCR primitives

::: ingestlib.foundations.ocr.models.Region

::: ingestlib.foundations.ocr.models.BoundingBox
