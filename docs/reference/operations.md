# Operations API

The four operations. Each has a sync form and an async `a`-prefixed
form — use the async form inside a running event loop. `parse` is the only
operation that needs the OCR server; `classify`, `split`, and `extract`
accept either a `ParseResult` or a raw file path.

```python
from ingestlib.operations import parse, classify, split, extract
from ingestlib.operations import aparse, aclassify, asplit, aextract
```

## parse

::: ingestlib.operations.parse.pipeline.aparse

::: ingestlib.operations.parse.pipeline.parse

## classify

::: ingestlib.operations.classify.classifier.aclassify

::: ingestlib.operations.classify.classifier.classify

## split

::: ingestlib.operations.split.splitter.asplit

::: ingestlib.operations.split.splitter.split

## extract

::: ingestlib.operations.extract.extractor.aextract

::: ingestlib.operations.extract.extractor.extract
