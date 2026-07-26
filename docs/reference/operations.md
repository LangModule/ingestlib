# Operations API

The three pipeline stages. Each has a sync form and an async `a`-prefixed
form — use the async form inside a running event loop. `parse` is the only
operation that needs the OCR server; `classify` and `split` accept either a
`ParseResult` or a raw file path.

```python
from ingestlib.operations import parse, classify, split
from ingestlib.operations import aparse, aclassify, asplit
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
