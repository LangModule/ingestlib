# Services API

The composed flows. Sync and async forms of each — use the async form
inside a running event loop.

```python
from ingestlib.services import ingest, retrieve
from ingestlib.services import aingest, aretrieve
```

## ingest

::: ingestlib.services.ingest.ingestor.aingest

::: ingestlib.services.ingest.ingestor.ingest

## retrieve

::: ingestlib.services.retrieve.retriever.aretrieve

::: ingestlib.services.retrieve.retriever.retrieve

## Lifecycle

Manage the corpus as files change — replace, remove, sync, backfill. See
[Manage a corpus](../how-to/manage-corpus.md) for the guide.

```python
from ingestlib.services import remove, sync, backfill
```

::: ingestlib.services.lifecycle.remover.remove

::: ingestlib.services.lifecycle.syncer.sync

::: ingestlib.services.lifecycle.backfiller.backfill
