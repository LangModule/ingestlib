# Async, notebooks & logging

## Every call has two forms

Each operation and service ships a sync form and an `a`-prefixed async
form — same signature, same behavior:

| Sync | Async |
|---|---|
| `parse` | `aparse` |
| `classify` | `aclassify` |
| `split` | `asplit` |
| `ingest` | `aingest` |
| `retrieve` | `aretrieve` |

Use the async forms anywhere an event loop is already running — web
handlers, agents, concurrent batch jobs:

```python
from ingestlib.services import aingest, aretrieve

result = await aingest("report.pdf")
hits = await aretrieve("what changed year over year?")
```

Internally the pipeline is async-first: chunk embedding runs
bounded-parallel, and blocking store/artifact calls are kept off the event
loop.

## Notebooks

Jupyter runs an event loop, so the **sync forms refuse** with an error that
names the fix:

```python
ingest("report.pdf")
# RuntimeError: you are inside a running event loop (a notebook?) —
# call `await aingest(...)` instead of the sync form
```

Notebooks can `await` at the top level — just use the async form directly:

```python
result = await aingest("report.pdf")
```

## Concurrency notes

- Safe: concurrent `aretrieve()` calls; ingesting **different** documents
  concurrently.
- Pointless: ingesting the same file concurrently (checksum dedup makes
  one of them a no-op, but they'll race the same work first).
- Parse itself processes one document's pages as a unit — parallelism
  belongs at the document level.

## Logging

ingestlib logs every stage to stderr through its own logger namespace —
INFO shows real progress (per-stage timing, chunk counts, upsert sizes):

```text
14:32:07 INFO  ingestlib.operations.parse.pipeline   parse start: report.pdf (24 pages)
14:32:48 INFO  ingestlib.services.ingest.ingestor    ingest done: report.pdf → research_paper, 24 chunk(s) in 41.2s
```

Control it with environment variables:

| Variable | Effect |
|---|---|
| `INGESTLIB_LOG_LEVEL` | `DEBUG` · `INFO` (default) · `WARNING` · `ERROR` |
| `INGESTLIB_LOG_THIRD_PARTY=1` | also raise SDK loggers (httpx, botocore, …) to the same level |
| `INGESTLIB_LOG_COLOR=0` | disable ANSI colors (auto-detects TTY otherwise) |

Or reconfigure at runtime — idempotent, never duplicates handlers:

```python
from ingestlib.utils.logger import configure

configure(level="DEBUG")
```

Noisy third-party SDKs are quieted to WARNING by default — but a level
your application set explicitly is always respected, and ingestlib's
logger never propagates into your root logger's handlers.

---

Next: [Troubleshooting](troubleshooting.md) — the error catalog.
