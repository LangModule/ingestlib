# Manage a corpus

Real corpora change: files get edited, renamed, added, and deleted. ingestlib
keeps the stores matching what's on disk — a new version **replaces** the old
one instead of piling up beside it, and a folder **syncs** in one call.

## The identity model

Two identities, and the difference is the whole point:

- **Content identity** — `doc_id = sha256(bytes)`. A one-byte edit is a new
  `doc_id`.
- **Logical identity** — `(namespace, source path)`. The file *at this path*,
  whatever its current bytes.

Lifecycle operates on the logical identity. Re-ingesting an edited file at the
same path replaces the previous version; the old vectors and artifacts are
deleted, so retrieval never returns stale content.

## Replace: just re-ingest

`ingest()` detects that this path already holds an older version and replaces
it — no flag required:

```python
from ingestlib.services import ingest

ingest("report.pdf")                 # first time → status="ingested"
# ... the file is edited ...
r = ingest("report.pdf")             # → status="replaced"
r.replaced_doc_id                    # the old version, now fully deleted
```

The new version is embedded and upserted **before** the old one is removed, so
a query during the swap always finds one complete version — never zero. Statuses
you'll see:

| status | meaning |
|---|---|
| `ingested` | new document |
| `replaced` | this path held an older version; it was deleted after the new one went live |
| `moved` | same bytes arrived from a new path — only the registry was re-pointed, nothing re-ran |
| `skipped` | same bytes, same path, already fully ingested |

Renaming a file is a **move**, not a re-ingest: ingest sees the content is
already stored and just updates the recorded path (cheap, no pipeline). This is
also what keeps a later `sync(..., prune=True)` from deleting a file you merely
renamed.

Superseding a version that lives at a *different* path (rare) takes the explicit
override:

```python
ingest("report-v2.pdf", replaces=old_doc_id)
```

## Remove one document

```python
from ingestlib.services import remove

remove("report.pdf")          # by source path
remove("7b6b95d79149")        # or by doc_id (full or a unique prefix)
```

Erases the document from **both** stores — vectors first, then every artifact
(parse, renders, split, manifest). Returns a `RemoveResult` with the counts.

## Sync a folder

`sync()` makes the corpus match a directory in one call:

```python
from ingestlib.services import sync

result = sync("corpus/")
result.counts        # {'ingest': 3, 'replace': 1, 'move': 1, 'skip': 12}
```

It walks the folder, hashes each file, and reconciles against the registry:

| On disk | In the corpus | Action |
|---|---|---|
| present, unchanged | yes | skip |
| present, edited | yes | replace |
| present, renamed | yes (other path) | move |
| present | no | ingest |
| absent | yes | **prune** — only with `prune=True` |

### Preview first

```python
plan = sync("corpus/", prune=True, dry_run=True)
for action in plan.actions:
    print(action.action, action.path)
```

`dry_run=True` returns the full plan and changes nothing — always the safe first
move before a prune.

### Pruning safely

`prune=True` deletes documents whose file has disappeared. Three guardrails make
that safe:

- **Opt-in.** Off by default; a missing file is kept unless you ask.
- **Root-scoped.** Only documents whose recorded path lies *under the synced
  directory* (and in the synced namespace) can be pruned. Syncing `folder-a/`
  can never delete `folder-b/`'s documents. A file excluded by a narrow `glob`
  still exists, so it is never pruned either.
- **Empty-scan refusal.** If the scan finds no ingestible files at all while
  prune would delete documents — a wrong path, an unmounted drive — sync refuses
  instead of wiping the corpus.

```python
sync("corpus/", prune=True)                 # delete gone files
sync("corpus/", glob="**/*.pdf")            # only PDFs count
sync("corpus/", namespace="tenant-a")       # one partition
```

One bad file (a corrupt PDF) becomes an `error` action and sync continues —
`result.errors` collects them.

## Rebuild the vector store — `backfill()`

Artifacts are the source of truth; the vector store is an index over them.
`backfill()` re-embeds every stored document's chunks straight from the split
artifacts — **no re-parse, no OCR server** — so a corpus re-indexes in embedding
time, not pipeline time.

```python
from ingestlib.services import backfill

backfill()                                  # into the configured store
```

Reach for it when you:

- **switch `embedding_provider`** — new vector space, every document must be
  re-embedded ([why](../concepts/providers.md#switching-the-embedding-provider))
- **point at a new `vector_store`** connector
- **rebuild a wiped index**

Upserts are idempotent, so running it twice is safe. Documents parsed but never
split are skipped (they need a real ingest) and listed in the result.

## From the command line

Everything above has a CLI verb — see the [CLI reference](../reference/cli.md):

```bash
ingestlib ingest report.pdf docs/          # files or folders
ingestlib sync corpus/ --prune --dry-run   # preview
ingestlib sync corpus/ --prune             # execute
ingestlib list                             # the registry
ingestlib remove report.pdf                # erase one
ingestlib backfill                         # rebuild the index
ingestlib search "what were the risks?"    # cited retrieval
```

## One document, one namespace

A document belongs to a single namespace. Ingesting the same *content* into a
second namespace isn't supported — artifacts are keyed by content, so the two
would share one artifact prefix. Use a namespace per corpus/tenant and let each
hold its own documents.

On the CLI, `ingest`, `remove`, `sync`, `backfill`, and `search` all act on
**one** namespace — the `--namespace` you pass, or the unnamed default when you
omit it. `ingestlib list` is the exception: with no flag it shows **every**
namespace (pass `--namespace` to scope it). So if `ingest --namespace tenant-a`
was followed by a bare `remove report.pdf`, the remove looks in the default
namespace and reports nothing found — run `ingestlib list` to see which
namespace a document is in, then pass the matching `--namespace`.

Paths are recorded absolute. Pointing a second machine at a shared artifact
store will show the documents as "moved" on the first sync there — cheap,
self-healing metadata updates, nothing re-runs.
