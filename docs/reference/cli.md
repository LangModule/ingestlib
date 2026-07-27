# CLI

Installed with the package as the `ingestlib` command
(`uv run ingestlib …`; `python -m ingestlib` is equivalent).
`ingestlib --version` prints the installed version.

| Command | Purpose |
|---|---|
| `init` | scaffold config.yaml (+ .env) |
| `doctor` | verify the configured stack with real calls |
| `ingest` | index files or folders |
| `sync` | reconcile a folder with the corpus |
| `list` | show every stored document |
| `remove` | erase a document (by path or doc_id) |
| `backfill` | rebuild the vector store from stored artifacts |
| `search` | cited retrieval from the shell |

`init` and `doctor` set up and verify a stack; the rest manage the documents in
it — the corpus-management guide is [Manage a corpus](../how-to/manage-corpus.md).
Every corpus command takes `--namespace`.

## `ingestlib init`

Writes the setup files into the **current directory** — where
[config discovery](../concepts/configuration.md#discovery) looks.

```bash
uv run ingestlib init [--local] [--force]
```

| Flag | Effect |
|---|---|
| *(none)* | Default stack: Bedrock + sqlite + Jina. Writes `config.yaml` (with an `aws:` section to fill) and `.env` (with key slots). |
| `--local` | Zero-cloud stack: Ollama + sqlite + local artifacts + no reranker. Writes `config.yaml` only — no keys are needed, so no `.env`. |
| `--force` | Overwrite existing files. Without it, init refuses and exits 1 if `config.yaml` or `.env` already exists. |

Output ends with the preset's next steps (models to pull, keys to fill,
the OCR server command, and `ingestlib doctor`).

## `ingestlib doctor`

Verifies the configured stack with **real calls** — a chat round-trip, an
actual embedding, a store reachability probe. Checks only what your
config.yaml choices require.

```bash
uv run ingestlib doctor
```

| Check | Failure mode |
|---|---|
| Python version | `fail` below 3.12 |
| Config discovery + parse | `fail` — and doctor stops; nothing else can run |
| LibreOffice | `warn` — DOCX/PPTX need it; PDF/images don't |
| OCR server | `warn` — parse/ingest need it; classify/split/extract/retrieve don't |
| LLM provider | `fail` — real chat call |
| Embedding provider | `fail` — real embedding; prints the dimension |
| Reranker | `fail` (`skip` when `reranker: none`) |
| Artifact store | `fail` — S3 credentials + bucket, or local-folder writability |
| Vector store | `fail` — liveness probe; never creates indexes or schema |

Marks: `✓` ok · `!` warning · `✗` failure · `-` skipped.

**Exit code** `0` when nothing failed (warnings allowed), `1` otherwise —
safe to gate a deployment script on.

Every failure line carries the same one-sentence fix the library raises at
runtime — the [troubleshooting catalog](../how-to/troubleshooting.md)
lists them all.

## Costs

Doctor's LLM and reranker probes are real calls: fractions of a cent on
cloud providers, free on Ollama. Local checks (config, LibreOffice,
sqlite, folders) are free.

## `ingestlib ingest`

Index one or more files or folders. A folder expands to its ingestible files
(PDF/DOCX/PPTX/PNG/JPEG/WebP, recursive).

```bash
ingestlib ingest report.pdf
ingestlib ingest docs/ contracts/ invoice.pdf   # mix files and folders
ingestlib ingest report.pdf --namespace tenant-a
```

Prints a per-stage progress line and the outcome per file (`ingested`,
`replaced`, `moved`, or `skipped` — see [Manage a corpus](../how-to/manage-corpus.md#replace-just-re-ingest)).
Exit `1` if any file failed; the rest still process.

## `ingestlib sync`

Reconcile a folder with the corpus: new files ingest, edited files replace,
renamed files move, and — with `--prune` — deleted files are removed.

```bash
ingestlib sync corpus/                     # ingest/replace/move
ingestlib sync corpus/ --dry-run           # print the plan, change nothing
ingestlib sync corpus/ --prune             # also delete gone files
ingestlib sync corpus/ --prune --dry-run   # preview the prune first
```

| Flag | Effect |
|---|---|
| `--prune` | delete documents whose file is gone (root-scoped; refuses on an empty scan) |
| `--dry-run` | print the plan, execute nothing |
| `--namespace` | reconcile one partition |

Always `--dry-run` before a first `--prune`. Guardrails are detailed in
[Pruning safely](../how-to/manage-corpus.md#pruning-safely). Exit `1` if any
file errored.

## `ingestlib list`

The registry as a table — doc_id (short), pages, chunks, category, namespace,
source path.

```bash
ingestlib list
ingestlib list --namespace tenant-a        # one partition (default: all)
```

## `ingestlib remove`

Erase one document from **both** stores (vectors, then artifacts).

```bash
ingestlib remove report.pdf                # by source path
ingestlib remove 7b6b95d79149              # by doc_id (full or a unique prefix)
```

Exit `1` when the target matches nothing (or a prefix is ambiguous) — nothing
is deleted.

## `ingestlib backfill`

Rebuild the vector store from stored split artifacts — no re-parse. For a
provider switch, a new store connector, or a wiped index (see
[backfill](../how-to/manage-corpus.md#rebuild-the-vector-store-backfill)).

```bash
ingestlib backfill
ingestlib backfill --namespace tenant-a
```

## `ingestlib search`

Cited retrieval from the shell — the ingest→verify loop without a Python
session.

```bash
ingestlib search "what were the risks?"
ingestlib search "revenue growth" --top-k 10
ingestlib search "invoice INV-20114" --no-rerank
```

Prints ranked hits with score, citation, heading, and a snippet.
