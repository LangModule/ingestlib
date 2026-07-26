# CLI

Installed with the package as the `ingestlib` command
(`uv run ingestlib …`; `python -m ingestlib` is equivalent).
`ingestlib --version` prints the installed version.

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
| OCR server | `warn` — parse/ingest need it; classify/split/retrieve don't |
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
