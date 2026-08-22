# Troubleshooting

ingestlib's errors are written to name their own fix — this page is the
catalog. First move for any setup problem:

```bash
uv run ingestlib doctor
```

It probes every configured choice with a real call and prints the failing
piece with its remedy. What follows is the same knowledge, organized for
searching.

## Configuration

| You see | It means / do this |
|---|---|
| `config.yaml not found in … or any parent directory` | You're outside the project directory. `cd` to it, run `uv run ingestlib init`, or point `INGESTLIB_CONFIG` at the file. |
| `config.yaml has no 'aws' section, but these choices use AWS: llm_provider: bedrock (the default)…` | Either add `aws: {profile, region, account_id}` or pick the non-AWS options it lists (openai/ollama providers, local artifacts). |
| `config.yaml's 'aws' section is missing profile…` | Complete the section — or delete it entirely if nothing uses AWS. |
| `unknown vector_store 'x' — choose one of […]` | Typo'd choice; the message lists every valid option. |
| `the qdrant connector needs its SDK … pip install "ingestlib[qdrant]"` | Server-backed stores are [pip extras](../get-started/installation.md#1-install-the-package) — run exactly the command the error prints. |

## AWS & Bedrock

| You see | It means / do this |
|---|---|
| `AWS profile 'x' … not found in ~/.aws — available profiles: ['…']` | The profile name in `config.yaml` doesn't exist. Use one from the printed list, or leave it empty for the default credential chain. ingestlib never falls back silently — a typo must not spend against the wrong account. |
| `AWS session expired or no credentials — run 'aws sso login --profile x'` | Exactly that. |
| `enable model access in the Bedrock console (Model access page, region …)` | Your account hasn't enabled the Nova models — it's a console toggle, separate from IAM. |
| `IAM denied the call — attach bedrock:InvokeModel permission…` | The profile's identity lacks the permission (this one *is* IAM). |
| `S3 bucket 'x' already exists in another AWS account (bucket names are global)…` | S3 names are global across **all** accounts. Pick a unique `s3.bucket` in config.yaml. |

## OpenAI & Jina

| You see | It means / do this |
|---|---|
| `OpenAI rejected the API key` | Check `OPENAI_API_KEY` in `.env` — new keys at platform.openai.com. |
| `…insufficient_quota — this is billing, not rate limiting` | Add credit to the OpenAI account; retrying won't help. |
| `Jina rejected the API key (401)` | Check `JINA_API_KEY`; new keys at jina.ai/api-dashboard. |
| `Jina token balance exhausted (402)` | The free tier ran out — top up, get a new key, or set `reranker: none`. |
| `Jina rate limit held through N attempts (429)` | Free tier allows 100 requests/minute; wait and retry. |

## Ollama

| You see | It means / do this |
|---|---|
| `could not reach … — start Ollama ('ollama serve'…)` | The server isn't running (or `ollama.base_url` points at the wrong place). |
| `model 'x' not found — 'ollama pull x'` | Pull the exact model the error names. |
| Charts come back with wrong values, no error | You're on an `-mlx` build — it silently drops images. Use the GGUF build ([why](local-stack.md#the-rules-that-keep-it-working)). |
| `the Ollama runner keeps dropping requests…` | The server is memory-pressured (usually two models resident). Transient drops are retried automatically; if it persists, `ollama ps` shows what's loaded, `ollama stop <model>` frees one, or restart Ollama. |

## Parsing & documents

| You see | It means / do this |
|---|---|
| `OCR server is not answering at …` | Start it ([command](../get-started/installation.md#3-the-ocr-inference-server)). Just started it? The first launch downloads ~1.8 GB — wait for loading to finish. |
| `LibreOffice is not installed ('soffice' not found)…` | Needed for DOCX/PPTX only; the error includes install commands. PDFs and images work without it. |
| `this PDF is password-protected…` | Remove the password (open and print/export to a new PDF). |
| `not a readable PDF — the file appears corrupt or truncated` | Re-export or re-download the file. |
| `x has no extractable text … Run parse() first` | You called `classify()`/`split()`/`extract()` directly on a scan or image — they read native text. Parse it (OCR), then pass the `ParseResult`. |

## Structured retrieval (SQL)

| You see | It means / do this |
|---|---|
| `SQL source DSN is unset — its ${VAR} did not resolve; set the connection URL in .env` | The `${VAR}` in `sources.yaml`'s `dsn:` isn't defined in `.env`. Set it — to a **read-only** connection URL ([guide](structured-retrieval.md#the-permission-model)). |
| `unknown source 'x' — declare it in sources.yaml` | The name passed to `retrieve(sources=[...])` isn't in `sources.yaml`; the message lists the configured ones. |
| `unknown SQL source type 'x' — one of […]` | `sources.yaml`'s `type:` must be `postgres`/`mysql`/`sqlite`/`duckdb`/`snowflake`/`documents`. |
| `Can't load plugin` / `No module named 'psycopg'` (or `pymysql`, …) | The SQL backend's driver isn't installed — `pip install "ingestlib[postgres]"` (or `[mysql]`/`[duckdb]`/`[snowflake]`). |
| `statement type 'delete' is not allowed for this source` | A generated or verified query used a statement outside `allow` — the guardrail did its job. Widen `allow` only if you mean to (and never past what the read-only role permits). |

## Runtime

| You see | It means / do this |
|---|---|
| `you are inside a running event loop (a notebook?) — call 'await aingest(...)'` | Use the async form ([details](async-and-logging.md#notebooks)). |
| `no parse artifact stored for doc_id '…' — run parse() … list_documents()` | That checksum was never ingested into *this* artifact store — `artifacts.list_documents()` shows what is. |
| `this Python was built without SQLite extension support…` | macOS system Python can't load sqlite-vec — use a python.org, homebrew, or uv-managed interpreter. |
| `retrieve()` returns no hits | Not an error — see [the checklist](retrieve.md#no-hits). |

## Still stuck?

Set `INGESTLIB_LOG_LEVEL=DEBUG` for full detail, and open an issue with
the log: [github.com/LangModule/ingestlib/issues](https://github.com/LangModule/ingestlib/issues).
