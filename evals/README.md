# ingestlib evaluation suite

Measures ingestlib's capabilities against **published public benchmarks** — the
same datasets the document-AI models report on — so the numbers are comparable
to known systems and reproducible by anyone. Gold sets we build ourselves plug
in later through the same harness.

This is measurement, **never assertion**. Scores drift with models, prompts, and
data; a snapshot records where things stand — it does not gate CI.

## What gets measured, and with what data

| Capability | Public dataset | Metric | Needs the OCR server (GPU)? |
|---|---|---|---|
| **Chart → table** (the enricher — headline) | ChartX (structural extraction), PlotQA | RNSS + numeric F1 (RMS-style) | No — LLM-vision only |
| **Parse** — page → markdown | olmOCR-Bench, OmniDocBench | unit-test pass rate; text edit-distance | Yes |
| **Tables** | FinTabNet (financial), PubTabNet | TEDS / S-TEDS | Yes |
| **Formulas** | UniMER-Test, im2latex | CDM | Yes |
| **text2SQL** | Spider / BIRD | execution-match | No (needs a DB) |

`parse`/`tables`/`formulas` run the full parse pipeline, which needs the
PaddleOCR-VL inference server on a GPU — run those on the GPU box. `chart→table`
and `text2SQL` are GPU-free (LLM + optional DB) and run anywhere.

## Layout

```
evals/
  harness.py          shared: load → run → score → snapshot to results/
  datasets/           loaders — read each public dataset into a normalized sample list
  adapters/           run ingestlib over a sample's input → a normalized prediction
  scorers/            the metric per capability (transparent; wrap official where noted)
  run_chart_eval.py   one runner per capability
  run_parse_eval.py
  run_table_eval.py
  run_formula_eval.py
  report.py           aggregate results/ snapshots → one report
  data/               downloaded datasets (gitignored — fetch per the table below)
  results/            timestamped JSON snapshots
```

Each runner is invoked as a module so intra-suite imports resolve:

```bash
uv run python -m evals.run_chart_eval --dataset data/chartx/manifest.jsonl
```

## Getting the data

Public datasets are **not** vendored — they download into `evals/data/`
(gitignored). Each `datasets/<name>.py` documents its fetch (Hugging Face id or
URL) and normalizes the raw format into the harness's sample shape, so a runner
never touches a dataset's idiosyncratic layout directly.

## Design rules

- **Measures, never asserts** — no eval is a test; the fast suite stays separate.
- **Public data + official metrics** — reproducible and comparable; we wrap the
  reference scorer where one exists rather than reinvent it.
- **Normalized samples** — every dataset loader emits the same small shape, so
  adapters and scorers are dataset-agnostic; only the loader knows the raw format.
- **Snapshots carry their provenance** — provider, models, and date, because a
  score means nothing without the config that produced it.

## Runners

| Command | Capability | Data to fetch first |
|---|---|---|
| `make eval-chart` | chart→table (GPU-free) | ChartX/PlotQA → `data/chartx/manifest.jsonl` |
| `make eval-parse` | page markdown (GPU) | OmniDocBench/olmOCR → `data/omnidocbench/manifest.jsonl` |
| `make eval-table` | table cells (GPU) | FinTabNet/PubTabNet → `data/fintabnet/manifest.jsonl` |
| `make eval-formula` | LaTeX (GPU) | UniMER/im2latex → `data/unimer/manifest.jsonl` |
| `make eval-sql` | text2SQL (GPU-free) | Spider/BIRD → sources.yaml + `data/spider/dataset.yaml` |
| `make eval-report` | combined scorecard | — |

## Status

All runners built and importing against the real ingestlib paths. Each needs its
public dataset converted into the manifest shape its loader documents; then run
the command above. Gold sets we build ourselves plug in through the same loaders.
