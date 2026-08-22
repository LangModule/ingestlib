# Query databases (SQL)

`retrieve()` can answer from your **SQL databases** alongside the document
corpus — one call, one merged result. A natural-language question becomes a
read-only SQL query, executed inside a permission boundary, and returned in
the same shape as a document hit.

```python
from ingestlib.services import retrieve

result = retrieve("how many prescriptions are ready?", sources=["prescriptions"])
for r in result.results:
    print(r.source, r.source_type, "→", r.content)
    print(r.provenance["sql"])          # the exact query that ran
```

Documents and databases compose in the *same* call — ask one question, get
answers from both:

```python
retrieve("is rx 4471 ready, and what does its insert say about dosage?",
         sources=["prescriptions", "package_inserts"])
```

## The permission model

Generated SQL is bounded, never trusted. Four layers, defense in depth:

1. **A read-only database role** — the hard floor. Point ingestlib at a role
   that *cannot* write, and no generated query can either, whatever it says.
2. **A statement allowlist** (`allow`) — only these statement types run; a
   generated `DELETE`/`DROP` is rejected before it reaches the driver.
3. **A row limit** — a `LIMIT` is injected when the query has none, so a
   query can never scan unbounded.
4. **A timeout** — a runaway query is killed.

!!! warning "Always use a read-only role"

    The allowlist and limits are defense in depth; the read-only role is the
    floor that holds even if they don't. Create a dedicated read-only user for
    ingestlib and put *that* connection URL in `.env`.

## Declare your sources

Structured retrieval reads a `sources.yaml` sidecar beside `config.yaml`
(like [`rules.yaml`](content-rules.md)) — create it only to use this feature;
there are no default sources. Connection URLs are secrets: set them in `.env`
and reference them as `${VAR}`. The full annotated reference is
[`sources.example.yaml`](https://github.com/LangModule/ingestlib/blob/main/sources.example.yaml).

```yaml
# sources.yaml
prescriptions:
  type: postgres                  # postgres | mysql | sqlite | duckdb | snowflake
  dsn: ${RX_DB_DSN}               # a READ-ONLY connection URL, set in .env
  description: "Prescription fulfillment — status, ready times, patients, drugs."
  allow: [select]                 # statement types the model may generate
  row_limit: 1000                 # cap rows returned
  timeout: 10                     # seconds before a query is killed
  tables:                         # schema hints — the accuracy lever
    rx: "one row per prescription — rx_id, patient_id, drug_name, status, ready_at"
    patients: "prescription owners — patient_id, name, dob, plan_id"

package_inserts:
  type: documents                 # your existing ingestlib corpus
  namespace: package_inserts      # which partition to search
```

```bash
# .env — the connection URL is a secret, never in sources.yaml
RX_DB_DSN=postgresql://ro_user:pw@host:5432/rx
```

Each SQL backend needs its
[pip extra](../get-started/installation.md#1-install-the-package):

```bash
uv add "ingestlib[postgres]"      # or mysql · duckdb · snowflake · sqlite (no extra)
```

`tables` hints are the biggest accuracy lever — a one-line description of what
each table holds steers generation far more than the column names alone. Write
them like you'd brief an analyst.

## Verified queries — for answers that must be exact

Generation is best-effort. For the questions you can't afford to get wrong,
declare a **verified query**: reviewed SQL that runs whenever a question
matches its description, instead of a generated one.

```yaml
prescriptions:
  # ... type, dsn, tables as above ...
  verified:
    rx_status:
      description: "Fulfillment status and ready time for a prescription"
      sql: "SELECT status, ready_at FROM rx WHERE rx_id = :rx_id"
      params: [rx_id]
```

A question close enough to `description` (semantic match) runs the reviewed
`sql`; the model only fills the named `params` from the question. Everything
else falls through to generation. `result.results[0].provenance["verified"]`
tells you which path answered.

## Reading the result

`sources=` returns normalized `SourceResult`s in `result.results` (not
`result.hits`), and `result.context` renders them prompt-ready — so a
[cited answer](cited-answers.md) reads databases and documents the same way:

```python
result = retrieve("which prescriptions are ready today?", sources=["prescriptions"])
r = result.results[0]

r.content                 # the rows, rendered for a prompt
r.source_type             # "structured" (a database) | "documents" (the corpus)
r.provenance              # {"sql": "...", "params": {...}, "verified": False}
r.raw                     # {"columns": [...], "rows": [...]} — the raw result
```

## From the shell and over MCP

```bash
ingestlib search "how many are ready?" --sources prescriptions,package_inserts
```

The [MCP `search` tool](mcp-server.md) takes the same `sources` argument, so
an agent can query your databases read-only — SQL results carry their exact
query in `provenance`. Structured retrieval is read-only, so it stays a *read*
tool, available even under `--read-only`.

`ingestlib doctor` health-checks every declared source (a `SELECT 1` per SQL
source) once `sources.yaml` exists.

## Honest scope

- **Generated SQL is only as good as the model and your hints.** On a clean,
  well-described schema it is strong; on a sprawling schema with cryptic
  columns and heavy joins, accuracy drops — the documented reality of
  text-to-SQL. Use `tables` hints, and reach for `verified:` on the queries
  that must be right.
- **A wrong query that *errors* is retried once** (the error is fed back to the
  model); a query that runs but returns the *wrong* number looks identical to a
  right one — no loop catches that. Treat generated answers as an analyst
  draft, not gospel, until you've measured accuracy on your own schema.
- **The read-only role is the safety guarantee** — the worst case is a wrong
  *read*, never a damaged database.

---

Next: [Build cited answers](cited-answers.md) — turn results (from documents
or databases) into an answer that cites its sources.
