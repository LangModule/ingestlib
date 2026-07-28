# Serve to agents (MCP)

`ingestlib mcp` exposes your corpus to any [MCP](https://modelcontextprotocol.io)
client — Claude Desktop, Claude Code, Cursor — as **tools**. An agent can then
run the whole loop against your **self-hosted** stack: ingest documents, search
with citations, extract fields, keep a folder in sync — nothing leaving your
machine.

It's an interface over the same functions the CLI and library use — install the
extra and point a client at it.

```bash
pip install "ingestlib[mcp]"
```

## The tools

| Tool | Does | Modifies corpus |
|---|---|---|
| `search` | hybrid retrieval + rerank → cited chunks | — |
| `extract` | fill a JSON-Schema you provide, with provenance | — |
| `classify` | document type + confidence | — |
| `list_documents` | what's stored (id, path, namespace, pages, category) | — |
| `doctor` | health-check the configured stack | — |
| `ingest` | add a document | ✎ |
| `sync` | reconcile a folder (new/replace/move/prune) | ✎ |
| `remove` | erase a document from both stores | ✎ |
| `backfill` | rebuild the vector store from artifacts | ✎ |

Read tools are always available; the ✎ (write) tools are hidden when you run
`--read-only`. Your MCP client also asks for approval before each call.

## Local clients — stdio

stdio is the default and needs no token. Point your client's MCP config at the
command:

```json
{
  "mcpServers": {
    "ingestlib": {
      "command": "ingestlib",
      "args": ["mcp"],
      "env": { "INGESTLIB_CONFIG": "/path/to/your/config.yaml" }
    }
  }
}
```

(That's the Claude Desktop / Cursor shape — a command the client launches and
talks to over stdio.) Now ask the agent things like *"what does my corpus say
about X?"* (→ `search`) or *"pull the totals from invoice.pdf"* (→ `extract`).

## Remote — streamable HTTP (with auth)

For an agent on another machine, run the HTTP transport. Because it's reachable
over the network **and can modify your corpus**, it's guarded:

- It **requires a bearer token** — set `MCP_TOKEN` in `.env` (any strong random
  string); the server refuses to start HTTP without one.
- It **binds `127.0.0.1` by default** — pass `--host 0.0.0.0` to expose it, and
  only behind a reverse proxy with TLS.

```bash
# .env
MCP_TOKEN=$(openssl rand -hex 32)
```

```bash
ingestlib mcp --transport http --port 8000
```

Clients send `Authorization: Bearer <MCP_TOKEN>`. Add `--read-only` to expose
only the read tools on a shared endpoint.

## Extract over MCP

`extract`'s schema is defined by the **agent** as a JSON Schema — no Python
class needed. The server turns it into a validated model and runs the real
extraction:

```json
{
  "type": "object",
  "properties": {
    "invoice_number": { "type": "string" },
    "total": { "type": "number", "description": "grand total in dollars" }
  },
  "required": ["invoice_number"]
}
```

Each returned field reports its pages, whether it was `grounded` in the source,
and an honest confidence — same guarantees as [`extract()`](extract.md).

## Configuration

Everything is optional (defaults shown):

```yaml
# config.yaml
mcp:
  read_only: false      # true → only the read tools
  host: 127.0.0.1       # streamable-http bind address
  port: 8000
```

```bash
# .env — only for the http transport
MCP_TOKEN=
```

The server uses your existing stack (vector store, providers, artifacts), so
`ingestlib doctor` verifies it just like the rest of the pipeline.

---

Next: [Reference → CLI](../reference/cli.md) for every `ingestlib mcp` flag.
