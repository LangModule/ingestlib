"""MCP server wiring — registration, read_only gating, http auth, transport
selection, and the CLI entrypoint. Always run: needs the mcp SDK (dev group),
no network — the http path is exercised via the middleware + arg plumbing,
never a live socket."""
import pytest

from ingestlib.config import MCPConfig
from ingestlib.mcp.server import _BearerAuth, build_server, serve


# ---------- tool registration + read_only ----------


async def test_full_server_registers_all_tools():
    tools = await build_server(read_only=False).list_tools()
    names = {t.name for t in tools}
    assert names == {"search", "ingest", "extract", "classify", "list_documents",
                     "get_document", "collections", "describe_schema", "remove", "sync",
                     "reindex", "recollect", "verify", "doctor", "registry_status",
                     "registry_init", "registry_backup", "registry_restore"}
    # descriptions come from the function docstrings
    search = next(t for t in tools if t.name == "search")
    assert "cited" in (search.description or "").lower()


async def test_read_only_hides_write_tools():
    tools = await build_server(read_only=True).list_tools()
    names = {t.name for t in tools}
    assert names == {"search", "extract", "classify", "list_documents", "get_document",
                     "collections", "describe_schema", "verify", "doctor", "registry_status"}
    assert not ({"ingest", "remove", "sync", "reindex", "recollect",
                 "registry_init", "registry_backup", "registry_restore"} & names)


async def test_tools_expose_input_schemas():
    """FastMCP introspects the type hints — an agent sees typed params."""
    tools = {t.name: t for t in await build_server(read_only=False).list_tools()}
    props = tools["search"].inputSchema["properties"]
    assert "question" in props and "top_k" in props


# ---------- http bearer auth (ASGI middleware) ----------


async def _drive(mw, headers, path=None):
    scope = {"type": "http", "headers": headers}
    if path is not None:
        scope["path"] = path
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    await mw(scope, receive, send)
    return sent


async def test_bearer_auth_rejects_missing_or_wrong_token():
    inner_ran = []

    async def inner(scope, receive, send):
        inner_ran.append(True)

    mw = _BearerAuth(inner, "sekret")

    sent = await _drive(mw, [])                                  # no header
    assert sent[0]["status"] == 401 and not inner_ran
    sent = await _drive(mw, [(b"authorization", b"Bearer wrong")])
    assert sent[0]["status"] == 401 and not inner_ran


async def test_bearer_auth_passes_correct_token():
    inner_ran = []

    async def inner(scope, receive, send):
        inner_ran.append(True)

    mw = _BearerAuth(inner, "sekret")
    await _drive(mw, [(b"authorization", b"Bearer sekret")])
    assert inner_ran == [True]


async def test_health_endpoint_is_unauthenticated():
    """The container/K8s liveness probe hits /health with no bearer token."""
    inner_ran = []

    async def inner(scope, receive, send):
        inner_ran.append(True)

    mw = _BearerAuth(inner, "sekret")
    sent = await _drive(mw, [], path="/health")  # no Authorization header
    assert sent[0]["status"] == 200
    assert sent[1]["body"] == b"ok"
    assert not inner_ran  # answered by the middleware, never reached the MCP app


async def test_non_http_scope_passes_through():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["type"])

    await _BearerAuth(inner, "t")({"type": "lifespan"}, None, None)
    assert seen == ["lifespan"]


# ---------- serve() transport gating ----------


def _cfg(monkeypatch, *, token="", read_only=False, host="127.0.0.1", port=8000):
    import ingestlib.mcp.server as server
    monkeypatch.setattr(server, "get_mcp_config",
                        lambda: MCPConfig(read_only=read_only, host=host, port=port, token=token))


def test_http_without_token_refuses(monkeypatch):
    _cfg(monkeypatch, token="")
    with pytest.raises(ValueError, match="requires a bearer token"):
        serve(transport="http")


def test_unknown_transport_refuses(monkeypatch):
    _cfg(monkeypatch, token="x")
    with pytest.raises(ValueError, match="unknown transport"):
        serve(transport="carrier-pigeon")


def test_stdio_starts_the_server(monkeypatch):
    """stdio needs no token; serve() should build and run() — stub run to
    avoid blocking on stdin."""
    _cfg(monkeypatch, token="")
    import ingestlib.mcp.server as server
    ran = {}
    monkeypatch.setattr(server.FastMCP, "run",
                        lambda self, transport="stdio", **kw: ran.setdefault("t", transport))
    serve(transport="stdio")
    assert ran["t"] == "stdio"


# ---------- CLI entrypoint ----------


def test_cli_mcp_dispatches_to_serve(monkeypatch):
    import ingestlib.mcp as mcp
    from ingestlib.cli import main

    captured = {}
    monkeypatch.setattr(mcp, "serve", lambda **kw: captured.update(kw))
    assert main(["mcp", "--transport", "http", "--port", "9001", "--read-only"]) == 0
    assert captured == {"transport": "http", "host": None, "port": 9001, "read_only": True}


def test_cli_mcp_defaults(monkeypatch):
    import ingestlib.mcp as mcp
    from ingestlib.cli import main

    captured = {}
    monkeypatch.setattr(mcp, "serve", lambda **kw: captured.update(kw))
    main(["mcp"])
    assert captured["transport"] == "stdio" and captured["read_only"] is None
