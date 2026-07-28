"""FastMCP server wiring — register the tools, run a transport.

stdio for local clients (Claude Desktop / Code / Cursor); streamable-http for
remote agents (the 2026 production transport — plain SSE is legacy, not used).
HTTP is guarded: it binds 127.0.0.1 by default and REQUIRES a bearer token
(MCP_TOKEN), refusing to start without one. `read_only` hides the
corpus-modifying tools. This module imports the `mcp` SDK — reached only
through ingestlib.mcp.serve(), which turns a missing SDK into the pip hint.
"""
from mcp.server.fastmcp import FastMCP

from ingestlib.config import get_mcp_config
from ingestlib.mcp.tools import ALL_TOOLS, WRITE_TOOLS
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)


def build_server(read_only: bool) -> FastMCP:
    """A FastMCP server with the tools registered (write tools omitted when
    read_only). Pure — no transport started — so tests can introspect it."""
    server = FastMCP("ingestlib")
    exposed = [t for t in ALL_TOOLS if not (read_only and t.__name__ in WRITE_TOOLS)]
    for tool in exposed:
        server.add_tool(tool)  # name + description inferred from the function
    logger.info(
        "mcp: %d tool(s) registered%s",
        len(exposed), " (read-only)" if read_only else "",
    )
    return server


class _BearerAuth:
    """Minimal ASGI middleware: every HTTP request must carry
    `Authorization: Bearer <token>` or gets a 401. Pure ASGI — no framework
    dependency beyond what the http transport already pulls in."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        if headers.get(b"authorization") != self._expected:
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.app(scope, receive, send)


def serve(*, transport: str = "stdio", host: str | None = None,
          port: int | None = None, read_only: bool | None = None) -> None:
    """Start the server. CLI/config supply the defaults; explicit args win."""
    cfg = get_mcp_config()
    read_only = cfg.read_only if read_only is None else read_only
    host = host or cfg.host
    port = port or cfg.port

    if transport == "stdio":
        logger.info("mcp: serving over stdio%s", " (read-only)" if read_only else "")
        build_server(read_only).run(transport="stdio")
        return

    if transport in ("http", "streamable-http"):
        if not cfg.token:
            raise ValueError(
                "the http transport requires a bearer token — set MCP_TOKEN in "
                ".env (any strong random string), or use --transport stdio for "
                "a local client"
            )
        if host in ("0.0.0.0", "::"):
            logger.warning(
                "mcp: binding %s exposes the server on the network — ensure "
                "MCP_TOKEN is strong and prefer a reverse proxy with TLS", host,
            )
        server = build_server(read_only)
        server.settings.host = host
        server.settings.port = port
        app = _BearerAuth(server.streamable_http_app(), cfg.token)
        import uvicorn

        logger.info("mcp: serving streamable-http on %s:%d%s (bearer auth on)",
                    host, port, " (read-only)" if read_only else "")
        uvicorn.run(app, host=host, port=port, log_level="warning")
        return

    raise ValueError(
        f"unknown transport {transport!r} — choose 'stdio' or 'http'"
    )
