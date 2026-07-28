"""MCP server — expose ingestlib to agents (Claude Desktop, Cursor, …).

    ingestlib mcp                          # stdio (local agent clients)
    ingestlib mcp --transport http --port 8000   # remote; needs MCP_TOKEN

An interface layer over the existing services (like the CLI): the tools in
`tools.py` wrap ingest/search/extract/sync/remove/backfill/…; `server.py`
registers them with FastMCP and runs the transport. Loaded lazily so the
core library never imports the `mcp` SDK unless the server is started.
"""


def serve(*, transport: str = "stdio", host: str | None = None,
          port: int | None = None, read_only: bool | None = None) -> None:
    """Start the ingestlib MCP server. Lazy entry point — imports the SDK here
    so a missing `mcp` extra raises the pip-install hint, not on package import."""
    try:
        from ingestlib.mcp.server import serve as _serve
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("ingestlib"):
            raise
        raise ImportError(
            'the mcp server needs its SDK (module '
            f'{exc.name!r} is not installed) — install it with: '
            'pip install "ingestlib[mcp]"'
        ) from exc

    _serve(transport=transport, host=host, port=port, read_only=read_only)


__all__ = ["serve"]
