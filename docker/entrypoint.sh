#!/usr/bin/env sh
# Run a given command, else init the registry (idempotent) and serve the MCP
# server. http transport needs MCP_TOKEN; 0.0.0.0 to be reachable off-container.
set -e

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

if [ -n "${INGESTLIB_REGISTRY_URL:-}" ]; then
    echo "ingestlib: initialising registry schema…"
    ingestlib registry init || echo "ingestlib: registry init failed — continuing"
fi

exec ingestlib mcp --transport http --host 0.0.0.0 --port 8000
