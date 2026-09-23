# syntax=docker/dockerfile:1
# check=skip=FromPlatformFlagConstDisallowed
# ingestlib runtime — the library + every connector, with the MCP server as the
# default front-end. amd64 only: paddlepaddle (the layout model) has no arm64 wheel.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    INGESTLIB_CONFIG=/config/config.yaml

# libreoffice: DOCX/PPTX→PDF · postgresql-client: pg_dump for registry backup ·
# libgl1/libglib2.0-0/libgomp1: the vision deps' shared libraries · curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-core libreoffice-writer libreoffice-impress \
        postgresql-client \
        curl \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Dependencies first (this layer caches unless pyproject/uv.lock change): export
# the exact locked versions — mlx-vlm's darwin marker skips it on linux.
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --extra all --no-emit-project --no-dev -o /tmp/req.txt \
    && uv pip install --system --no-cache -r /tmp/req.txt \
    && rm /tmp/req.txt
# Then the project — this layer re-runs on a code change, deps stay cached.
COPY src ./src
RUN uv pip install --system --no-cache --no-deps .

COPY docker/entrypoint.sh /usr/local/bin/ingestlib-entrypoint
RUN chmod +x /usr/local/bin/ingestlib-entrypoint

# Drop root: run as an unprivileged user. --create-home gives a writable home
# for any runtime cache/model downloads (e.g. the layout model during parse).
RUN useradd --create-home --user-group --uid 10001 app && chown -R app:app /app
ENV HOME=/home/app
USER app

# mount config.yaml (+ .env) at /config
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["ingestlib-entrypoint"]
