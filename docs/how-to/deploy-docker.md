# Deploy with Docker

Bring up the whole self-hosted stack on one machine with a single command: the
ingestlib app (its MCP server), the registry Postgres, a vector store, and an
S3-compatible artifact store — pre-wired on one network.

The same code runs whether you `pip install ingestlib` into a single process,
bring up this Docker stack, or scale it out on Kubernetes — only the packaging
changes. This guide covers the Docker stack.

## What comes up

`docker compose -f infra/docker-compose.stack.yml --profile stack up` starts:

| Service | Role | Port |
|---|---|---|
| `app` | the ingestlib **MCP server** (agents connect here) | 8000 |
| `registry` | ingestlib's metadata database (Postgres) | 5433 |
| `qdrant` | the vector store | 6333 |
| `minio` | the artifact store (document bytes), S3-compatible | 9000 · console 9001 |

The LLM and embeddings are a **cloud API by key** (OpenAI by default), so nothing
heavy runs locally except — optionally — the OCR model (below).

## 1. Prerequisites

- Docker with Compose v2.
- `OPENAI_API_KEY` (chat + embeddings), `JINA_API_KEY` (reranking), and
  `MCP_TOKEN` (any strong random string the MCP server requires for its bearer
  auth).
- For **parse only**: an NVIDIA GPU (see [OCR](#ocr-the-one-gpu-piece)).

The app image is **linux/amd64** (the layout model, paddlepaddle, ships no arm64
wheel); on an Apple Silicon host Docker builds and runs it under emulation.

## 2. Start it

```bash
export OPENAI_API_KEY=sk-...
export JINA_API_KEY=jina_...
export MCP_TOKEN=$(openssl rand -hex 32)

docker compose -f infra/docker-compose.stack.yml --profile stack up -d
```

The app entrypoint runs `ingestlib registry init` (the schema migrations) on
first boot, then serves the MCP server. Check health:

```bash
curl -fsS http://localhost:8000/health        # → ok
```

Connect any MCP client to `http://localhost:8000` with the header
`Authorization: Bearer $MCP_TOKEN`. Or run the CLI inside the container:

```bash
docker exec -it ingestlib_app ingestlib doctor
docker exec -it ingestlib_app ingestlib ingest /path/in/container
```

Browse stored artifacts at the MinIO console: `http://localhost:9001`
(`minioadmin` / `minioadmin`).

## OCR — the one GPU piece

Parse runs PaddleOCR-VL, a vision model that **needs a GPU**. On an NVIDIA host
with the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit),
add the `gpu` profile to also run the OCR server:

```bash
docker compose -f infra/docker-compose.stack.yml --profile stack --profile gpu up -d
```

The first run downloads ~1.8 GB of weights into a named volume (once). The app
reaches it at `http://ocr:8111` — already wired in `infra/deploy/config.yaml`.

**No NVIDIA GPU?** The honest matrix:

| Host | parse | everything else |
|---|---|---|
| Linux + NVIDIA GPU | ✅ in-container | ✅ |
| macOS (Apple Silicon) | run OCR host-side (`mlx-vlm`), point `paddle_vl.server_url` at it | ✅ |
| CPU-only | ❌ unavailable | ✅ classify · split · extract · retrieve · SQL · MCP |

`ingestlib doctor` reports whether the OCR server is reachable; the rest of the
pipeline works regardless.

## Configure it

Providers, stores, and rules live in `infra/deploy/config.yaml` (mounted at
`/config`). It defaults to OpenAI + qdrant + MinIO + Jina reranking. To change a
provider or store, edit that file — add any new key to the app's environment in
the compose — and recreate the `app` service.

## Publishing / pulling the image

Release tags build and push the app image to GitHub Container Registry
(`.github/workflows/docker-image.yml`). To deploy a published image instead of
building locally, replace the `app` service's `build:` with
`image: ghcr.io/<owner>/ingestlib-app:<version>`.

## Tear down

```bash
docker compose -f infra/docker-compose.stack.yml --profile stack --profile gpu down
# add -v to also wipe the data volumes (registry, qdrant, minio, weights)
```
