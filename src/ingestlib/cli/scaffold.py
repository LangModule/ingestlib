"""`ingestlib init` — write the setup files into the current directory.

Two presets:
    init          — the default stack (Bedrock + sqlite + Jina): config.yaml
                    with an aws section to fill in, .env with the Jina key slot
    init --local  — the zero-cloud stack (Ollama + sqlite + local artifacts +
                    no reranker): a five-choice config.yaml, no .env needed

Files land in the current directory because that is where config discovery
looks (CWD and its parents). Existing files are never overwritten without
--force.
"""
from pathlib import Path

_DEFAULT_CONFIG = """\
# ingestlib configuration — created by `ingestlib init`.
# Only what you choose lives here; every omitted key is a library default
# (see config.example.yaml in the repo for the full annotated reference).

# Your AWS profile from ~/.aws/credentials, with Bedrock access.
aws:
  profile: your-aws-profile
  region: us-east-1
  account_id: "000000000000"

# sqlite (default: one local file, no server, no keys) | pinecone | qdrant
# | pgvector | mongodb | milvus | opensearch | weaviate — secrets in .env
vector_store: sqlite

# jina (needs JINA_API_KEY) | aws (same AWS credentials) | none
reranker: jina

# Who serves LLM and embedding calls:
#   bedrock (default) | openai (needs OPENAI_API_KEY) | ollama (local, no key)
llm_provider: bedrock
embedding_provider: bedrock

# s3 (durable, shareable) | local (a plain folder — zero cloud)
artifact_store: s3

# OCR server for parse/ingest: mlx-vlm-server (Apple Silicon) | vllm-server (NVIDIA)
paddle_vl:
  backend: mlx-vlm-server
  server_url: http://localhost:8111/
"""

_DEFAULT_ENV = """\
# ingestlib secrets — fill only what your config.yaml choices need.

# reranker: jina — free key at https://jina.ai/api-dashboard
JINA_API_KEY=

# llm_provider / embedding_provider: openai
OPENAI_API_KEY=

# Server-backed vector stores each need their pip extra (e.g.
# pip install "ingestlib[qdrant]") plus one URL/key here when selected —
# e.g. PINECONE_API_KEY, QDRANT_URL, PGVECTOR_URL, MONGODB_URL, MILVUS_URL,
# OPENSEARCH_URL, WEAVIATE_URL. sqlite (the default) needs none.
"""

_LOCAL_CONFIG = """\
# ingestlib configuration — created by `ingestlib init --local`.
# The zero-cloud stack: local LLM, local vectors, local artifacts. No keys.

llm_provider: ollama          # qwen3.5:9b via a local Ollama server
embedding_provider: ollama    # qwen3-embedding:0.6b (1024-dim)
vector_store: sqlite          # one local file, created on first use
artifact_store: local         # parse/classify/split outputs as plain files
reranker: none                # vector order as-is; jina/aws need accounts

# OCR server for parse/ingest: mlx-vlm-server (Apple Silicon) | vllm-server (NVIDIA)
paddle_vl:
  backend: mlx-vlm-server
  server_url: http://localhost:8111/
"""

_NEXT_STEPS_DEFAULT = """\
Next steps:
  1. Edit config.yaml — set your aws profile (needs Bedrock access), or
     switch llm_provider/embedding_provider to openai or ollama
  2. Edit .env — fill the keys your choices need (Jina for the reranker)
  3. Start the OCR server (needed for parse/ingest only):
       python -m mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6
  4. Verify the stack:
       ingestlib doctor
"""

_NEXT_STEPS_LOCAL = """\
Next steps:
  1. Install Ollama (https://ollama.com) and pull the models:
       ollama pull qwen3.5:9b
       ollama pull qwen3-embedding:0.6b
  2. Start the OCR server (needed for parse/ingest only):
       python -m mlx_vlm.server --port 8111 --model PaddlePaddle/PaddleOCR-VL-1.6
  3. Verify the stack:
       ingestlib doctor

No API keys and no .env needed — nothing leaves your machine.
"""


def run_init(local: bool = False, force: bool = False) -> int:
    """Write the preset's files into CWD. Returns a process exit code."""
    files = {"config.yaml": _LOCAL_CONFIG if local else _DEFAULT_CONFIG}
    if not local:
        files[".env"] = _DEFAULT_ENV

    existing = [name for name in files if Path(name).exists()]
    if existing and not force:
        print(f"refusing to overwrite existing {', '.join(existing)} — "
              f"rerun with --force to replace")
        return 1

    for name, content in files.items():
        Path(name).write_text(content)
        print(f"wrote {name}")

    print()
    print(_NEXT_STEPS_LOCAL if local else _NEXT_STEPS_DEFAULT, end="")
    return 0
