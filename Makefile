.PHONY: test test-all test-llm test-nova test-embedding test-rerank test-rerank-aws test-rerank-jina test-ocr test-parse test-classify test-split test-s3 test-pinecone test-qdrant test-sqlite test-pgvector test-mongodb test-milvus test-services eval

# fast suite — every opt-in e2e group skips (RUN_* gates unset)
test:
	uv run pytest tests/

# entire suite including every opt-in group (needs VL server running + Bedrock access)
test-all:
	RUN_AWS_RERANK=1 RUN_OCR_E2E=1 RUN_PARSE_E2E=1 RUN_CLASSIFY_E2E=1 RUN_SPLIT_E2E=1 RUN_S3_E2E=1 RUN_PINECONE_E2E=1 RUN_QDRANT_E2E=1 RUN_PGVECTOR_E2E=1 RUN_MONGODB_E2E=1 RUN_MILVUS_E2E=1 RUN_SERVICES_E2E=1 uv run pytest tests/

# --- llm layer (mirrors src/ingestlib/foundations/llm/) ---

test-llm:
	uv run pytest tests/foundations/llm/

test-nova:
	uv run pytest tests/foundations/llm/bedrock/nova/

test-embedding:
	uv run pytest tests/foundations/llm/bedrock/embedding/

# both providers — AWS-hitting tests skip without RUN_AWS_RERANK=1
test-rerank:
	uv run pytest tests/foundations/llm/bedrock/rerank/ tests/foundations/llm/jina/

# opt-in: hits amazon.rerank-v1:0 (2 RPM quota — expect ~1 min of throttle sleeps)
test-rerank-aws:
	RUN_AWS_RERANK=1 uv run pytest tests/foundations/llm/bedrock/rerank/

test-rerank-jina:
	uv run pytest tests/foundations/llm/jina/

# --- ocr layer (mirrors src/ingestlib/foundations/ocr/) — needs the VL inference server ---

test-ocr:
	RUN_OCR_E2E=1 uv run pytest tests/foundations/ocr/

# --- parse operation (mirrors src/ingestlib/operations/parse/) — needs VL server + Bedrock ---

test-parse:
	RUN_PARSE_E2E=1 uv run pytest tests/operations/parse/

# --- classify operation (mirrors src/ingestlib/operations/classify/) — needs Bedrock ---

test-classify:
	RUN_CLASSIFY_E2E=1 uv run pytest tests/operations/classify/

# --- split operation (mirrors src/ingestlib/operations/split/) — needs Bedrock ---

test-split:
	RUN_SPLIT_E2E=1 uv run pytest tests/operations/split/

# --- storage (S3 artifacts) — needs AWS credentials ---

test-s3:
	RUN_S3_E2E=1 uv run pytest tests/storage/

# --- pinecone connector — needs PINECONE_API_KEY + Bedrock (embeddings) ---

test-pinecone:
	RUN_PINECONE_E2E=1 uv run pytest tests/storage/pinecone/

# --- qdrant connector — needs a Qdrant server at QDRANT_URL + Bedrock ---

test-qdrant:
	RUN_QDRANT_E2E=1 uv run pytest tests/storage/qdrant/

# --- sqlite connector — no gate: no server exists, in-process IS the real thing ---

test-sqlite:
	uv run pytest tests/storage/sqlite/

# --- pgvector connector — needs a Postgres at PGVECTOR_URL (no Bedrock) ---

test-pgvector:
	RUN_PGVECTOR_E2E=1 uv run pytest tests/storage/pgvector/

# --- mongodb connector — needs a MongoDB at MONGODB_URL with search (no Bedrock) ---

test-mongodb:
	RUN_MONGODB_E2E=1 uv run pytest tests/storage/mongodb/

# --- milvus connector — needs a Milvus at MILVUS_URL (no Bedrock) ---

test-milvus:
	RUN_MILVUS_E2E=1 uv run pytest tests/storage/milvus/

# --- services (ingest + retrieve) — needs the FULL stack ---

test-services:
	RUN_SERVICES_E2E=1 uv run pytest tests/services/

# --- retrieval quality eval — measurement, not a test (needs the full stack;
#     first run also needs the VL server to ingest the fixture corpus) ---

eval:
	uv run python evals/run_eval.py
