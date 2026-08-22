"""`ingestlib doctor` — verify the configured stack with real calls.

Checks only what config.yaml's choices require, in dependency order. Every
check makes the cheapest REAL call that proves the piece works (an LLM ping
costs a fraction of a cent; local checks are free) — the error-hint helpers
throughout the library turn failures into one-sentence fixes, and doctor is
where they all surface together.

Statuses: ok — works; warn — degraded but the core pipeline runs (e.g. the
OCR server is down: parse/ingest need it, classify/split/extract/retrieve
don't); fail — this choice is broken. Exit code 1 when anything fails.
"""
import shutil
import sys

Check = tuple[str, str]  # (status, detail) with status ok | warn | fail | skip

_MARKS = {"ok": "✓", "warn": "!", "fail": "✗", "skip": "-"}


def check_python() -> Check:
    v = sys.version_info
    if v >= (3, 12):
        return "ok", f"python {v.major}.{v.minor}.{v.micro}"
    return "fail", f"python {v.major}.{v.minor} — ingestlib needs 3.12+"


def check_libreoffice() -> Check:
    if shutil.which("soffice"):
        return "ok", "LibreOffice (soffice) found"
    return "warn", (
        "LibreOffice not found — DOCX/PPTX inputs need it (PDF and images "
        "work without it). Install: brew install --cask libreoffice (macOS) "
        "or sudo apt install libreoffice-core (Linux)"
    )


def check_ocr_server() -> Check:
    from ingestlib.config import get_paddle_vl_config
    from ingestlib.foundations.ocr.paddle_vl import _check_server

    cfg = get_paddle_vl_config()
    try:
        _check_server(cfg.server_url, cfg.backend)
    except Exception as exc:
        return "warn", (
            f"{exc}\n(parse/ingest need the OCR server; "
            f"classify, split, extract, and retrieve run without it)"
        )
    return "ok", f"OCR server answering at {cfg.server_url}"


def check_llm() -> Check:
    from ingestlib.config import get_config
    from ingestlib.foundations.llm import chat

    provider = get_config().llm_provider
    try:
        chat("Reply with exactly: OK")
    except Exception as exc:
        return "fail", f"llm_provider {provider}: {exc}"
    return "ok", f"llm_provider {provider}: chat call succeeded"


def check_embeddings() -> Check:
    from ingestlib.config import get_config
    from ingestlib.foundations.llm import embed_text

    provider = get_config().embedding_provider
    try:
        vector = embed_text("doctor check")
    except Exception as exc:
        return "fail", f"embedding_provider {provider}: {exc}"
    return "ok", f"embedding_provider {provider}: {len(vector)}-dim vectors"


def check_reranker() -> Check:
    from ingestlib.config import get_config

    name = get_config().reranker
    if name == "none":
        return "skip", "reranker none: retrieve() returns vector order"
    try:
        if name == "jina":
            from ingestlib.foundations.llm import jina_rerank as rerank
        else:
            from ingestlib.foundations.llm import aws_rerank as rerank
        rerank("doctor check", ["a tiny document"], top_n=1)
    except Exception as exc:
        return "fail", f"reranker {name}: {exc}"
    return "ok", f"reranker {name}: rerank call succeeded"


def check_artifact_store() -> Check:
    from ingestlib.config import get_config

    store = get_config().artifact_store
    if store == "local":
        from ingestlib.config import get_artifacts_config

        path = get_artifacts_config().path
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".doctor-write-check"
            probe.write_text("ok")
            probe.unlink()
        except OSError as exc:
            return "fail", f"artifacts folder {path} is not writable: {exc}"
        return "ok", f"artifacts in local folder {path}"

    from botocore.exceptions import ClientError

    from ingestlib.config import get_s3_config
    from ingestlib.storage.s3.client import get_s3_client, s3_error_hint

    bucket = get_s3_config().bucket
    try:
        get_s3_client().head_bucket(Bucket=bucket)
    except Exception as exc:
        code = exc.response["Error"]["Code"] if isinstance(exc, ClientError) else ""
        if code in ("404", "NoSuchBucket"):
            return "ok", f"S3 bucket {bucket!r} will be created on first use"
        return "fail", f"artifact_store s3: {s3_error_hint(exc, bucket) or exc}"
    return "ok", f"artifacts in S3 bucket {bucket!r}"


def check_vector_store() -> Check:
    from ingestlib.config import get_config

    name = get_config().vector_store
    try:
        _ping_store(name)
    except Exception as exc:
        return "fail", f"vector_store {name}: {exc}"
    return "ok", f"vector_store {name}: reachable"


def _ping_store(name: str) -> None:
    """Cheapest real liveness proof per connector — never creates indexes."""
    import ingestlib.storage as storage

    # Resolve through the lazy loader first: a missing SDK raises the
    # pip-extra instruction instead of a bare ModuleNotFoundError.
    if name in storage._BY_CONFIG_NAME:
        storage._load_store_class(storage._BY_CONFIG_NAME[name])

    if name == "sqlite":
        # Open + load the sqlite-vec extension (the real failure mode) WITHOUT
        # querying — a query would create the schema at the probe vector's
        # dimension and poison the first real ingest.
        from ingestlib.storage.sqlite.client import connection, schema_exists

        with connection() as conn:
            schema_exists(conn)
    elif name == "pinecone":
        from ingestlib.storage.pinecone.client import get_pinecone_client

        get_pinecone_client().list_indexes()
    elif name == "qdrant":
        from ingestlib.storage.qdrant.client import get_qdrant_client

        get_qdrant_client().get_collections()
    elif name == "pgvector":
        from ingestlib.storage.pgvector.client import connect

        with connect() as conn:
            conn.execute("SELECT 1")
    elif name == "mongodb":
        from ingestlib.storage.mongodb.client import get_collection

        get_collection().database.client.admin.command("ping")
    elif name == "milvus":
        from ingestlib.storage.milvus.client import get_milvus_client

        get_milvus_client().list_collections()
    elif name == "opensearch":
        from ingestlib.storage.opensearch.client import get_opensearch_client

        get_opensearch_client().info()
    elif name == "weaviate":
        from ingestlib.storage.weaviate.client import get_weaviate_client

        if not get_weaviate_client().is_ready():
            raise RuntimeError("server reports not ready")
    else:
        raise ValueError(f"unknown vector_store {name!r}")


def check_sources() -> Check:
    from ingestlib.config import get_sources_config

    specs = get_sources_config().sources
    if not specs:
        return "skip", "no sources.yaml — structured retrieval not configured"

    import asyncio

    from ingestlib.sources.registry import resolve_sources

    try:
        sources = resolve_sources(list(specs))
    except Exception as exc:
        return "fail", f"sources: {exc}"
    failures = []
    for src in sources:
        try:
            status, detail = asyncio.run(src.health())
        except Exception as exc:
            status, detail = "fail", str(exc)
        if status != "ok":
            failures.append(detail)
    if failures:
        return "fail", "; ".join(failures)
    return "ok", f"{len(specs)} structured/document source(s) reachable"


def _print(status: str, detail: str) -> None:
    first, *rest = detail.splitlines()
    print(f"  {_MARKS[status]} {first}")
    for line in rest:
        print(f"    {line}")


def run_doctor() -> int:
    """Run every applicable check, print each result, return the exit code."""
    import os

    from ingestlib.utils.logger import configure

    # The report lines ARE the output — quiet the library's INFO chatter
    # (an explicit INGESTLIB_LOG_LEVEL still wins).
    if not os.environ.get("INGESTLIB_LOG_LEVEL"):
        configure(level="WARNING")

    print("ingestlib doctor")

    status, detail = check_python()
    _print(status, detail)
    failed = status == "fail"

    # Config gates everything else — without it no other check can run.
    try:
        from ingestlib.config import _find_config_path, get_config

        cfg = get_config()
        _print("ok", (
            f"config {_find_config_path()} (llm={cfg.llm_provider} "
            f"embed={cfg.embedding_provider} store={cfg.vector_store} "
            f"artifacts={cfg.artifact_store} reranker={cfg.reranker})"
        ))
    except Exception as exc:
        _print("fail", f"config: {exc}")
        print("\n1 check failed — fix config.yaml first (ingestlib init writes one)")
        return 1

    for check in (
        check_libreoffice,
        check_ocr_server,
        check_llm,
        check_embeddings,
        check_reranker,
        check_artifact_store,
        check_vector_store,
        check_sources,
    ):
        status, detail = check()
        _print(status, detail)
        failed = failed or status == "fail"

    if failed:
        print("\nsome checks failed — fix the ✗ lines above and rerun")
        return 1
    print("\nall checks passed — the configured stack is ready")
    return 0
