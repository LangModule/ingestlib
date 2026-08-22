"""`ingestlib search "question"` — cited retrieval from the shell.

Closes the loop the CLI leaves open: after `ingest`/`sync` succeed, the
natural next question is "can I find things?" — and this answers it without
a Python session. A thin wrapper over retrieve(). With `--sources` it queries
declared sources (documents and/or SQL databases from sources.yaml) instead.
"""
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)


def run_search(
    question: str,
    *,
    top_k: int = 5,
    namespace: str = "",
    rerank: bool = True,
    sources: list[str] | None = None,
) -> int:
    import os

    from ingestlib.utils.logger import configure

    if not os.environ.get("INGESTLIB_LOG_LEVEL"):
        configure(level="WARNING")

    from ingestlib.services import retrieve

    result = retrieve(
        question, top_k=top_k, namespace=namespace, rerank=rerank, sources=sources
    )

    # sources= path: normalized results (documents and/or SQL rows)
    if sources:
        if not result.results:
            print("no results from the given source(s)")
            return 0
        for i, r in enumerate(result.results, start=1):
            print(f"[{i}] {r.source} ({r.source_type})")
            for line in r.content.splitlines()[:6]:
                print(f"     {line[:160]}")
        return 0

    # default path: document hits
    if not result.hits:
        print("no hits — is anything ingested into this store/namespace? "
              "(try `ingestlib list`)")
        return 0

    for i, hit in enumerate(result.hits, start=1):
        score = hit.rerank_score if hit.rerank_score is not None else hit.vector_score
        heading = hit.chunk.heading or hit.chunk.section or ""
        print(f"[{i}] {score:.3f}  {hit.citation}")
        if heading:
            print(f"     {heading}")
        snippet = (hit.chunk.text or hit.chunk.markdown).strip().replace("\n", " ")
        print(f"     {snippet[:140]}")
    return 0
