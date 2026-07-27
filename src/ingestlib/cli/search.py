"""`ingestlib search "question"` — cited retrieval from the shell.

Closes the loop the CLI leaves open: after `ingest`/`sync` succeed, the
natural next question is "can I find things?" — and this answers it without
a Python session. A thin wrapper over retrieve().
"""
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)


def run_search(
    question: str, *, top_k: int = 5, namespace: str = "", rerank: bool = True
) -> int:
    import os

    from ingestlib.utils.logger import configure

    if not os.environ.get("INGESTLIB_LOG_LEVEL"):
        configure(level="WARNING")

    from ingestlib.services import retrieve

    result = retrieve(question, top_k=top_k, namespace=namespace, rerank=rerank)
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
