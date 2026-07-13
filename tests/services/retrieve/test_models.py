"""Hit / RetrievalResult behavior — pure, always run."""
from ingestlib.services.retrieve.models import Hit, RetrievalResult
from ingestlib.storage.base import RetrievedChunk


def _hit(i: int = 1, markdown: str = "content") -> Hit:
    chunk = RetrievedChunk(
        score=0.8, document_id="7b6b95d79149c162" + "0" * 48, chunk_id=i,
        section="methods", markdown=markdown, pages=[4, 5],
    )
    return Hit(chunk=chunk, vector_score=0.8, rerank_score=0.9)


def test_citation_format():
    assert _hit().citation == "doc 7b6b95d79149 · p.4,5 · methods"


def test_context_is_numbered_and_cited():
    r = RetrievalResult(question="q", hits=[_hit(1, "first"), _hit(2, "second")])
    ctx = r.context
    assert ctx.index("[1]") < ctx.index("first") < ctx.index("[2]") < ctx.index("second")
    assert "doc 7b6b95d79149" in ctx


def test_empty_result_has_empty_context():
    assert RetrievalResult(question="q").context == ""
