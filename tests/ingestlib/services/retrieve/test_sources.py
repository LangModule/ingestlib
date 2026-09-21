"""aretrieve(sources=[...]) — the fan-out that merges normalized SourceResults
into one envelope. Sources are stubbed at the registry seam; what's under test is
the merge, the top_k threading, and that the sources path leaves `hits` empty."""
import pytest

from ingestlib.services.retrieve import aretrieve
from ingestlib.sources.base import Source, SourceResult


class _FakeSource(Source):
    def __init__(self, name, results):
        self.name = name
        self._results = results
        self.seen = []

    async def answer(self, question, *, top_k=5):
        self.seen.append((question, top_k))
        return self._results

    async def health(self):
        return "ok", "fake"


def _sr(source, content, stype="structured"):
    return SourceResult(content=content, source=source, source_type=stype)


async def test_fan_out_merges_results_in_source_order(monkeypatch):
    from ingestlib.sources import registry

    a = _FakeSource("db", [_sr("db", "row1"), _sr("db", "row2")])
    b = _FakeSource("corpus", [_sr("corpus", "chunk", "documents")])
    monkeypatch.setattr(registry, "resolve_sources", lambda names: [a, b])

    result = await aretrieve("what is ready?", sources=["db", "corpus"], top_k=3)

    assert [r.source for r in result.results] == ["db", "db", "corpus"]
    assert result.hits == []                          # sources path never fills hits
    assert a.seen == [("what is ready?", 3)]          # top_k threaded through
    assert result.context.startswith("[1] (db) row1")


async def test_sources_none_uses_the_document_path(monkeypatch):
    from ingestlib.services.retrieve import retriever

    async def fake_hits(question, **kw):
        return []

    monkeypatch.setattr(retriever, "_retrieve_document_hits", fake_hits)
    result = await aretrieve("q")
    assert result.results == [] and result.hits == []


async def test_empty_question_rejected_even_with_sources():
    with pytest.raises(ValueError, match="non-empty"):
        await aretrieve("   ", sources=["db"])
