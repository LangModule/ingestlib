"""DocumentSource — the corpus as a Source, mapping each Hit into a SourceResult
so documents compose in the same retrieve(sources=[...]) fan-out as databases.
The document retrieval it wraps is stubbed at its seam."""
from ingestlib.sources.documents import DocumentSource


def _hit(markdown="the answer text"):
    from ingestlib.services.retrieve.models import Hit
    from ingestlib.storage.base import RetrievedChunk

    chunk = RetrievedChunk(
        score=0.7, document_id="d" * 64, chunk_id=3, section="methods",
        markdown=markdown, text=markdown, pages=[4, 5], region_ids={4: [2]},
    )
    return Hit(chunk=chunk, vector_score=0.7, rerank_score=0.91)


async def test_answer_maps_hits_to_source_results(monkeypatch):
    from ingestlib.services.retrieve import retriever

    async def fake_hits(question, **kw):
        assert kw["top_k"] == 3 and kw["namespace"] == "tenant-a"
        return [_hit("participants were recruited in Cairo")]

    monkeypatch.setattr(retriever, "_retrieve_document_hits", fake_hits)
    [r] = await DocumentSource("corpus", namespace="tenant-a").answer("who?", top_k=3)

    assert r.source == "corpus" and r.source_type == "documents"
    assert r.content.startswith("doc ")          # citation prefix
    assert "Cairo" in r.content
    assert r.provenance == {"document_id": "d" * 64, "pages": [4, 5], "region_ids": {4: [2]}}
    assert r.score == 0.91                        # rerank_score wins
    assert r.raw is not None


async def test_score_falls_back_to_vector_when_no_rerank(monkeypatch):
    from ingestlib.services.retrieve import retriever

    hit = _hit().model_copy(update={"rerank_score": None})

    async def fake_hits(question, **kw):
        return [hit]

    monkeypatch.setattr(retriever, "_retrieve_document_hits", fake_hits)
    [r] = await DocumentSource("corpus").answer("q")
    assert r.score == 0.7                          # vector_score fallback


async def test_health_ok_when_store_resolves(monkeypatch):
    import ingestlib.storage as storage

    monkeypatch.setattr(storage, "default_store", lambda: object())
    status, detail = await DocumentSource("corpus").health()
    assert status == "ok" and "documents" in detail


async def test_health_fails_when_store_unavailable(monkeypatch):
    import ingestlib.storage as storage

    def boom():
        raise RuntimeError("no vector store configured")

    monkeypatch.setattr(storage, "default_store", boom)
    status, detail = await DocumentSource("corpus").health()
    assert status == "fail" and "no vector store" in detail
