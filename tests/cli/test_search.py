"""`ingestlib search` driven through main() with retrieve() stubbed at the
CLI seam — the command formats hits, honors --top-k / --no-rerank, and
handles the empty result cleanly."""
import pytest

from ingestlib.cli import main


def _fake_result(question, n=2, rerank=True):
    from ingestlib.services.retrieve.models import Hit, RetrievalResult
    from ingestlib.storage.base import RetrievedChunk

    hits = [
        Hit(
            chunk=RetrievedChunk(
                score=0.9, document_id="a" * 12 + "b" * 52, chunk_id=i,
                section="methods", heading="Participant recruitment",
                text="Participants were recruited through community centers.",
                pages=[4],
            ),
            vector_score=0.8 - i * 0.1,
            rerank_score=(0.95 - i * 0.1) if rerank else None,
        )
        for i in range(n)
    ]
    return RetrievalResult(question=question, hits=hits)


@pytest.fixture()
def stub_retrieve(monkeypatch):
    captured = {}

    def fake_retrieve(question, *, top_k=10, namespace="", rerank=True, **kw):
        captured.update(top_k=top_k, namespace=namespace, rerank=rerank)
        n = 0 if question == "nothing" else min(top_k, 2)
        return _fake_result(question, n=n, rerank=rerank)

    # run_search does `from ingestlib.services import retrieve` at call time
    import ingestlib.services as services
    monkeypatch.setattr(services, "retrieve", fake_retrieve)
    return captured


def test_search_prints_cited_hits(stub_retrieve, capsys):
    assert main(["search", "how were participants recruited?"]) == 0
    out = capsys.readouterr().out
    assert "[1]" in out
    assert "Participant recruitment" in out
    assert "· p.4 · methods" in out


def test_top_k_and_namespace_flow_through(stub_retrieve, capsys):
    assert main(["search", "q", "--top-k", "3", "--namespace", "tenant-a"]) == 0
    assert stub_retrieve["top_k"] == 3
    assert stub_retrieve["namespace"] == "tenant-a"
    assert stub_retrieve["rerank"] is True


def test_no_rerank_flag(stub_retrieve, capsys):
    assert main(["search", "q", "--no-rerank"]) == 0
    assert stub_retrieve["rerank"] is False


def test_empty_hits_is_not_an_error(stub_retrieve, capsys):
    assert main(["search", "nothing"]) == 0
    assert "no hits" in capsys.readouterr().out


def test_runtime_failure_is_a_clean_line_not_a_traceback(monkeypatch, capsys):
    """A dead provider/store must degrade to one '✗ <fix>' line, not a stack
    trace — the top-level handler in main(), shared by every command."""
    import ingestlib.services as services

    def boom(question, **kw):
        raise RuntimeError("cannot reach the Ollama server — start Ollama")

    monkeypatch.setattr(services, "retrieve", boom)
    monkeypatch.delenv("INGESTLIB_LOG_LEVEL", raising=False)

    assert main(["search", "q"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("✗ ")
    assert "start Ollama" in out
    assert "Traceback" not in out


def test_debug_env_reraises_for_a_real_traceback(monkeypatch):
    """INGESTLIB_LOG_LEVEL=DEBUG is the escape hatch — the exception propagates
    so a developer sees the full stack."""
    import ingestlib.services as services

    def boom(question, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(services, "retrieve", boom)
    monkeypatch.setenv("INGESTLIB_LOG_LEVEL", "DEBUG")

    with pytest.raises(RuntimeError, match="boom"):
        main(["search", "q"])


# --- --sources: normalized results from documents and/or SQL databases ---

def test_search_sources_prints_normalized_rows(monkeypatch, capsys):
    from ingestlib.services.retrieve.models import RetrievalResult
    from ingestlib.sources.base import SourceResult

    captured = {}

    def fake_retrieve(question, *, top_k=5, namespace="", rerank=True, sources=None, **kw):
        captured["sources"] = sources
        return RetrievalResult(question=question, results=[
            SourceResult(content="rx_id | status\n1 | ready", source="rx",
                         source_type="structured"),
        ])

    import ingestlib.services as services
    monkeypatch.setattr(services, "retrieve", fake_retrieve)

    assert main(["search", "ready?", "--sources", "rx, corpus"]) == 0
    out = capsys.readouterr().out
    assert captured["sources"] == ["rx", "corpus"]        # comma-split, trimmed
    assert "[1] rx (structured)" in out
    assert "rx_id | status" in out


def test_search_sources_empty_results_is_a_clean_message(monkeypatch, capsys):
    from ingestlib.services.retrieve.models import RetrievalResult

    def fake_retrieve(question, **kw):
        return RetrievalResult(question=question, results=[])

    import ingestlib.services as services
    monkeypatch.setattr(services, "retrieve", fake_retrieve)

    assert main(["search", "q", "--sources", "rx"]) == 0
    assert "no results from the given source" in capsys.readouterr().out
