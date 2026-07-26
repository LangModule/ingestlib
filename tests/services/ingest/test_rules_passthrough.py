"""Content-rule arguments flow from ingest() into classify and split — pure.

Every network boundary is patched at the ingestor's module seams; the test
verifies plumbing, not models: the five rule arguments must reach
aclassify/asplit exactly as passed, and by default as None (preset
resolution stays the operations' job, not the ingestor's).
"""
import importlib
from pathlib import Path

import pytest

from ingestlib.operations.classify.models import ClassifyResult
from ingestlib.operations.parse.models import PageResult, ParseResult
from ingestlib.operations.split.models import Chunk, Section, SplitResult
from ingestlib.services import aingest
from ingestlib.storage.base import VectorStore

# services/__init__ re-exports `ingest` the function, shadowing the subpackage
# on dotted-path lookup — grab the module object explicitly.
ingestor = importlib.import_module("ingestlib.services.ingest.ingestor")

_RULES = {"invoice": "itemized charges"}
_VOCAB = {"methods": "how it was done"}


class _FakeStore(VectorStore):
    def upsert_chunks(self, document_id, chunks, embeddings, category="", namespace=""):
        return len(chunks)

    def query(self, vector, top_k=10, filters=None, namespace="", text=None):
        return []

    def delete_document(self, document_id, namespace=""):
        return 0


@pytest.fixture()
def piped(monkeypatch, tmp_path):
    """Patch every boundary; return (run, captured) where captured records the
    kwargs classify and split were called with."""
    captured: dict[str, dict] = {}

    async def fake_aparse(path, *, dpi=200):
        return ParseResult(
            pages=[PageResult(page_num=1, markdown="content")],
            source_path=Path(path), source_format="pdf",
        )

    async def fake_aclassify(source, categories=None, *, target_pages=None, max_pages=None):
        captured["classify"] = {
            "categories": categories, "target_pages": target_pages, "max_pages": max_pages,
        }
        return ClassifyResult(category="report", confidence=0.9)

    async def fake_asplit(source, *, category=None, max_chunk_tokens=768,
                          vocabulary=None, unmatched=None):
        captured["split"] = {"vocabulary": vocabulary, "unmatched": unmatched}
        chunk = Chunk(chunk_id=0, section="s", text="t", markdown="m",
                      embedding_text="[s]\n\nm", pages=[1])
        return SplitResult(sections=[Section(name="s", pages=[1], chunks=[chunk])])

    async def fake_embed(text, purpose="GENERIC_INDEX", dimension=1024):
        return [0.0, 0.1]

    monkeypatch.setattr(ingestor, "aparse", fake_aparse)
    monkeypatch.setattr(ingestor, "aclassify", fake_aclassify)
    monkeypatch.setattr(ingestor, "asplit", fake_asplit)
    monkeypatch.setattr(ingestor, "aembed_text", fake_embed)
    for name in ("save_parse", "save_classify", "save_split", "save_ingest_manifest"):
        monkeypatch.setattr(ingestor.artifacts, name, lambda *a, **k: None)

    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"content-hash-me")

    async def run(**kwargs):
        return await aingest(doc, store=_FakeStore(), skip_existing=False, **kwargs)

    return run, captured


async def test_rule_arguments_reach_classify_and_split(piped):
    run, captured = piped
    result = await run(
        categories=_RULES, target_pages="1,3", max_pages=2,
        vocabulary=_VOCAB, unmatched="skip",
    )
    assert captured["classify"] == {
        "categories": _RULES, "target_pages": "1,3", "max_pages": 2,
    }
    assert captured["split"] == {"vocabulary": _VOCAB, "unmatched": "skip"}
    assert result.status == "ingested" and result.vectors == 1


async def test_default_is_none_so_presets_stay_the_operations_job(piped):
    run, captured = piped
    await run()
    assert captured["classify"] == {
        "categories": None, "target_pages": None, "max_pages": None,
    }
    assert captured["split"] == {"vocabulary": None, "unmatched": None}
