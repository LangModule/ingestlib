"""SplitResult / Chunk behavior — pure, always run."""
import pytest

from ingestlib.operations.split.models import Chunk, Section, SplitResult


def _chunk(cid, section="s"):
    return Chunk(chunk_id=cid, section=section, text="t", markdown="m",
                 embedding_text="[s]\n\nm", pages=[1])


def test_chunks_flatten_in_document_order():
    r = SplitResult(sections=[
        Section(name="a", pages=[1], chunks=[_chunk(0, "a"), _chunk(1, "a")]),
        Section(name="b", pages=[2], chunks=[_chunk(2, "b")]),
    ])
    assert [c.chunk_id for c in r.chunks] == [0, 1, 2]
    assert r.section_names == ["a", "b"]


def test_section_by_name_and_missing():
    r = SplitResult(sections=[Section(name="a", pages=[1])])
    assert r.section_by_name("a").pages == [1]
    with pytest.raises(KeyError):
        r.section_by_name("zzz")


def test_models_frozen():
    c = _chunk(0)
    with pytest.raises(Exception):
        c.heading = "nope"  # type: ignore[misc]
