"""run_sync itself, plus every sync wrapper naming its async form in a loop."""
import pytest

from ingestlib.utils.sync import run_sync


def test_run_sync_returns_the_coroutine_value_outside_a_loop():
    async def compute():
        return 41 + 1

    assert run_sync(compute(), "acompute") == 42


def test_run_sync_passes_other_runtime_errors_through_unchanged():
    async def explode():
        raise RuntimeError("backend fell over")

    with pytest.raises(RuntimeError, match="backend fell over"):
        run_sync(explode(), "aexplode")


async def test_parse_inside_a_loop_names_aparse():
    from ingestlib.operations import parse

    with pytest.raises(RuntimeError, match="await aparse"):
        parse("whatever.pdf")


async def test_classify_inside_a_loop_names_aclassify():
    from ingestlib.operations import classify

    with pytest.raises(RuntimeError, match="await aclassify"):
        classify("whatever.pdf")


async def test_split_inside_a_loop_names_asplit():
    from ingestlib.operations import split

    with pytest.raises(RuntimeError, match="await asplit"):
        split("whatever.pdf")


async def test_ingest_inside_a_loop_names_aingest():
    from ingestlib.services import ingest

    with pytest.raises(RuntimeError, match="await aingest"):
        ingest("whatever.pdf")


async def test_retrieve_inside_a_loop_names_aretrieve():
    from ingestlib.services import retrieve

    with pytest.raises(RuntimeError, match="await aretrieve"):
        retrieve("a question")
