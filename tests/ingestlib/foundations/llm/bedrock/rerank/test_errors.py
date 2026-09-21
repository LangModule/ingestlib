"""Client-side validation guards — no AWS calls, always run."""
import pytest

from ingestlib.foundations.llm import aws_rerank
from ingestlib.foundations.llm.bedrock.rerank import MAX_DOCUMENTS


def test_empty_docs_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        aws_rerank("query", [])


def test_over_limit_docs_raises_value_error():
    with pytest.raises(ValueError, match=f"at most {MAX_DOCUMENTS}"):
        aws_rerank("query", ["doc"] * (MAX_DOCUMENTS + 1))
