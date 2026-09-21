"""ClassifyResult / CategoryScore behavior — pure, always run."""
import pytest

from ingestlib.operations.classify.models import CategoryScore, ClassifyResult


def test_result_is_frozen():
    r = ClassifyResult(category="invoice", confidence=0.9)
    with pytest.raises(Exception):
        r.category = "other"  # type: ignore[misc]


def test_confidence_bounds_enforced():
    with pytest.raises(Exception):
        ClassifyResult(category="invoice", confidence=1.5)
    with pytest.raises(Exception):
        CategoryScore(label="a", score=-0.1)


def test_defaults():
    r = ClassifyResult(category="report", confidence=0.5)
    assert r.alternatives == [] and r.reasoning == "" and r.pages_used == 0
