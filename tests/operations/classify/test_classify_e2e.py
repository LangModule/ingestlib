"""Real classify against Bedrock Nova. Opt-in via RUN_CLASSIFY_E2E=1."""
import os
import re
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
while _TESTS_DIR.name != "tests":
    _TESTS_DIR = _TESTS_DIR.parent
_PDF = _TESTS_DIR / "data" / "pdf"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CLASSIFY_E2E") != "1",
    reason="classify e2e is opt-in: set RUN_CLASSIFY_E2E=1 (needs Bedrock access)",
)


def test_open_ended_standalone_produces_snake_case_label():
    from ingestlib.operations.classify import classify

    r = classify(f"{_PDF}/insurance-acord.pdf")
    assert re.fullmatch(r"[a-z][a-z0-9_]*", r.category), r.category
    assert "insur" in r.category or "certificate" in r.category
    assert 0.0 <= r.confidence <= 1.0
    assert r.pages_used == 1
    assert r.alternatives == []  # open-ended mode never has alternatives


def test_constrained_picks_supplied_label_with_alternatives():
    from ingestlib.operations.classify import classify

    cats = {
        "invoice": "Bill for goods or services with amounts due",
        "insurance_certificate": "Proof of insurance coverage listing policies and limits",
        "research_paper": "Academic study with methods and findings",
    }
    r = classify(f"{_PDF}/insurance-acord.pdf", categories=cats)
    assert r.category == "insurance_certificate"
    assert all(a.label in cats and a.label != r.category for a in r.alternatives)


def test_unfitting_categories_yield_uncategorized():
    from ingestlib.operations.classify import classify

    r = classify(
        f"{_PDF}/insurance-acord.pdf",
        categories={"recipe": "Cooking instructions", "poem": "Poetry or verse"},
    )
    assert r.category == "uncategorized"


def test_parse_result_input_needs_no_servers():
    from ingestlib.operations.classify import classify
    from ingestlib.operations.parse.models import PageResult, ParseResult

    pr = ParseResult(
        pages=[PageResult(page_num=1, markdown="INVOICE #99. 3 units @ $5. Total $15 due net 30.")],
        source_path=Path("synthetic.pdf"),
        source_format="pdf",
    )
    r = classify(pr)
    assert "invoice" in r.category
    assert r.pages_used == 1


def test_empty_categories_dict_raises():
    from ingestlib.operations.classify import classify

    with pytest.raises(ValueError, match="non-empty"):
        classify(f"{_PDF}/insurance-acord.pdf", categories={})
