"""Rules/page-settings resolution in classify — pure, always run.

achat_structured is monkeypatched to capture the prompt and schema, so these
verify what would be asked of the model: which categories constrain the call,
which pages made it in, and how the rules.yaml preset interacts with
explicit arguments. No network.
"""
from dataclasses import replace
from pathlib import Path

import pytest

from ingestlib.config import ClassifyConfig, get_config
from ingestlib.operations.classify import classifier
from ingestlib.operations.classify.classifier import MAX_RULES, aclassify
from ingestlib.operations.parse.models import PageResult, ParseResult

_RULES = {"invoice": "itemized charges", "contract": "signed agreement terms"}


def _doc(n_pages: int = 3) -> ParseResult:
    return ParseResult(
        pages=[PageResult(page_num=i, markdown=f"content of page {i}") for i in range(1, n_pages + 1)],
        source_path=Path("x.pdf"),
        source_format="pdf",
    )


@pytest.fixture()
def capture(monkeypatch):
    """Fake achat_structured recording (prompt, schema) and answering validly."""
    calls: list[tuple[str, type]] = []

    async def fake(prompt, schema, images=None, system=None):
        calls.append((prompt, schema))
        if schema is classifier._ConstrainedVerdict:
            return classifier._ConstrainedVerdict(
                category="invoice", confidence=0.9, reasoning="r", alternatives=[]
            )
        return classifier._Verdict(category="report", confidence=0.9, reasoning="r")

    monkeypatch.setattr(classifier, "achat_structured", fake)
    return calls


@pytest.fixture()
def preset(monkeypatch):
    """Pin rules.yaml's classify preset for the test."""
    def _set(rules=None, target_pages="", max_pages=0):
        pinned = replace(
            get_config(),
            classify=ClassifyConfig(
                rules=rules or {}, target_pages=target_pages, max_pages=max_pages
            ),
        )
        monkeypatch.setattr(classifier, "get_config", lambda: pinned)
    _set()
    return _set


async def test_explicit_categories_constrain_the_call(capture, preset):
    result = await aclassify(_doc(), _RULES)
    prompt, schema = capture[0]
    assert schema is classifier._ConstrainedVerdict
    assert "invoice: itemized charges" in prompt and "uncategorized" in prompt
    assert result.category == "invoice"


async def test_no_arguments_and_no_preset_is_open_ended(capture, preset):
    result = await aclassify(_doc())
    _, schema = capture[0]
    assert schema is classifier._Verdict
    assert result.category == "report"


async def test_preset_rules_apply_when_no_categories_passed(capture, preset):
    preset(rules=_RULES)
    await aclassify(_doc())
    prompt, schema = capture[0]
    assert schema is classifier._ConstrainedVerdict
    assert "contract: signed agreement terms" in prompt


async def test_empty_dict_forces_open_ended_despite_preset(capture, preset):
    preset(rules=_RULES)
    await aclassify(_doc(), {})
    _, schema = capture[0]
    assert schema is classifier._Verdict


async def test_preset_page_settings_apply(capture, preset):
    preset(target_pages="2", max_pages=5)
    result = await aclassify(_doc(3))
    prompt, _ = capture[0]
    assert "content of page 2" in prompt and "content of page 1" not in prompt
    assert result.pages_used == 1


async def test_explicit_page_settings_beat_the_preset(capture, preset):
    preset(target_pages="2")
    result = await aclassify(_doc(5), target_pages="1, 4-5", max_pages=2)
    prompt, _ = capture[0]
    assert "content of page 1" in prompt and "content of page 4" in prompt
    assert "content of page 5" not in prompt, "max_pages caps after selection"
    assert result.pages_used == 2


async def test_too_many_rules_raises(preset):
    rules = {f"type_{i}": "d" for i in range(MAX_RULES + 1)}
    with pytest.raises(ValueError, match=str(MAX_RULES)):
        await aclassify(_doc(), rules)
