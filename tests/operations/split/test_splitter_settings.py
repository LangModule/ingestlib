"""User-vocabulary/unmatched resolution in split — pure, always run.

achat_structured is monkeypatched with scripted per-page labels, and
propose_vocabulary with a sentinel, so these verify the control flow: when
Pass 1 runs, what the label prompts contain, how each unmatched mode treats
pages that fit nothing, and how the rules.yaml preset interacts with
explicit arguments. No network.
"""
from dataclasses import replace
from pathlib import Path

import pytest

from ingestlib.config import SplitConfig, get_config
from ingestlib.foundations.ocr.models import BoundingBox, Region
from ingestlib.operations.parse.models import PageResult, ParseResult
from ingestlib.operations.split import sections, splitter
from ingestlib.operations.split.sections import _PageLabel, _VocabSection
from ingestlib.operations.split.splitter import MAX_CATEGORIES, asplit

_VOCAB = {"methods": "how the study was run", "results": "findings and outcomes"}


def _doc(n_pages: int = 3) -> ParseResult:
    def page(i: int) -> PageResult:
        text = f"content of page {i}"
        region = Region(
            region_type="text",
            bbox=BoundingBox(x=0, y=0, width=100, height=50),
            region_id=0, text=text, content=text,
        )
        return PageResult(page_num=i, text=text, markdown=text, regions=[region])

    return ParseResult(
        pages=[page(i) for i in range(1, n_pages + 1)],
        source_path=Path("x.pdf"),
        source_format="pdf",
    )


@pytest.fixture()
def scripted(monkeypatch):
    """Script the per-page label answers; record every label prompt.

    Returns (set_labels, prompts). Pass 1 raises unless explicitly allowed —
    a user vocabulary must never trigger discovery.
    """
    state = {"labels": [], "discovery_allowed": False, "discovery_calls": 0}
    prompts: list[str] = []

    async def fake_labels(prompt, schema, images=None, system=None):
        prompts.append(prompt)
        return _PageLabel(category=state["labels"].pop(0))

    async def fake_discovery(pages):
        if not state["discovery_allowed"]:
            raise AssertionError("Pass 1 ran despite a user vocabulary")
        state["discovery_calls"] += 1
        return [_VocabSection(name="document", description="entire document")]

    # operations/__init__ re-exports `split` the function, shadowing the
    # subpackage on dotted-path lookup — patch via the module objects.
    monkeypatch.setattr(sections, "achat_structured", fake_labels)
    monkeypatch.setattr(splitter, "propose_vocabulary", fake_discovery)

    def set_labels(labels, discovery_allowed=False):
        state["labels"] = list(labels)
        state["discovery_allowed"] = discovery_allowed
        return state

    return set_labels, prompts


@pytest.fixture()
def preset(monkeypatch):
    """Pin rules.yaml's split preset for the test."""
    def _set(categories=None, unmatched="other"):
        pinned = replace(
            get_config(),
            split=SplitConfig(categories=categories or {}, unmatched=unmatched),
        )
        monkeypatch.setattr(splitter, "get_config", lambda: pinned)
    _set()
    return _set


async def test_user_vocabulary_skips_pass1_and_names_sections(scripted, preset):
    set_labels, _ = scripted
    set_labels(["methods", "methods", "results"])
    result = await asplit(_doc(3), vocabulary=_VOCAB)
    assert result.section_names == ["methods", "results"]
    assert result.sections[0].pages == [1, 2]
    assert result.sections[0].description == "how the study was run"
    assert result.pages_used == 3


async def test_label_prompt_contains_categories_and_escape(scripted, preset):
    set_labels, prompts = scripted
    set_labels(["methods", "results", "results"])
    await asplit(_doc(3), vocabulary=_VOCAB)  # default unmatched="other"
    assert "- methods: how the study was run" in prompts[0]
    assert "answer exactly 'other'" in prompts[0]


async def test_require_mode_forces_a_match(scripted, preset):
    set_labels, prompts = scripted
    set_labels(["methods", "other", "garbage"])
    result = await asplit(_doc(3), vocabulary=_VOCAB, unmatched="require")
    assert "answer exactly 'other'" not in prompts[0], "require mode offers no escape"
    # both the unwanted "other" and the junk label repair to the left neighbor
    assert result.section_names == ["methods"]
    assert result.sections[0].pages == [1, 2, 3]


async def test_other_mode_groups_an_other_section(scripted, preset):
    set_labels, _ = scripted
    set_labels(["methods", "other", "results"])
    result = await asplit(_doc(3), vocabulary=_VOCAB)
    assert result.section_names == ["methods", "other", "results"]
    other = result.section_by_name("other")
    assert other.pages == [2]
    assert other.description == "pages matching no user category"
    assert other.chunks, "an other section is real content — it still chunks"
    assert any(v.name == "other" for v in result.vocabulary)


async def test_skip_mode_drops_unmatched_pages(scripted, preset):
    set_labels, _ = scripted
    set_labels(["methods", "other", "methods"])
    result = await asplit(_doc(3), vocabulary=_VOCAB, unmatched="skip")
    assert result.section_names == ["methods"]
    covered = [p for s in result.sections for p in s.pages]
    assert covered == [1, 3], "page 2 must be dropped entirely"
    assert "content of page 2" not in "".join(c.markdown for c in result.chunks)
    assert result.pages_used == 3, "pages_used counts pages read, not kept"


async def test_skip_mode_all_pages_unmatched_is_empty(scripted, preset):
    set_labels, _ = scripted
    set_labels(["other", "other"])
    result = await asplit(_doc(2), vocabulary=_VOCAB, unmatched="skip")
    assert result.sections == []
    assert result.pages_used == 2
    assert [v.name for v in result.vocabulary] == ["methods", "results"]


async def test_preset_categories_apply_when_no_vocabulary_passed(scripted, preset):
    preset(categories=_VOCAB, unmatched="skip")
    set_labels, _ = scripted
    set_labels(["methods", "other", "results"])
    result = await asplit(_doc(3))
    assert result.section_names == ["methods", "results"], "preset skip mode applied"


async def test_explicit_unmatched_beats_preset(scripted, preset):
    preset(categories=_VOCAB, unmatched="skip")
    set_labels, _ = scripted
    set_labels(["methods", "other", "results"])
    result = await asplit(_doc(3), unmatched="other")
    assert "other" in result.section_names, "explicit mode wins over the preset"


async def test_empty_dict_forces_discovery_despite_preset(scripted, preset):
    preset(categories=_VOCAB)
    set_labels, _ = scripted
    state = set_labels(["document", "document"], discovery_allowed=True)
    result = await asplit(_doc(2), vocabulary={})
    assert state["discovery_calls"] == 1
    assert result.section_names == ["document"]


async def test_too_many_categories_raises(preset):
    vocab = {f"section_{i}": "d" for i in range(MAX_CATEGORIES + 1)}
    with pytest.raises(ValueError, match=str(MAX_CATEGORIES)):
        await asplit(_doc(), vocabulary=vocab)


async def test_unknown_unmatched_raises(preset):
    with pytest.raises(ValueError, match="unmatched"):
        await asplit(_doc(), vocabulary=_VOCAB, unmatched="banana")


async def test_unmatched_without_vocabulary_raises(preset):
    with pytest.raises(ValueError, match="user vocabulary"):
        await asplit(_doc(), unmatched="skip")
