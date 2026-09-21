"""The honesty layer — citations verified, values grounded, confidence capped.
Pure, always run."""
from pydantic import BaseModel

from ingestlib.operations.extract.context import SourcePage
from ingestlib.operations.extract.extractor import _FieldEvidence
from ingestlib.operations.extract.grounding import (
    assess_item,
    parse_ref,
    resolve_bare_region,
    value_grounded,
)


def _page(page_num: int, region_texts: dict[int, str] | None) -> SourcePage:
    """region_texts=None models the native path (no region markers)."""
    texts = region_texts or {}
    return SourcePage(
        page_num=page_num,
        text="\n".join(texts.values()) if region_texts is not None else "native page text",
        region_ids=frozenset(texts) if region_texts is not None else None,
        region_texts=texts,
        images=[],
    )


# ---------- ref parsing ----------


def test_parse_ref_forms():
    assert parse_ref("p4:r2") == (4, 2)
    assert parse_ref("p4") == (4, None)
    assert parse_ref("[p4:r2]") == (4, 2), "models copy markers verbatim"
    assert parse_ref(" p10:r0 ") == (10, 0)
    assert parse_ref("region 2") is None
    assert parse_ref("r2") is None, "bare regions go through resolve_bare_region"


def test_bare_region_resolves_only_when_unambiguous():
    window = [_page(1, {2: "alpha"}), _page(2, {5: "beta"})]
    assert resolve_bare_region("r5", window) == (2, 5)
    ambiguous = [_page(1, {2: "a"}), _page(2, {2: "b"})]
    assert resolve_bare_region("r2", ambiguous) is None
    assert resolve_bare_region("r2", [_page(1, None)]) is None, "native has no regions"


# ---------- grounding ----------


def test_numbers_ground_across_printed_forms():
    assert value_grounded(383285.0, "Total net sales $ 383,285") is True
    assert value_grounded(20.0, "AMOUNT $ 20.00") is True
    assert value_grounded(18, "grew 18% year over year") is True
    assert value_grounded(99.5, "nothing here") is False


def test_strings_ground_case_and_space_insensitive():
    assert value_grounded("Apple Inc.", "APPLE  INC. CONSOLIDATED") is True
    assert value_grounded("BART", "Thanks for riding BART.") is True
    assert value_grounded("Acme Corp", "totally unrelated") is False


def test_uncheckable_values_return_none():
    assert value_grounded(None, "anything") is None
    assert value_grounded(True, "true anything") is None
    assert value_grounded("$,%", "text") is None, "nothing left after normalization"


# ---------- assess_item ----------


class _Receipt(BaseModel):
    merchant: str
    total: float


def _evidence(field: str, sources: list[str], confidence: float = 0.9) -> _FieldEvidence:
    return _FieldEvidence(field=field, sources=sources, confidence=confidence)


def test_valid_citation_grounds_and_keeps_confidence():
    window = [_page(10, {3: "BART San Francisco AMOUNT $ 20.00"})]
    data = _Receipt(merchant="BART", total=20.0)
    fields, pages = assess_item(
        data,
        [_evidence("merchant", ["p10:r3"]), _evidence("total", ["p10:r3"], 0.95)],
        window,
    )
    assert fields["total"].grounded is True
    assert fields["total"].confidence == 0.95
    assert fields["total"].region_ids == {10: [3]}
    assert pages == [10]


def test_hallucinated_region_is_dropped_and_capped():
    window = [_page(10, {3: "real text"})]
    data = _Receipt(merchant="BART", total=20.0)
    fields, _ = assess_item(
        data, [_evidence("merchant", ["p10:r99"], 0.99)], window,
    )
    assert fields["merchant"].region_ids == {}
    assert fields["merchant"].confidence == 0.3, "no valid citation → capped"


def test_page_outside_window_is_not_trusted():
    window = [_page(10, {3: "text"})]
    fields, _ = assess_item(
        _Receipt(merchant="x", total=1.0),
        [_evidence("merchant", ["p99:r3"], 0.99)],
        window,
    )
    assert fields["merchant"].confidence == 0.3


def test_ungrounded_value_capped_at_half():
    window = [_page(10, {3: "completely different content"})]
    fields, _ = assess_item(
        _Receipt(merchant="BART", total=20.0),
        [_evidence("merchant", ["p10:r3"], 0.99)],
        window,
    )
    assert fields["merchant"].grounded is False
    assert fields["merchant"].confidence == 0.5


def test_uncited_field_gets_floor_confidence():
    window = [_page(10, {3: "BART"})]
    fields, _ = assess_item(
        _Receipt(merchant="BART", total=20.0),
        [_evidence("merchant", ["p10:r3"])],   # nothing for `total`
        window,
    )
    assert fields["total"].confidence == 0.3
    assert fields["total"].grounded is None
    assert fields["total"].pages == []


def test_duplicate_refs_dedup_region_ids():
    window = [_page(10, {3: "BART 20.00"})]
    fields, _ = assess_item(
        _Receipt(merchant="BART", total=20.0),
        [_evidence("total", ["p10:r3", "p10:r3", "[p10:r3]"])],
        window,
    )
    assert fields["total"].region_ids == {10: [3]}


def test_native_page_citation_grounds_against_page_text():
    window = [_page(1, None)]  # native path
    class _Doc(BaseModel):
        phrase: str

    fields, pages = assess_item(
        _Doc(phrase="native page"),
        [_evidence("phrase", ["p1"], 0.8)],
        window,
    )
    assert fields["phrase"].grounded is True
    assert fields["phrase"].region_ids == {}, "page-level only on the native path"
    assert pages == [1]


def test_evidence_for_unknown_field_is_ignored():
    window = [_page(1, {0: "text"})]
    fields, _ = assess_item(
        _Receipt(merchant="x", total=1.0),
        [_evidence("not_a_field", ["p1:r0"])],
        window,
    )
    assert set(fields) == {"merchant", "total"}
