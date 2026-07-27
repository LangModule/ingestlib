"""Citation validation and grounding — extract()'s honesty layer.

The model claims where it read each value; nothing here trusts the claim.
Cited regions must exist on the cited page, and the value's text must
actually appear in the cited source, or the field's confidence is capped:

    cited + found in source        → the model's confidence, as reported
    cited but NOT found            → capped at 0.5, grounded=False
    no (valid) citation at all     → capped at 0.3

Grounding is plain normalized text matching against content we already
hold — no second model, which is why it's cheap enough to always run.
"""
import re
from typing import Any

from pydantic import BaseModel

from ingestlib.operations.extract.context import SourcePage
from ingestlib.operations.extract.models import FieldValue
from ingestlib.utils.logger import get_logger


logger = get_logger(__name__)

_UNGROUNDED_CAP = 0.5
_UNCITED_CAP = 0.3

_REF = re.compile(r"^p(\d+)(?::r(\d+))?$")
_BARE_REGION = re.compile(r"^r(\d+)$")


def parse_ref(ref: str) -> tuple[int, int | None] | None:
    """'p4:r2' → (4, 2); 'p4' → (4, None); malformed → None.

    Brackets are tolerated ('[p4:r2]') — models copy markers verbatim."""
    m = _REF.match(ref.strip().strip("[]"))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def resolve_bare_region(ref: str, window: list[SourcePage]) -> tuple[int, int] | None:
    """A page-less 'r2' is usable only when exactly ONE window page has that
    region id — otherwise it's ambiguous and dropped."""
    m = _BARE_REGION.match(ref.strip().strip("[]"))
    if not m:
        return None
    region_id = int(m.group(1))
    holders = [
        p.page_num for p in window
        if p.region_ids is not None and region_id in p.region_ids
    ]
    return (holders[0], region_id) if len(holders) == 1 else None


def _normalize(text: str) -> str:
    """Comparison form: lowercase, no spaces/commas/currency marks."""
    return re.sub(r"[\s,€$£%]", "", text.lower())


def _number_forms(value: float) -> list[str]:
    """'20.0' must match '20.00', '20', or '20.5' in printed text."""
    forms = {f"{value}", f"{value:g}", f"{value:.2f}"}
    if float(value).is_integer():
        forms.add(str(int(value)))
    return list(forms)


def value_grounded(value: Any, source_text: str) -> bool | None:
    """Does the value's text appear in the cited source? None = not checkable."""
    if value is None or isinstance(value, bool):
        return None
    haystack = _normalize(source_text)
    if isinstance(value, (int, float)):
        return any(_normalize(f) in haystack for f in _number_forms(float(value)))
    needle = _normalize(str(value))
    if not needle:
        return None
    return needle in haystack


def assess_item(
    data: BaseModel,
    evidence: list,                     # _FieldEvidence entries from the model
    window: list[SourcePage],
) -> tuple[dict[str, FieldValue], list[int]]:
    """Turn the model's claims into verified FieldValues.

    Returns (per-field provenance, the item's page list).
    """
    by_page = {p.page_num: p for p in window}
    claims = {e.field: e for e in evidence if e.field in type(data).model_fields}

    fields: dict[str, FieldValue] = {}
    item_pages: set[int] = set()

    for name in type(data).model_fields:
        claim = claims.get(name)
        if claim is None:
            fields[name] = FieldValue(confidence=_UNCITED_CAP)
            continue

        region_ids: dict[int, list[int]] = {}
        pages: set[int] = set()
        cited_text: list[str] = []
        for ref in claim.sources:
            parsed = parse_ref(ref) or resolve_bare_region(ref, window)
            if parsed is None:
                continue
            page_num, region_id = parsed
            page = by_page.get(page_num)
            if page is None:
                continue  # cited a page outside this window — not trusted
            if region_id is None:
                if page.region_ids is None:  # native path: page-level is all there is
                    pages.add(page_num)
                    cited_text.append(page.text)
                continue
            if page.region_ids is None or region_id not in page.region_ids:
                logger.info(
                    "dropping hallucinated citation p%d:r%d for field %r",
                    page_num, region_id, name,
                )
                continue
            pages.add(page_num)
            region_ids.setdefault(page_num, []).append(region_id)
            cited_text.append(page.region_texts[region_id])

        value = getattr(data, name, None)
        confidence = min(max(claim.confidence, 0.0), 1.0)
        if not pages:
            fields[name] = FieldValue(confidence=min(confidence, _UNCITED_CAP))
            continue

        grounded = value_grounded(value, "\n".join(cited_text))
        if grounded is False:
            confidence = min(confidence, _UNGROUNDED_CAP)
        fields[name] = FieldValue(
            confidence=confidence,
            region_ids={k: sorted(set(v)) for k, v in region_ids.items()},
            pages=sorted(pages),
            grounded=grounded,
        )
        item_pages.update(pages)

    return fields, sorted(item_pages)
