"""Data models returned by extract(): FieldValue, ExtractedItem, ExtractResult."""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class FieldValue(BaseModel):
    """Provenance record for ONE top-level field of an extracted item.

    confidence — the model's self-score, CAPPED by verification: a value
                 whose citation didn't check out can't report certainty
    region_ids — page → parse region ids the value was read from (empty on
                 the native no-parse path, which is page-level only)
    pages      — 1-based pages the value came from
    grounded   — True: the value's text was found in its cited source;
                 False: cited but not found; None: not checkable (booleans,
                 empty values, uncited fields)
    """

    model_config = ConfigDict(frozen=True)

    confidence: float = Field(ge=0.0, le=1.0)
    region_ids: dict[int, list[int]] = Field(default_factory=dict)
    pages: list[int] = Field(default_factory=list)
    grounded: bool | None = None


class ExtractedItem(BaseModel):
    """One validated instance of the caller's schema, with per-field provenance.

    value  — an instance of the schema passed to extract() (a plain dict
             after an artifact round-trip until revalidated by load_extract)
    fields — one FieldValue per top-level schema field
    pages  — every page this item drew from
    """

    model_config = ConfigDict(frozen=True)

    value: Any
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    pages: list[int] = Field(default_factory=list)

    @field_serializer("value")
    def _serialize_value(self, value: Any) -> Any:
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    @property
    def citation(self) -> str:
        """Human-readable source pointer, e.g. 'p.4' or 'p.1,2'."""
        pages = ",".join(str(p) for p in self.pages) or "?"
        return f"p.{pages}"


class ExtractResult(BaseModel):
    """Everything extract() produced from one document."""

    model_config = ConfigDict(frozen=True)

    items: list[ExtractedItem] = Field(default_factory=list)
    schema_name: str
    mode: str                       # "one" | "many"
    pages_used: int = 0
    duration_seconds: float = 0.0

    @property
    def values(self) -> list[Any]:
        """Just the validated schema instances, in document order."""
        return [item.value for item in self.items]
