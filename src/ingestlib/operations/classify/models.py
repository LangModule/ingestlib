"""Data models returned by classify(): CategoryScore and ClassifyResult.

Frozen Pydantic v2 models, matching the parse operation's conventions.
"""
from pydantic import BaseModel, ConfigDict, Field


class CategoryScore(BaseModel):
    """One runner-up category with its relevance score (populated only when the
    caller supplied a categories dict)."""

    model_config = ConfigDict(frozen=True)

    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClassifyResult(BaseModel):
    """Document classification verdict.

    category      — snake_case label; one of the caller's categories (or
                    "uncategorized") when categories were supplied, otherwise
                    an open-ended label Nova generated from the content
    confidence    — Nova's 0-1 confidence in the verdict
    reasoning     — one-to-two sentence justification
    alternatives  — ranked runner-up categories; empty in open-ended mode
    pages_used    — how many pages were actually read (caps at 100)
    """

    model_config = ConfigDict(frozen=True)

    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    alternatives: list[CategoryScore] = Field(default_factory=list)
    pages_used: int = 0
