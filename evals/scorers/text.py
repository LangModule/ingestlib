"""Text-similarity metrics — normalized edit distance + exact match.

Used for parse (page markdown vs gold) and formula recognition (predicted LaTeX
vs gold). Edit similarity is 1 - Levenshtein/max_len over normalized strings;
exact is all-or-nothing after the same normalization.

The field-standard formula metric is CDM (Character Detection Matching), which is
invariant to LaTeX representation differences; wrap its reference implementation
for publication numbers. Normalized edit distance is the transparent first signal.
"""
import re

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str, collapse_ws: bool = True) -> str:
    text = text.strip()
    if collapse_ws:
        text = _WHITESPACE.sub(" ", text)
    return text


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (insert/delete/substitute = 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(pred: str, gold: str, collapse_ws: bool = True) -> float:
    """1 - Levenshtein/max_len over the normalized strings (1.0 when both empty)."""
    p, g = normalize(pred, collapse_ws), normalize(gold, collapse_ws)
    if not p and not g:
        return 1.0
    return 1.0 - levenshtein(p, g) / max(len(p), len(g))


def score_text(pred: str, gold: str, collapse_ws: bool = True) -> dict[str, float]:
    """Edit similarity + exact match for one (prediction, gold) pair."""
    p, g = normalize(pred, collapse_ws), normalize(gold, collapse_ws)
    return {
        "edit_similarity": edit_similarity(pred, gold, collapse_ws),
        "exact": 1.0 if p == g else 0.0,
    }
