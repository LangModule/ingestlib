"""Chart→table metrics — did the model read the chart's numbers correctly.

A chart→table prediction is a markdown table; the gold is the chart's true data.
The signal that matters for a data chart is the set of numeric values, so both
sides are reduced to their numbers and compared two ways:

  - numeric F1 — a predicted number counts if it matches some gold number within
    a relative tolerance; precision/recall/F1 over the two number sets.
  - RNSS (Relative Number Set Similarity, from DePlot) — a softer score that
    credits near-misses by their relative distance, matched greedily.

RMS (Relative Mapping Similarity) is the stricter DePlot metric that also checks
the row/column mapping; wrap its reference implementation here when cell-level
scoring is needed. Numbers-only is the honest first signal for "read the values".
"""
import re

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def extract_numbers(table_text: str) -> list[float]:
    """Every numeric value in a markdown table (commas stripped; %, $, ~ ignored)."""
    cleaned = table_text.replace(",", "")
    out: list[float] = []
    for match in _NUMBER.finditer(cleaned):
        try:
            out.append(float(match.group()))
        except ValueError:
            continue
    return out


def _matches(pred: float, gold: float, rel_tol: float) -> bool:
    """True when pred is within rel_tol of gold (absolute when gold is zero)."""
    if gold == 0:
        return abs(pred) <= rel_tol
    return abs(pred - gold) / abs(gold) <= rel_tol


def numeric_f1(pred: list[float], gold: list[float], rel_tol: float = 0.05) -> dict[str, float]:
    """Precision/recall/F1 treating each gold number as matchable once."""
    remaining = list(gold)
    hits = 0
    for value in pred:
        for i, target in enumerate(remaining):
            if _matches(value, target, rel_tol):
                hits += 1
                remaining.pop(i)
                break
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rnss(pred: list[float], gold: list[float]) -> float:
    """Relative Number Set Similarity — greedy min-relative-distance matching,
    scored 1 - average distance over the larger set (0 when either is empty)."""
    if not pred or not gold:
        return 0.0
    remaining = list(gold)
    total = 0.0
    for value in pred:
        if not remaining:
            break
        best_i, best_d = 0, float("inf")
        for i, target in enumerate(remaining):
            denom = abs(target) if target != 0 else 1.0
            d = min(1.0, abs(value - target) / denom)
            if d < best_d:
                best_i, best_d = i, d
        total += 1.0 - best_d
        remaining.pop(best_i)
    return total / max(len(pred), len(gold))


def score_chart(prediction: str, gold: str, rel_tol: float = 0.05) -> dict[str, float]:
    """All chart→table metrics for one (predicted table, gold table) pair."""
    pred_nums = extract_numbers(prediction)
    gold_nums = extract_numbers(gold)
    return {
        **numeric_f1(pred_nums, gold_nums, rel_tol),
        "rnss": rnss(pred_nums, gold_nums),
        "pred_count": float(len(pred_nums)),
        "gold_count": float(len(gold_nums)),
    }
