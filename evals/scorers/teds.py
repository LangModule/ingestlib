"""Table-similarity metrics — did the parse recover the table's cells.

The field-standard metric is TEDS (Tree-Edit-Distance Similarity), which scores an
HTML table against gold as normalized tree-edit distance; wrap its reference
implementation (PubTabNet's) for publication numbers — it needs apted + lxml.

The transparent first signal here is cell-content F1 (position-agnostic multiset of
normalized cell texts) plus a shape match (same rows × cols). It reads whether the
values were recovered without requiring the tree-edit dependency.
"""
import re

_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _norm(cell: str) -> str:
    return _WHITESPACE.sub(" ", _TAG.sub("", cell)).strip().lower()


def _cells_from_html(html: str) -> list[str]:
    return [_norm(c) for c in _CELL.findall(html) if _norm(c)]


def _cells_from_markdown(md: str) -> list[str]:
    cells: list[str] = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if re.fullmatch(r"[|\s:\-]+", line):  # the |---|---| separator row
            continue
        for cell in line.strip("|").split("|"):
            token = _norm(cell)
            if token:
                cells.append(token)
    return cells


def extract_cells(table_text: str) -> list[str]:
    """Normalized cell texts from an HTML or markdown table."""
    if "<td" in table_text.lower() or "<table" in table_text.lower():
        return _cells_from_html(table_text)
    return _cells_from_markdown(table_text)


def _shape(table_text: str) -> tuple[int, int]:
    """(rows, cols) of an HTML or markdown table — 0,0 when it isn't a grid."""
    if "<tr" in table_text.lower():
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_text, re.IGNORECASE | re.DOTALL)
        cols = max((len(re.findall(r"<t[dh]\b", r, re.IGNORECASE)) for r in rows), default=0)
        return len(rows), cols
    body = [
        line for line in table_text.splitlines()
        if line.strip().startswith("|") and not re.fullmatch(r"[|\s:\-]+", line.strip())
    ]
    cols = max((line.strip("| ").count("|") + 1 for line in body), default=0)
    return len(body), cols


def score_table(prediction: str, gold: str) -> dict[str, float]:
    """Cell-content F1 (multiset) + a shape match for one (predicted, gold) table."""
    pred = extract_cells(prediction)
    gold_cells = extract_cells(gold)
    remaining = list(gold_cells)
    hits = 0
    for cell in pred:
        if cell in remaining:
            hits += 1
            remaining.remove(cell)
    precision = hits / len(pred) if pred else 0.0
    recall = hits / len(gold_cells) if gold_cells else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "cell_precision": precision,
        "cell_recall": recall,
        "cell_f1": f1,
        "shape_match": 1.0 if _shape(prediction) == _shape(gold) else 0.0,
    }
