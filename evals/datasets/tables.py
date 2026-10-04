"""Table-recognition samples — a normalized view over FinTabNet / PubTabNet.

Each table dataset converts once into a JSONL manifest of the shape:

    {"id": "fintabnet-0001", "image": "images/0001.png",
     "table": "<table><tr><td>a</td><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>"}

- `image` (a cropped table, or a page) is relative to the manifest's directory.
- `table` is the ground-truth table as HTML (or a markdown table) — the scorer
  reads cells from either.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TableSample:
    id: str
    image_path: Path
    gold_table: str


def load_manifest(manifest_path: str | Path) -> list[TableSample]:
    """Read a table JSONL manifest into samples (image paths resolved relative to it)."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"table manifest not found: {path}\n"
            f"build one from FinTabNet/PubTabNet — see evals/datasets/tables.py for the format"
        )
    base = path.parent
    samples: list[TableSample] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        image = (base / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_no} — image not found: {image}")
        samples.append(TableSample(
            id=str(row["id"]),
            image_path=image,
            gold_table=str(row["table"]),
        ))
    return samples
