"""Chart→table samples — a normalized view over ChartX / PlotQA.

The harness never touches a dataset's raw layout. Each chart dataset is converted
once into a JSONL manifest of the shape below, and this loader reads that:

    {"id": "chartx-0001", "image": "images/0001.png",
     "table": "| year | sales |\n|---|---|\n| 2019 | 12.4 |\n| 2020 | 15.1 |",
     "type": "bar"}

- `image` is relative to the manifest file's directory.
- `table` is the ground-truth data as a markdown table (or CSV text) — the
  scorer reduces it to numbers, so either renders fine.
- `type` (optional) lets a run break scores down by chart kind.

To build a manifest from a public set: for each sample, write its chart image and
its ground-truth table (ChartX ships the table per chart; PlotQA ships the
underlying data) as one JSONL line. Keep the images under the manifest's folder.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChartSample:
    id: str
    image_path: Path
    gold_table: str
    chart_type: str = ""


def load_manifest(manifest_path: str | Path) -> list[ChartSample]:
    """Read a chart→table JSONL manifest into samples (image paths resolved
    relative to the manifest's directory)."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"chart manifest not found: {path}\n"
            f"build one from ChartX/PlotQA — see evals/datasets/charts.py for the format"
        )
    base = path.parent
    samples: list[ChartSample] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        image = (base / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_no} — image not found: {image}")
        samples.append(ChartSample(
            id=str(row["id"]),
            image_path=image,
            gold_table=str(row["table"]),
            chart_type=str(row.get("type", "")),
        ))
    return samples
