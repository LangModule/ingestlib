"""Formula-recognition samples — a normalized view over UniMER-Test / im2latex.

Each formula dataset converts once into a JSONL manifest of the shape:

    {"id": "unimer-0001", "image": "images/0001.png", "latex": "\\frac{a}{b} + c^2"}

- `image` (a cropped formula) is relative to the manifest's directory.
- `latex` is the ground-truth LaTeX the recognizer should produce.
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FormulaSample:
    id: str
    image_path: Path
    gold_latex: str


def load_manifest(manifest_path: str | Path) -> list[FormulaSample]:
    """Read a formula JSONL manifest into samples (image paths resolved relative to it)."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"formula manifest not found: {path}\n"
            f"build one from UniMER-Test/im2latex — see evals/datasets/formulas.py for the format"
        )
    base = path.parent
    samples: list[FormulaSample] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        image = (base / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{path}:{line_no} — image not found: {image}")
        samples.append(FormulaSample(
            id=str(row["id"]),
            image_path=image,
            gold_latex=str(row["latex"]),
        ))
    return samples
