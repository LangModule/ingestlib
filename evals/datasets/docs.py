"""Full-page parse samples — a normalized view over OmniDocBench / olmOCR-Bench.

Each doc dataset converts once into a JSONL manifest of the shape:

    {"id": "omnidoc-0001", "doc": "pdfs/0001.pdf",
     "markdown": "# Title\n\nBody text…\n\n| a | b |\n|---|---|\n| 1 | 2 |"}

- `doc` (a PDF or image) is relative to the manifest's directory.
- `markdown` is the ground-truth page/document content the parse should produce.

OmniDocBench ships per-page markdown ground truth; olmOCR-Bench ships unit tests
(convert each to a markdown-equivalence check, or extend this with a test runner).
"""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocSample:
    id: str
    doc_path: Path
    gold_markdown: str


def load_manifest(manifest_path: str | Path) -> list[DocSample]:
    """Read a parse JSONL manifest into samples (doc paths resolved relative to it)."""
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"parse manifest not found: {path}\n"
            f"build one from OmniDocBench/olmOCR — see evals/datasets/docs.py for the format"
        )
    base = path.parent
    samples: list[DocSample] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        doc = (base / row["doc"]).resolve()
        if not doc.is_file():
            raise FileNotFoundError(f"{path}:{line_no} — doc not found: {doc}")
        samples.append(DocSample(
            id=str(row["id"]),
            doc_path=doc,
            gold_markdown=str(row["markdown"]),
        ))
    return samples
