"""Run ingestlib's parse over a formula image → the recovered LaTeX.

Parse emits formula regions with their content as LaTeX (region_type "formula").
This collects them for a formula crop. Needs the OCR server (GPU).
"""
from pathlib import Path

from ingestlib.operations import aparse


async def predict_latex(image_path: Path) -> str:
    """A formula image → its recovered LaTeX."""
    result = await aparse(image_path)
    formulas = [
        region.content
        for page in result.pages
        for region in page.regions
        if region.region_type == "formula" and region.content.strip()
    ]
    return "\n".join(formulas)
