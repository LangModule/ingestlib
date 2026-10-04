"""Run ingestlib's parse over a table image → the recovered table as HTML.

Parse emits table regions with their structure as HTML (region_type "table"). This
collects them for a table crop or page. Needs the OCR server (GPU).
"""
from pathlib import Path

from ingestlib.operations import aparse


async def predict_table_html(image_path: Path) -> str:
    """A table image → its recovered table region(s) as HTML."""
    result = await aparse(image_path)
    tables = [
        region.content
        for page in result.pages
        for region in page.regions
        if region.region_type == "table" and region.content.strip()
    ]
    return "\n".join(tables)
