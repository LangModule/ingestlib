"""Run ingestlib's chart→table path on a chart image.

This calls the exact primitive parse()'s enricher uses on a chart crop — the same
_PROMPT and the same vision LLM call — so the eval measures the shipped behavior,
not a reimplementation. It needs only the configured LLM (no OCR server).
"""
from ingestlib.foundations.llm import Image as LLMImage
from ingestlib.foundations.llm import achat
from ingestlib.operations.parse.enricher import _PROMPT


async def predict_chart_table(image_bytes: bytes, image_format: str = "png") -> str:
    """A chart image → the model's markdown data table."""
    response = await achat(_PROMPT, images=[LLMImage(data=image_bytes, format=image_format)])
    return response.strip()
