"""PaddleOCR-VL wrapper — layout detection + VLM recognition for every region type.

Two-stage pipeline: PP-DocLayout detects regions locally, then the 0.9B VL model
(served via mlx_vlm.server on Metal, or vLLM on NVIDIA) recognizes each region in
parallel — text, tables (HTML), formulas (LaTeX), charts (HTML data tables), seals.

Public API: run_full_pipeline / arun_full_pipeline. Callers pass raw image bytes
and receive a LayoutResult.
"""
import asyncio
import logging
import os
import threading
import time
import warnings
from html.parser import HTMLParser
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import Any

import httpx
from PIL import Image

# Skip the per-process network probe of model hosters — models download from the
# default hoster on first use either way. Overridable via the environment.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import PaddleOCRVL  # noqa: E402

from ingestlib.config import get_paddle_vl_config  # noqa: E402
from ingestlib.foundations.ocr.models import (  # noqa: E402
    BoundingBox,
    LayoutResult,
    Region,
    RegionType,
)
from ingestlib.utils.logger import get_logger  # noqa: E402


logger = get_logger(__name__)

# paddlex resets its own logger to INFO at import (after our configure() already
# quieted it) — re-apply the third-party policy so its chatter doesn't drown ours.
if os.environ.get("INGESTLIB_LOG_THIRD_PARTY") != "1":
    logging.getLogger("paddlex").setLevel(logging.WARNING)

# The pipeline's default VL config carries min/max_pixels, which the mlx-vlm-server
# backend doesn't accept — paddlex warns and drops them on every predict. Not actionable.
warnings.filterwarnings("ignore", message=r".*does not support `(min|max)_pixels`.*")


# PP-DocLayout's fine-grained labels → the 12 canonical RegionType values.
# Both short and long variants mapped for safety. Unknown labels fall through to "text".
_LABEL_MAP: dict[str, RegionType] = {
    # titles
    "doc_title": "title",
    "document_title": "title",
    "paragraph_title": "title",
    # text flavors → "text"
    "text": "text",
    "abstract": "text",
    "content": "text",
    "table_of_contents": "text",
    "footnote": "text",
    "footnotes": "text",
    "algorithm": "text",
    "aside_text": "text",
    "number": "text",
    # captions
    "figure_title": "figure_caption",
    "figure_caption": "figure_caption",
    "image_caption": "figure_caption",
    "vision_footnote": "figure_caption",
    "table_title": "table_caption",
    "table_caption": "table_caption",
    # direct
    "table": "table",
    "formula": "formula",
    "formula_number": "formula",
    "chart": "chart",
    # a chart's title is caption text — typing it "chart" would send a text
    # crop through the chart→data-table enricher (fabrication risk)
    "chart_title": "figure_caption",
    "seal": "seal",
    "reference": "reference",
    "references": "reference",
    "figure": "figure",
    "image": "figure",
    "header": "header",
    "footer": "footer",
    "header_image": "header",
    "footer_image": "footer",
    "page_number": "footer",
}


_pipeline: PaddleOCRVL | None = None
_lock = threading.Lock()


def _check_server(server_url: str, backend: str) -> None:
    """Fail fast with a clear message if the VLM inference server isn't reachable."""
    try:
        httpx.get(f"{server_url.rstrip('/')}/v1/models", timeout=3.0).raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"PaddleOCR-VL inference server not reachable at {server_url} "
            f"(backend={backend}). Start it with:\n"
            f"  uv run python -m mlx_vlm.server --port 8111 "
            f"--model PaddlePaddle/PaddleOCR-VL-1.6"
        ) from exc


def _get_pipeline() -> PaddleOCRVL:
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None:
            cfg = get_paddle_vl_config()
            _check_server(cfg.server_url, cfg.backend)
            logger.info(
                "building PaddleOCR-VL pipeline: server=%s model=%s",
                cfg.server_url, cfg.api_model_name,
            )
            t0 = time.perf_counter()
            _pipeline = PaddleOCRVL(
                vl_rec_backend=cfg.backend,
                vl_rec_server_url=cfg.server_url,
                vl_rec_api_model_name=cfg.api_model_name,
                use_chart_recognition=True,   # charts → HTML data tables
                use_seal_recognition=True,
                # Never enable: unwarping distorts coordinates, breaking the
                # bbox → source-PDF mapping the hover-highlight UI depends on.
                use_doc_unwarping=False,
            )
            logger.info("PaddleOCR-VL pipeline ready in %.1fs", time.perf_counter() - t0)
    return _pipeline


def _predict(image_bytes: bytes) -> list[Any]:
    """Write bytes to a temp PNG and run PaddleOCR-VL, returning per-page results."""
    logger.info("paddle_vl predict start: image_size=%d bytes", len(image_bytes))
    t0 = time.perf_counter()
    with NamedTemporaryFile(suffix=".png", delete=True) as f:
        Image.open(BytesIO(image_bytes)).save(f.name, format="PNG")
        results = list(_get_pipeline().predict(f.name))
    logger.info(
        "paddle_vl predict done in %.2fs (%d page result(s))",
        time.perf_counter() - t0, len(results),
    )
    return results


def _get_res(pp_result: Any) -> dict[str, Any]:
    """Result payload lives under a 'res' key inside res.json."""
    payload = pp_result.json
    if isinstance(payload, dict) and "res" in payload:
        return payload["res"]
    return payload if isinstance(payload, dict) else {}


def _page_dims_from_result(res: dict[str, Any], fallback_bytes: bytes) -> tuple[int, int]:
    """Prefer the pipeline's reported page size; fall back to PIL."""
    w, h = res.get("width"), res.get("height")
    if isinstance(w, int) and isinstance(h, int):
        return w, h
    img = Image.open(BytesIO(fallback_bytes))
    return img.width, img.height


def _bbox_from_coord(coord: Any) -> BoundingBox:
    x1, y1, x2, y2 = coord
    return BoundingBox(x=float(x1), y=float(y1), width=float(x2 - x1), height=float(y2 - y1))


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML-to-text: strips tags, keeps text with single spaces between cells."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    @property
    def text(self) -> str:
        return " ".join(self._parts)


def _strip_html(source: str) -> str:
    if "<" not in source:
        return source
    parser = _HTMLTextExtractor()
    parser.feed(source)
    return parser.text


def _plain_text(region_type: RegionType, raw: str) -> str:
    """Convert block_content to plain text based on region_type.

    Tables AND charts come back as HTML — strip tags to get cell values.
    Formulas come back as LaTeX; the LaTeX source is treated as plain text.
    Figures have no text — return empty. Seals carry recognized text.
    """
    if region_type == "figure":
        return ""
    if region_type in ("table", "chart"):
        return _strip_html(raw)
    return raw


def _block_to_region(block: dict[str, Any], region_id: int) -> Region:
    """One entry from parsing_res_list → Region.

    text     — plain-text form (HTML stripped for tables/charts, empty for figures).
    content  — raw block_content (HTML for tables/charts, LaTeX for formulas, text otherwise).
    """
    label = block.get("block_label", "text")
    region_type = _LABEL_MAP.get(label, "text")
    raw = str(block.get("block_content") or "")
    return Region(
        region_type=region_type,
        bbox=_bbox_from_coord(block["block_bbox"]),
        region_id=region_id,
        text=_plain_text(region_type, raw),
        content=raw,
        confidence=1.0,  # parsing_res_list does not expose per-block confidence
    )


def _extract_regions(res: dict[str, Any]) -> list[Region]:
    """Return regions in the pipeline's reading order, region_id = reading-order index."""
    blocks = res.get("parsing_res_list") or []
    return [_block_to_region(b, region_id=i) for i, b in enumerate(blocks)]


def _extract_markdown(pp_result: Any) -> str:
    md = pp_result.markdown
    if isinstance(md, dict):
        return str(md.get("markdown_texts") or "")
    return str(md) if md else ""


def _extract_text(regions: list[Region]) -> str:
    """Plain-text form of the page — one line per text-bearing region, reading order."""
    return "\n".join(r.text for r in regions if r.text)


# ---------- public API ----------


def run_full_pipeline(image_bytes: bytes) -> LayoutResult:
    """End-to-end parse of a page image: layout + VL recognition → LayoutResult."""
    results = _predict(image_bytes)
    if not results:
        w, h = Image.open(BytesIO(image_bytes)).size
        logger.warning("run_full_pipeline: predict returned no results, empty page %dx%d", w, h)
        return LayoutResult(regions=[], page_width=w, page_height=h)
    if len(results) > 1:
        logger.warning(
            "run_full_pipeline: predict returned %d page results for one image — "
            "using the first, dropping the rest", len(results),
        )
    pp = results[0]
    res = _get_res(pp)
    width, height = _page_dims_from_result(res, image_bytes)
    regions = _extract_regions(res)
    logger.info("run_full_pipeline: %d regions on %dx%d page", len(regions), width, height)
    return LayoutResult(
        regions=regions,
        page_width=width,
        page_height=height,
        text=_extract_text(regions),
        markdown=_extract_markdown(pp),
    )


async def arun_full_pipeline(image_bytes: bytes) -> LayoutResult:
    """Async run_full_pipeline() — runs the sync pipeline in a worker thread."""
    return await asyncio.to_thread(run_full_pipeline, image_bytes)
