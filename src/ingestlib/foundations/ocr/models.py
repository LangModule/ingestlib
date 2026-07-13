"""Shared OCR result types — consumed by paddle_vl.py and the parse operation.

Frozen dataclasses so downstream code updates a region via dataclasses.replace()
instead of mutating in place — safer under concurrent page processing.

Bounding boxes are in rendered-image pixels (top-left origin). For UI overlays,
convert with BoundingBox.normalized() (0–1 coords) or to_pdf_points(dpi).
"""
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

from PIL import Image


# The 12 canonical region types (mapped from PP-DocLayoutV3's fine-grained labels).
RegionType = Literal[
    "title",
    "text",
    "table",
    "table_caption",
    "figure",
    "figure_caption",
    "formula",
    "chart",
    "header",
    "footer",
    "reference",
    "seal",
]


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in pixel coordinates. Origin is top-left of the page image."""
    x: float
    y: float
    width: float
    height: float

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2) — the shape most image libraries expect."""
        return (self.x, self.y, self.x2, self.y2)

    def normalized(self, page_width: int, page_height: int) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2) scaled to 0–1 relative to the page — resolution-independent.

        The shape UI overlays expect: multiply by the on-screen page size to place
        a highlight regardless of the DPI the page was rendered at.
        """
        return (
            self.x / page_width,
            self.y / page_height,
            self.x2 / page_width,
            self.y2 / page_height,
        )

    def to_pdf_points(self, dpi: int) -> "BoundingBox":
        """This box converted from rendered-image pixels to PDF points (72/inch)."""
        scale = 72.0 / dpi
        return BoundingBox(
            x=self.x * scale,
            y=self.y * scale,
            width=self.width * scale,
            height=self.height * scale,
        )

    def crop(self, image_bytes: bytes) -> bytes:
        """Return a PNG-encoded crop of image_bytes bounded by this box.

        Used to extract figure/chart images and to hand region patches to Nova.
        """
        img = Image.open(BytesIO(image_bytes))
        cropped = img.crop((int(self.x), int(self.y), int(self.x2), int(self.y2)))
        buf = BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()


@dataclass(frozen=True)
class Region:
    """One layout-detected region on a page.

    region_id — reading-order index on the page (0-based). Stable identifier for
                linking markdown/JSON output back to this region (hover-highlight UI).
    text      — plain OCR output, always populated for text-bearing regions.
    content   — structured output whose format depends on region_type:
                  table   → HTML
                  chart   → HTML data table
                  formula → LaTeX
                  text/title/caption/header/footer/reference → markdown (== text)
                  seal    → recognized text
                  figure  → empty (crop via bbox for downstream vision)
    """
    region_type: RegionType
    bbox: BoundingBox
    region_id: int = 0
    text: str = ""
    content: str = ""
    confidence: float = 1.0


@dataclass(frozen=True)
class LayoutResult:
    """Full-page OCR output — regions in reading order plus assembled text/markdown.

    Consumed by parse/pipeline.py to build a PageResult (which adds page_num,
    Nova enrichment, and final markdown assembly).

    page_width/page_height give bboxes a coordinate system — needed downstream when
    the source image_bytes has been discarded but bboxes are retained in ParseResult.
    """
    regions: list[Region]
    page_width: int
    page_height: int
    text: str = ""      # regions concatenated in reading order
    markdown: str = ""  # regions rendered as markdown with tables/formulas preserved
