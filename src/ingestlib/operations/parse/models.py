"""Data models returned by parse(): FigureImage, PageResult, and ParseResult.

All are Pydantic v2 frozen models. Region (from ocr/models.py) embeds via
`arbitrary_types_allowed=True`.
"""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from ingestlib.foundations.ocr.models import Region


SourceFormat = Literal["pdf", "docx", "pptx"]


class FigureImage(BaseModel):
    """One visual region extracted from a page as an image.

    region_id    — reading-order index of the source region on its page
    region_type  — "figure" | "chart"
    image_bytes  — PNG crop of exactly this region from the rendered page
    caption      — nearest caption region's text, "" when none was found
    description  — Nova's interpretation: a data table for charts, a structured
                   description for figures/diagrams
    """

    model_config = ConfigDict(frozen=True)

    region_id: int
    region_type: str
    image_bytes: bytes
    caption: str = ""
    description: str = ""

    def filename(self, page_num: int) -> str:
        """Canonical export name — matches the references in PageResult.markdown."""
        return f"page{page_num}_region{self.region_id}_{self.region_type}.png"


class PageResult(BaseModel):
    """One parsed page.

    text         — plain text of the page
    markdown     — final markdown (tables as HTML, formulas as LaTeX, charts as
                   data tables, figures as image references + descriptions)
    regions      — layout regions in reading order, with bboxes and region_ids;
                   chart/figure content is Nova-enriched
    figures      — extracted visual regions (chart/figure) as PNG crops with
                   captions and descriptions
    native_text  — original text-layer content from the source document
    image_bytes  — full page rendered at image_dpi
    page_width   — image width in pixels
    page_height  — image height in pixels
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    page_num: int = Field(..., ge=1, description="1-indexed page number")
    text: str = ""
    markdown: str = ""
    regions: list[Region] = Field(default_factory=list)
    figures: list[FigureImage] = Field(default_factory=list)

    native_text: str = ""
    image_bytes: bytes | None = None

    image_format: Literal["png", "jpeg"] = "png"
    image_dpi: int = Field(default=200, ge=72)

    page_width: int | None = None
    page_height: int | None = None

    @computed_field
    @property
    def has_native_text(self) -> bool:
        """True when the source document supplied its own text layer for this page."""
        return bool(self.native_text.strip())

    @computed_field
    @property
    def word_count(self) -> int:
        """Whitespace-split word count of `text`."""
        return len(self.text.split()) if self.text else 0

    def region_by_id(self, region_id: int) -> Region:
        """Fetch a region by its region_id. Raises IndexError if absent."""
        for r in self.regions:
            if r.region_id == region_id:
                return r
        raise IndexError(f"region_id={region_id} not found on page {self.page_num}")


class ParseResult(BaseModel):
    """Full parse output — the foundation object every downstream operation consumes.

    pages             — list of PageResult in document order
    source_path       — path of the file that was parsed
    source_format     — pdf | docx | pptx
    was_converted     — True when the source was a DOCX/PPTX routed through
                        LibreOffice before parsing
    source_metadata   — properties extracted from the source file (title, author,
                        subject, etc.); keys depend on the source format
    source_checksum   — SHA256 hex digest of the source file bytes
    created_at        — UTC timestamp of when the parse completed
    parse_duration_seconds — wall-clock time the parse took
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    pages: list[PageResult]
    source_path: Path
    source_format: SourceFormat

    was_converted: bool = False
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    source_checksum: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parse_duration_seconds: float | None = None

    @computed_field
    @property
    def page_count(self) -> int:
        return len(self.pages)

    @computed_field
    @property
    def total_word_count(self) -> int:
        return sum(p.word_count for p in self.pages)

    @computed_field
    @property
    def markdown(self) -> str:
        """Whole-document markdown — pages joined in order."""
        return "\n\n".join(p.markdown for p in self.pages if p.markdown)

    def page_by_num(self, page_num: int) -> PageResult:
        """Fetch a page by its 1-indexed page number. Raises IndexError if absent."""
        for p in self.pages:
            if p.page_num == page_num:
                return p
        raise IndexError(
            f"page_num={page_num} not found in ParseResult (have {self.page_count} pages)"
        )

    def save_images(self, directory: Path | str) -> list[Path]:
        """Write every extracted figure/chart image to `directory` as PNG files.

        Filenames match the image references inside PageResult.markdown
        (page{N}_region{K}_{type}.png). Returns the written paths.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for page in self.pages:
            for fig in page.figures:
                out = directory / fig.filename(page.page_num)
                out.write_bytes(fig.image_bytes)
                written.append(out)
        return written
