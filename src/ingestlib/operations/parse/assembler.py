"""Assemble a page's final markdown and plain text from its regions.

Pure Python — regions render in reading order, each into a markdown block, so
every block in the output is traceable to its region_id (the anchor the
hover-highlight UI maps back to a bbox).
"""
from ingestlib.foundations.ocr.models import Region
from ingestlib.operations.parse.models import FigureImage


# Page furniture — kept on the regions list but left out of markdown AND text:
# repeated per-page headers/footers are pure noise in RAG chunks.
_PAGE_FURNITURE = ("header", "footer")


def _render_visual(region: Region, figure: FigureImage | None, page_num: int) -> str:
    """Image reference (matching save_images filenames) + the region's content.

    Chart content (a markdown data table) renders as-is below the image link;
    figure content (a description) renders blockquoted.
    """
    caption = figure.caption if figure else ""
    filename = (
        figure.filename(page_num)
        if figure
        else f"page{page_num}_region{region.region_id}_{region.region_type}.png"
    )
    lines = [f"![{caption or region.region_type}]({filename})"]
    content = (figure.description if figure else region.content).strip()
    if content:
        if region.region_type == "chart" and "|" in content:
            lines.append("")
            lines.append(content)
        else:
            lines.extend(f"> {ln}" for ln in content.splitlines())
    return "\n".join(lines)


def _render_formula(content: str) -> str:
    """Wrap bare LaTeX in a display block; leave already-delimited formulas alone."""
    stripped = content.strip()
    if stripped.startswith("$"):
        return stripped
    return f"$$\n{stripped}\n$$"


def render_region(
    region: Region,
    figures_by_id: dict[int, FigureImage],
    page_num: int,
) -> str:
    """One region → one markdown block. Empty string means 'skip this region'."""
    rtype = region.region_type
    if rtype in _PAGE_FURNITURE:
        return ""
    if rtype == "title":
        # Layout titles can span multiple visual lines — a heading must be one line.
        title = " ".join(region.text.split())
        return f"## {title}" if title else ""
    if rtype == "table":
        return region.content.strip()
    if rtype in ("chart", "figure"):
        return _render_visual(region, figures_by_id.get(region.region_id), page_num)
    if rtype == "formula":
        return _render_formula(region.content) if region.content.strip() else ""
    if rtype in ("figure_caption", "table_caption"):
        return f"*{region.text.strip()}*" if region.text.strip() else ""
    # text, reference, seal
    return region.text.strip()


def assemble_markdown(
    regions: list[Region],
    figures: list[FigureImage],
    page_num: int,
) -> str:
    """Render regions in reading order into the page's markdown."""
    figures_by_id = {f.region_id: f for f in figures}
    blocks = [render_region(r, figures_by_id, page_num) for r in regions]
    return "\n\n".join(b for b in blocks if b)


def assemble_text(regions: list[Region]) -> str:
    """Plain-text form of the page — one line per text-bearing region, reading order.

    Headers/footers are excluded (same as markdown); they remain on the regions list.
    """
    return "\n".join(
        r.text for r in regions if r.text and r.region_type not in _PAGE_FURNITURE
    )
