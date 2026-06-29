"""
Convert Chandra-OCR-2 *layout* HTML into granite-docling DocTags.

Adapted from `ingestion/converter.py` (branch main). Difference: that version was
written for the `chandra` Python package whose output bboxes are in pixels, so it
divided by img_w/img_h. Here we drive Chandra over the OpenAI-compatible endpoint
with `OCR_LAYOUT_PROMPT`, whose bboxes are **normalized 0-1000** — so we map
0-1000 → the DocTags 0-499 `<loc_>` grid directly, no image size needed.

`OCR_LAYOUT_PROMPT` / `PROMPT_ENDING` / `ALLOWED_*` are copied verbatim from
datalab-to/chandra (chandra/prompts.py) so the teacher emits the expected
`<div data-bbox=... data-label=...>` layout HTML.

`beautifulsoup4` is imported lazily inside `chandra_to_docling` so importing this
module (e.g. from the hash-backfill path) doesn't require bs4.
"""
import re

LOC_SCALE = 500  # granite-docling / SmolDocling <loc_> grid is 0..499

ALLOWED_TAGS = [
    "math", "br", "i", "b", "u", "del", "sup", "sub", "table", "tr", "td", "p",
    "th", "div", "pre", "h1", "h2", "h3", "h4", "h5", "ul", "ol", "li", "input",
    "a", "span", "img", "hr", "tbody", "small", "caption", "strong", "thead",
    "big", "code", "chem",
]
ALLOWED_ATTRIBUTES = [
    "class", "colspan", "rowspan", "display", "checked", "type", "border",
    "value", "style", "href", "alt", "align", "data-bbox", "data-label",
]

PROMPT_ENDING = f"""
Only use these tags {ALLOWED_TAGS}, and these attributes {ALLOWED_ATTRIBUTES}.

Guidelines:
* Inline math: Surround math with <math>...</math> tags. Math expressions should be rendered in KaTeX-compatible LaTeX. Use display for block math.
* Tables: Use colspan and rowspan attributes to match table structure.
* Formatting: Maintain consistent formatting with the image, including spacing, indentation, subscripts/superscripts, and special characters.
* Images: Include a description of any images in the alt attribute of an <img> tag. Do not fill out the src property. Describe in detail inside the div tag. Also convert charts to high fidelity data, and convert diagrams to mermaid.
* Forms: Mark checkboxes and radio buttons properly.
* Text: join lines together properly into paragraphs using <p>...</p> tags.  Use <br> tags for line breaks within paragraphs, but only when absolutely necessary to maintain meaning.
* Chemistry: Use <chem>...</chem> tags for chemical formulas with reactive SMILES.
* Lists: Preserve indents and proper list markers.
* Use the simplest possible HTML structure that accurately represents the content of the block.
* Make sure the text is accurate and easy for a human to read and interpret.  Reading order should be correct and natural.
""".strip()

OCR_LAYOUT_PROMPT = f"""
OCR this image to HTML, arranged as layout blocks.  Each layout block should be a div with the data-bbox attribute representing the bounding box of the block in x0 y0 x1 y1 format.  Bboxes are normalized 0-1000. The data-label attribute is the label for the block.

Use the following labels:
- Caption
- Footnote
- Equation-Block
- List-Group
- Page-Header
- Page-Footer
- Image
- Section-Header
- Table
- Text
- Complex-Block
- Code-Block
- Form
- Table-Of-Contents
- Figure
- Chemical-Block
- Diagram
- Bibliography
- Blank-Page

{PROMPT_ENDING}
""".strip()


def _loc(x1, y1, x2, y2) -> str:
    def n(v):  # bbox already normalized 0-1000 by OCR_LAYOUT_PROMPT
        return max(0, min(LOC_SCALE - 1, round(float(v) / 1000 * LOC_SCALE)))
    return f"<loc_{n(x1)}><loc_{n(y1)}><loc_{n(x2)}><loc_{n(y2)}>"


def _parse_bbox(bbox_str: str):
    parts = bbox_str.replace(",", " ").split()
    if len(parts) < 4:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None


def chandra_to_docling(chandra_html: str) -> str:
    """Chandra layout HTML (data-bbox/data-label divs, bbox 0-1000) → DocTags."""
    from bs4 import BeautifulSoup  # lazy: only needed when actually converting

    def clean(el) -> str:
        return el.get_text(separator=" ").strip()

    soup = BeautifulSoup(chandra_html, "html.parser")
    divs = soup.find_all("div", attrs={"data-bbox": True})
    lines: list[str] = []

    for div in divs:
        bbox = _parse_bbox(div.get("data-bbox", ""))
        if bbox is None:
            continue
        loc = _loc(*bbox)
        label = (div.get("data-label") or "").strip()

        if label == "Section-Header":
            lines.append(f"<section_header_level_1>{loc}{clean(div)}</section_header_level_1>")
        elif label == "Page-Header":
            lines.append(f"<page_header>{loc}{clean(div)}</page_header>")
        elif label == "Page-Footer":
            lines.append(f"<page_footer>{loc}{clean(div)}</page_footer>")
        elif label == "Caption":
            lines.append(f"<caption>{loc}{clean(div)}</caption>")
        elif label == "Table":
            table_tag = div.find("table")
            content = str(table_tag) if table_tag else clean(div)
            lines.append(f"<table>{loc}{content}</table>")
        elif label in ("Image", "Figure", "Diagram"):
            img_tag = div.find("img")
            alt = (img_tag.get("alt", "").strip() if img_tag else "") or clean(div)
            lines.append(f"<picture>{loc}{alt}</picture>")
        elif label == "List-Group":
            list_el = div.find("ul") or div.find("ol")
            if list_el:
                lines.append("<unordered_list>")
                for li in list_el.find_all("li"):
                    text = clean(li)
                    text = re.sub(r"^[\d]+\.\s*", "", text).strip()
                    text = re.sub(r"^[●•\-\*]\s*", "", text).strip()
                    lines.append(f"<list_item>{loc}{text}</list_item>")
                lines.append("</unordered_list>")
            else:
                lines.append(f"<text>{loc}{clean(div)}</text>")
        else:  # Text, Footnote, Code-Block, Form, Bibliography, … → text
            text = clean(div)
            if text:
                lines.append(f"<text>{loc}{text}</text>")

    return "<doctag>\n" + "\n".join(lines) + "\n</doctag>"