import re
from bs4 import BeautifulSoup
from common.logging import get_logger

log       = get_logger(__name__)
LOC_SCALE = 500


def _parse_bbox(bbox_str: str) -> tuple[int, int, int, int]:
    parts = bbox_str.strip().split()
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def _loc(x1, y1, x2, y2, img_w, img_h) -> str:
    def norm(v, size):
        return max(0, min(499, round(v / size * LOC_SCALE)))

    return (
        f"<loc_{norm(x1, img_w)}>"
        f"<loc_{norm(y1, img_h)}>"
        f"<loc_{norm(x2, img_w)}>"
        f"<loc_{norm(y2, img_h)}>"
    )


def _clean(el) -> str:
    return el.get_text(separator=" ").strip()


def chandra_to_docling(chandra_html: str, img_w: int, img_h: int) -> str:
    soup  = BeautifulSoup(chandra_html, "html.parser")
    divs  = soup.find_all("div", attrs={"data-bbox": True, "data-label": True})
    lines = []

    for div in divs:
        label = div.get("data-label", "").strip()
        bbox  = div.get("data-bbox", "").strip()
        if not bbox:
            continue

        x1, y1, x2, y2 = _parse_bbox(bbox)
        loc = _loc(x1, y1, x2, y2, img_w, img_h)

        if label == "Text":
            lines.append(f"<text>{loc}{_clean(div)}</text>")

        elif label == "Section-Header":
            lines.append(f"<section_header_level_1>{loc}{_clean(div)}</section_header_level_1>")

        elif label == "Page-Header":
            lines.append(f"<page_header>{loc}{_clean(div)}</page_header>")

        elif label == "Page-Footer":
            lines.append(f"<page_footer>{loc}{_clean(div)}</page_footer>")

        elif label == "Caption":
            lines.append(f"<caption>{loc}{_clean(div)}</caption>")

        elif label == "Table":
            table_tag = div.find("table")
            content   = str(table_tag) if table_tag else _clean(div)
            lines.append(f"<table>{loc}{content}</table>")

        elif label == "List-Group":
            list_el = div.find("ul") or div.find("ol")
            if list_el:
                lines.append("<unordered_list>")
                for li in list_el.find_all("li"):
                    text = _clean(li)
                    text = re.sub(r"^[\d]+\.\s*", "", text).strip()
                    text = re.sub(r"^[●•\-\*]\s*", "", text).strip()
                    lines.append(f"<list_item>{loc}{text}</list_item>")
                lines.append("</unordered_list>")
            else:
                lines.append(f"<text>{loc}{_clean(div)}</text>")

        else:
            text = _clean(div)
            if text:
                lines.append(f"<text>{loc}{text}</text>")
                log.info(f"Unknown label '{label}' → text")

    result = "<doctag>\n" + "\n".join(lines) + "\n</doctag>"
    log.info(f"Converted {len(divs)} divs → doctags ({len(result)} chars)")
    return result
