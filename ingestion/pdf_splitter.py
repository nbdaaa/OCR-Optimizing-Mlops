import uuid
import fitz
from common.logging import get_logger

log = get_logger(__name__)


def pdf_to_images(pdf_path: str, dpi: int = 150) -> list[str]:
    """
    Split a PDF into per-page PNG images.
    Returns list of temp file paths.
    """
    doc   = fitz.open(pdf_path)
    mat   = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []

    for i, page in enumerate(doc):
        pix  = page.get_pixmap(matrix=mat)
        path = f"/tmp/page_{i:04d}_{uuid.uuid4().hex[:6]}.png"
        pix.save(path)
        paths.append(path)
        log.info(f"Rendered page {i+1}/{len(doc)} → {path}")

    doc.close()
    log.info(f"PDF split into {len(paths)} page image(s)")
    return paths
