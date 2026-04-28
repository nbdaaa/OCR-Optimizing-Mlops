# ingestion/writer.py
import uuid
from PIL import Image
from common.storage import upload_image, upload_json
from accumulation.counter import increment
from accumulation.dedup import is_duplicate   # ← add this
from common.logging import get_logger
from ingestion.converter import chandra_to_docling

log = get_logger(__name__)


def save_sample(image_path: str, chandra_html: str) -> int | None:
    """
    Convert Chandra HTML → doctags, check for duplicates, save to HF.
    Returns new count if saved, None if duplicate.
    """
    # dedup check before doing anything
    if is_duplicate(image_path):
        log.info(f"Duplicate skipped: {image_path}")
        return None

    sample_id = uuid.uuid4().hex

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    doctags = chandra_to_docling(chandra_html, img_w, img_h)

    hf_image_path    = f"images/{sample_id}.png"
    hf_manifest_path = f"manifests/{sample_id}.jsonl"

    upload_image(image_path, hf_image_path)
    upload_json({
        "image_path":  hf_image_path,
        "output_text": doctags,
        "source":      "colab-ingest",
    }, hf_manifest_path)

    new_count = increment()
    log.info(f"Sample saved. Total count: {new_count}")
    return new_count