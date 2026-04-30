import uuid
from PIL import Image
from common.storage import upload_image, upload_json, upload_raw_html, write_local_count
from accumulation.dedup import is_duplicate
from common.logging import get_logger
from ingestion.converter import chandra_to_docling

log = get_logger(__name__)


def save_sample(image_path: str, chandra_html: str) -> bool:
    """
    Returns True if saved successfully, False if duplicate or error.
    Does NOT increment counter — caller handles batch increment.
    """
    if is_duplicate(image_path):
        log.info(f"Duplicate skipped: {image_path}")
        return False

    sample_id = uuid.uuid4().hex

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    doctags = chandra_to_docling(chandra_html, img_w, img_h)

    hf_image_path    = f"images/{sample_id}.png"
    hf_manifest_path = f"manifests/{sample_id}.jsonl"
    hf_raw_path      = f"chandra_raw/{sample_id}.html"

    upload_image(image_path, hf_image_path)
    upload_raw_html(chandra_html, hf_raw_path)
    upload_json({
        "image_path":  hf_image_path,
        "output_text": doctags,
        "chandra_raw": hf_raw_path,
        "source":      "colab-ingest",
        "img_w":       img_w,
        "img_h":       img_h,
    }, hf_manifest_path)

    write_local_count()
    log.info(f"Sample {sample_id} saved")
    return True    