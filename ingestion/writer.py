import uuid
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from common.storage import upload_image, upload_json
from accumulation.counter import increment
from common.logging import get_logger

log        = get_logger(__name__)
_converter = DocumentConverter()


def save_sample(image_path: str, chandra_html: str) -> int:
    """
    Convert Chandra HTML → doctags via Docling.
    Save image + doctags to HuggingFace and increment counter.
    Returns the new total sample count.
    """
    sample_id = uuid.uuid4().hex

    # Chandra HTML → DoclingDocument → doctags
    result  = _converter.convert_string(chandra_html, format=InputFormat.HTML)
    doctags = result.document.export_to_doctags()

    hf_image_path    = f"images/{sample_id}.png"
    hf_manifest_path = f"manifests/{sample_id}.jsonl"

    log.info(f"Uploading image → {hf_image_path}")
    upload_image(image_path, hf_image_path)

    log.info(f"Uploading manifest → {hf_manifest_path}")
    upload_json({
        "image_path":  hf_image_path,
        "output_text": doctags,
        "source":      "colab-ingest",
    }, hf_manifest_path)

    new_count = increment()
    log.info(f"Sample saved. Total count: {new_count}")
    return new_count
