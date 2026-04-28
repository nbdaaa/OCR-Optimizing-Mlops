import os
from dotenv import load_dotenv

load_dotenv()

endpoint = os.environ.get("CHANDRA_ENDPOINT", "")
if not endpoint:
    raise RuntimeError(
        "CHANDRA_ENDPOINT is not set. "
        "Start the Colab notebook and paste the ngrok URL into .env"
    )

os.environ["VLLM_API_BASE"]   = f"{endpoint.rstrip('/')}/v1"
os.environ["VLLM_MODEL_NAME"] = os.environ.get(
    "CHANDRA_MODEL", "datalab-to/chandra-ocr-2"
)

from PIL import Image
from chandra.model import InferenceManager
from chandra.model.schema import BatchInputItem
from common.logging import get_logger
from common.exceptions import ChandraBaseError

log      = get_logger(__name__)
_manager = None


def _get_manager() -> InferenceManager:
    global _manager
    if _manager is None:
        _manager = InferenceManager(method="vllm")
    return _manager


def call_chandra(image_path: str) -> str:
    """
    Send one image to Chandra OCR-2 via the Colab vLLM endpoint.
    Returns raw HTML output.
    """
    try:
        manager = _get_manager()
        image   = Image.open(image_path).convert("RGB")
        batch   = [BatchInputItem(image=image, prompt_type="ocr_layout")]
        result  = manager.generate(batch)[0]
        log.info(f"Chandra returned output for {image_path}")
        return result.raw
    except Exception as e:
        raise ChandraBaseError(f"Chandra inference failed: {e}") from e