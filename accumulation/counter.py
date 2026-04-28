import os
import threading
from dotenv import load_dotenv
from common.storage import download_json, upload_json
from common.logging import get_logger

load_dotenv()
log        = get_logger(__name__)
_lock      = threading.Lock()
_COUNT_FILE = "count.json"
_THRESHOLD  = int(os.environ.get("ACCUMULATION_THRESHOLD", 5000))


def get_count() -> int:
    try:
        return int(download_json(_COUNT_FILE).get("count", 0))
    except Exception:
        return 0


def increment(n: int = 1) -> int:
    with _lock:   # chỉ 1 thread được đọc-ghi count tại 1 thời điểm
        count = get_count() + n
        upload_json({"count": count}, _COUNT_FILE)
        log.info(f"Sample count → {count}")
        return count


def reset() -> None:
    with _lock:
        upload_json({"count": 0}, _COUNT_FILE)
        log.info("Sample count reset to 0")


def is_ready() -> bool:
    return get_count() >= _THRESHOLD