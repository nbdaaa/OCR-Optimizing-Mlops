import os
import threading
from dotenv import load_dotenv
from common.storage import get_local_sample_count
from common.logging import get_logger

load_dotenv()
log        = get_logger(__name__)
_lock      = threading.Lock()
_THRESHOLD = int(os.environ.get("ACCUMULATION_THRESHOLD", 5000))


def get_count() -> int:
    """Real count based on files in ./data/repo/chandra_raw/"""
    return get_local_sample_count()


def increment(n: int = 1) -> int:
    """No-op — count is derived from real files, not a counter."""
    return get_count()


def is_ready() -> bool:
    return get_count() >= _THRESHOLD


def reset() -> None:
    """Reset is handled by clear_local_repo() after push."""
    log.info("Counter reset — handled by clear_local_repo()")