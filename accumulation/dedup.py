import os
import threading
import imagehash
from PIL import Image
from common.storage import download_json, upload_json
from common.logging import get_logger

log        = get_logger(__name__)
_HASH_FILE = "hash_index.json"
_THRESHOLD = 8
_lock      = threading.Lock()   # ← prevent concurrent read/write


def _load_seen_hashes() -> dict:
    try:
        return download_json(_HASH_FILE)
    except Exception:
        return {}


def _save_seen_hashes(hashes: dict) -> None:
    upload_json(hashes, _HASH_FILE)


def is_duplicate(image_path: str) -> bool:
    """
    Thread-safe duplicate check.
    Reads hash index, checks for near-duplicate, registers if unique.
    """
    with _lock:   # ← only one thread checks/writes at a time
        try:
            img  = Image.open(image_path).convert("RGB")
            h    = str(imagehash.phash(img))
            seen = _load_seen_hashes()

            for existing_hash in seen.values():
                if imagehash.hex_to_hash(h) - imagehash.hex_to_hash(existing_hash) < _THRESHOLD:
                    return True   # duplicate

            # not a duplicate — register it
            seen[image_path] = h
            _save_seen_hashes(seen)
            return False

        except Exception as e:
            log.info(f"Dedup check failed for {image_path}: {e} — allowing")
            return False


def filter_duplicates(samples: list[dict]) -> list[dict]:
    """Batch dedup for dataset_builder — runs single-threaded."""
    seen   = _load_seen_hashes()
    unique = []

    for sample in samples:
        try:
            img  = Image.open(sample["image_path"]).convert("RGB")
            h    = str(imagehash.phash(img))
            is_dup = any(
                imagehash.hex_to_hash(h) - imagehash.hex_to_hash(existing) < _THRESHOLD
                for existing in seen.values()
            )
            if not is_dup:
                seen[sample["image_path"]] = h
                unique.append(sample)
            else:
                log.info(f"Duplicate skipped: {sample['image_path']}")
        except Exception as e:
            log.info(f"Could not hash {sample['image_path']}: {e} — keeping")
            unique.append(sample)

    _save_seen_hashes(seen)
    log.info(f"Dedup: {len(samples)} in → {len(unique)} unique")
    return unique