import os
import json
import tempfile
from pathlib import Path
from PIL import Image
import imagehash
from common.storage import download_json, upload_json
from common.logging import get_logger

log = get_logger(__name__)

_HASH_FILE = "hash_index.json"
_THRESHOLD = 8   # hamming distance — lower = stricter dedup


def _load_seen_hashes() -> dict:
    try:
        return download_json(_HASH_FILE)
    except Exception:
        return {}


def _save_seen_hashes(hashes: dict) -> None:
    upload_json(hashes, _HASH_FILE)


def filter_duplicates(samples: list[dict]) -> list[dict]:
    """
    samples: list of dicts with at least {"image_path": str, ...}
    Returns only unique samples. Updates hash index in HF.
    """
    seen = _load_seen_hashes()
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
            log.info(f"Could not hash {sample['image_path']}: {e} — keeping sample")
            unique.append(sample)

    _save_seen_hashes(seen)
    log.info(f"Dedup: {len(samples)} in → {len(unique)} unique")
    return unique