"""
Create data versions sequentially until the full dataset is exhausted.

Uses cast_column(decode=False) to keep images as raw bytes instead of
decoding to PIL — avoids the massive slowdown from image decoding.

Usage:
    python scripts/create_all_versions.py
    python scripts/create_all_versions.py --samples-per-version 10000
    python scripts/create_all_versions.py --samples-per-version 5000 --prefix v
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import load_dataset, Image as HFImage
from tqdm import tqdm
from src.data.data_versioning import create_version, get_next_offset


def _extract_bytes(sample: dict) -> dict:
    """Normalize HF image field: {"bytes": ..., "path": ...} → raw bytes."""
    img = sample.get("image")
    if isinstance(img, dict):
        sample = dict(sample)
        sample["image"] = img.get("bytes")
    return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples-per-version", type=int, default=5000,
        help="Raw samples per version (default: 5000)",
    )
    parser.add_argument(
        "--prefix", type=str, default="v",
        help="Version name prefix, e.g. 'v' → v1, v2, ... (default: v)",
    )
    args = parser.parse_args()

    bucket     = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")
    hf_repo    = os.environ.get("HF_REPO_DATA", "nbdaaa/all-ocr-data")
    chunk_size = args.samples_per_version

    start_offset = get_next_offset(bucket)

    print(f"Loading dataset index from {hf_repo} ...", flush=True)
    ds = load_dataset(hf_repo, split="train", streaming=False)
    print("Disabling PIL decoding ...", flush=True)
    ds = ds.cast_column("image", HFImage(decode=False))
    total_samples = len(ds)
    print(f"Dataset has {total_samples:,} samples. Resuming from offset {start_offset:,}.\n")

    if start_offset >= total_samples:
        print("Dataset fully versioned. Nothing to do.")
        return

    version_num   = (start_offset // chunk_size) + 1
    total_created = 0

    for chunk_start in range(start_offset, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        version   = f"{args.prefix}{version_num}"

        print(f"\n[{version}] offset={chunk_start:,}  samples={chunk_end - chunk_start:,}", flush=True)

        chunk_samples = [
            _extract_bytes(s)
            for s in tqdm(
                ds.select(range(chunk_start, chunk_end)),
                total=chunk_end - chunk_start,
                desc="  reading",
            )
        ]
        print("  filtering + dedup + upload ...", flush=True)

        metadata = create_version(
            version=version,
            samples=chunk_samples,
            offset=chunk_start,
        )
        print(
            f"✓  final_count={metadata['filter_stats']['final_count']:,}"
            f"  run_id={metadata['mlflow_run_id'][:8]}..."
        )

        total_created += 1
        version_num   += 1

    print(f"\nDone — {total_created} version(s) created.")


if __name__ == "__main__":
    main()
