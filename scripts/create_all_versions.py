"""
Create data versions sequentially until the full dataset is exhausted.
Loads the HF dataset once and passes slices to create_version() — avoids
re-downloading on every iteration.

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

from datasets import load_dataset
from src.data.data_versioning import create_version, get_next_offset


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

    # Resume from where previous run left off
    start_offset = get_next_offset(bucket)

    print(f"Loading dataset from {hf_repo} ...")
    all_samples = list(load_dataset(hf_repo, split="train", streaming=False))
    total_samples = len(all_samples)
    print(f"Loaded {total_samples:,} samples. Resuming from offset {start_offset:,}.\n")

    if start_offset >= total_samples:
        print("Dataset fully versioned. Nothing to do.")
        return

    version_num   = (start_offset // chunk_size) + 1
    total_created = 0

    for chunk_start in range(start_offset, total_samples, chunk_size):
        chunk = all_samples[chunk_start : chunk_start + chunk_size]
        version = f"{args.prefix}{version_num}"

        print(f"[{version}] offset={chunk_start:,}  samples={len(chunk):,} ...", end=" ", flush=True)
        metadata = create_version(
            version=version,
            samples=chunk,
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
