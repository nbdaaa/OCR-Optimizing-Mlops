"""
Create a fixed CER benchmark set by copying N samples from an existing data
version into ocr-data/benchmark/.

The benchmark is the held-out set the CI gate evaluates CER on. Pick a source
version that will NOT be used for training (e.g. v19), then delete that version
afterwards so its samples never leak into training:

    python scripts/create_benchmark.py --from-version v19 --count 100
    python scripts/delete_version.py v19
"""
import argparse
import io
import json
import os
import sys
import tempfile

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_versioning import upload_to_minio, write_parquet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-version", required=True, help="Source version, e.g. v19")
    parser.add_argument("--count", type=int, default=100, help="Samples to take (default 100)")
    args = parser.parse_args()

    bucket = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )

    print(f"Reading {args.from_version}/dataset.parquet ...", flush=True)
    resp = client.get_object(Bucket=bucket, Key=f"{args.from_version}/dataset.parquet")
    df = pd.read_parquet(io.BytesIO(resp["Body"].read()))

    bench = df.head(args.count)
    samples = bench.to_dict("records")
    print(f"Selected {len(samples)} samples for benchmark.", flush=True)

    metadata = {
        "version": "benchmark",
        "count": len(samples),
        "source_version": args.from_version,
        "note": "Held-out CER benchmark — never used for training.",
    }

    with tempfile.TemporaryDirectory() as tmp:
        write_parquet(samples, os.path.join(tmp, "dataset.parquet"))
        with open(os.path.join(tmp, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        upload_to_minio(tmp, "benchmark")

    print(f"Done — benchmark set written to {bucket}/benchmark/", flush=True)
    print(f"Now delete the source version so it isn't trained on:", flush=True)
    print(f"  python scripts/delete_version.py {args.from_version}", flush=True)


if __name__ == "__main__":
    main()
