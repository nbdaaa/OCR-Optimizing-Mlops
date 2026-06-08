"""
Data versioning pipeline: HF Hub → filter → dedup → MinIO + MLflow.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import boto3
import imagehash
import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────

# Rows written per parquet row-group. Smaller batches cap peak memory during
# writing (we never build the full arrow table at once), at the cost of slightly
# less compression. Tuned for small-RAM VMs.
PARQUET_BATCH_SIZE = int(os.environ.get("PARQUET_BATCH_SIZE", "200"))


# ── Validation ────────────────────────────────────────────────────────────────

def is_valid(sample: dict) -> bool:
    """Return True if sample passes all quality filters."""
    if sample.get("image") is None:
        return False

    text = sample.get("output_text", "")
    if not text or not text.strip():
        return False

    if text.strip() in ("<doctag>\n</doctag>", "<doctag></doctag>"):
        return False

    if "<loc_" not in text:
        return False

    if "<doctag>" not in text or "</doctag>" not in text:
        return False

    return True


# ── Metadata ──────────────────────────────────────────────────────────────────

def build_metadata(
    version: str,
    count: int,
    hf_repo: str,
    filter_stats: dict,
    split: str,
    offset: int,
) -> dict:
    """Build metadata dict for a data version (stored alongside dataset.parquet)."""
    return {
        "version": version,
        "count": count,
        "hf_repo": hf_repo,
        "filter_stats": filter_stats,
        "split": split,
        "offset": offset,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Deduplication ─────────────────────────────────────────────────────────────

def _to_pil(image: Any) -> Image.Image:
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return image.convert("RGB") if image.mode != "RGB" else image


def compute_image_hash(image: Any) -> str:
    """Return MD5 hex digest of image bytes. Accepts bytes or PIL.Image."""
    if isinstance(image, bytes):
        data = image
    else:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
    return hashlib.md5(data).hexdigest()


def compute_phash(image: Any) -> imagehash.ImageHash:
    """Return 64-bit perceptual hash (imagehash.ImageHash) of image."""
    return imagehash.phash(_to_pil(image))


def exact_dedup(samples: list[dict]) -> tuple[list[dict], dict]:
    """
    Remove exact duplicate images (same MD5 hash).
    When duplicates exist, keep the sample with the longest output_text.

    Returns:
        (deduped_samples, stats) where stats = {exact_removed, exact_groups}
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        h = compute_image_hash(sample["image"])
        groups[h].append(sample)

    result = []
    exact_removed = 0
    exact_groups = 0

    for group in groups.values():
        if len(group) > 1:
            exact_groups += 1
            exact_removed += len(group) - 1
            best = max(group, key=lambda s: len(s.get("output_text", "")))
        else:
            best = group[0]
        result.append(best)

    return result, {"exact_removed": exact_removed, "exact_groups": exact_groups}


def phash_dedup(samples: list[dict], threshold: int = 8) -> tuple[list[dict], dict]:
    """
    Remove near-duplicate images (pHash hamming distance <= threshold).
    When near-duplicates exist, keep the sample with the highest resolution
    (img_w * img_h).

    Returns:
        (deduped_samples, stats) where stats = {phash_removed, phash_groups, threshold}
    """
    if not samples:
        return [], {"phash_removed": 0, "phash_groups": 0, "threshold": threshold}

    n = len(samples)
    hashes = [
        compute_phash(s["image"])
        for s in tqdm(samples, desc="    computing pHash", leave=False)
    ]

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # O(n²) — acceptable for offline versioning workload
    for i in tqdm(range(n), desc="    comparing pairs", leave=False):
        for j in range(i + 1, n):
            if (hashes[i] - hashes[j]) <= threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    result = []
    phash_removed = 0
    phash_groups = 0

    for indices in groups.values():
        if len(indices) > 1:
            phash_groups += 1
            phash_removed += len(indices) - 1
            best = max(indices, key=lambda i: samples[i].get("img_w", 0) * samples[i].get("img_h", 0))
        else:
            best = indices[0]
        result.append(samples[best])

    return result, {"phash_removed": phash_removed, "phash_groups": phash_groups, "threshold": threshold}


# ── Offset ───────────────────────────────────────────────────────────────────

def get_next_offset(bucket: str) -> int:
    """
    Sum filter_stats.total across all existing versions in MinIO.
    This equals the number of raw HF samples already consumed, which is
    the correct starting offset for the next version.
    Returns 0 if no versions exist yet.
    """
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    try:
        response = client.list_objects_v2(Bucket=bucket, Delimiter="/")
        prefixes = response.get("CommonPrefixes", [])
    except Exception:
        return 0

    total_consumed = 0
    for prefix in prefixes:
        try:
            obj = client.get_object(Bucket=bucket, Key=f"{prefix['Prefix']}metadata.json")
            meta = json.loads(obj["Body"].read())
            total_consumed += meta.get("filter_stats", {}).get("total", 0)
        except Exception:
            continue

    return total_consumed


# ── Parquet writer ────────────────────────────────────────────────────────────

def _image_to_bytes(img: Any) -> bytes | None:
    """Coerce an image value (bytes, PIL.Image, or HF dict) to raw bytes."""
    if img is None:
        return None
    if isinstance(img, bytes):
        return img
    if isinstance(img, dict):
        return img.get("bytes")
    # PIL.Image
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _batch_to_table(batch: list[dict], keys: list[str]) -> pa.Table:
    """
    Build an arrow Table from one batch of samples.

    The 'image' column is declared as pa.binary() so pyarrow uses a fast
    zero-inference path for the raw image bytes (and normalizes bytes/PIL/HF
    dict to bytes). Other columns are inferred.
    """
    arrays = []
    names = []
    for k in keys:
        col = [s.get(k) for s in batch]
        if k == "image":
            arrays.append(pa.array([_image_to_bytes(v) for v in col], type=pa.binary()))
        else:
            arrays.append(pa.array(col))
        names.append(k)
    return pa.Table.from_arrays(arrays, names=names)


def write_parquet(samples: list[dict], path: str) -> None:
    """
    Write samples to parquet in row-group batches of PARQUET_BATCH_SIZE.

    Writing batch-by-batch with pq.ParquetWriter caps peak memory: we only ever
    materialize one small arrow table (~PARQUET_BATCH_SIZE rows) at a time
    instead of the entire dataset's image bytes — critical on small-RAM VMs.
    """
    if not samples:
        pq.write_table(pa.table({}), path)
        return

    keys: list[str] = list(samples[0].keys())
    writer = None
    try:
        n_batches = (len(samples) + PARQUET_BATCH_SIZE - 1) // PARQUET_BATCH_SIZE
        for start in tqdm(
            range(0, len(samples), PARQUET_BATCH_SIZE),
            total=n_batches,
            desc="        writing batches",
            leave=False,
        ):
            batch = samples[start : start + PARQUET_BATCH_SIZE]
            table = _batch_to_table(batch, keys)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


# ── Storage ───────────────────────────────────────────────────────────────────

def upload_to_minio(local_dir: str, version: str) -> None:
    """Upload dataset.parquet and metadata.json to MinIO ocr-data/{version}/."""
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )
    bucket = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")
    for filename in ("dataset.parquet", "metadata.json"):
        client.upload_file(os.path.join(local_dir, filename), bucket, f"{version}/{filename}")


def log_to_mlflow(version: str, metadata: dict, local_metadata_json: str) -> str:
    """
    Start an MLflow run, log params + log_input + the small metadata.json.

    Note: the dataset parquet itself lives in the ocr-data bucket (uploaded by
    upload_to_minio). We deliberately do NOT re-upload it as an MLflow artifact
    to avoid duplicating hundreds of MB into the mlflow-artifacts bucket.
    Only metadata.json is logged here for lineage tracking.

    Returns the MLflow run_id.
    """
    # MLflow's internal boto3 reads AWS_* vars for MinIO access
    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", ""))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", ""))

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("data-versioning")

    with mlflow.start_run(run_name=f"data-{version}") as run:
        mlflow.log_params({
            "version": version,
            "hf_repo": metadata["hf_repo"],
            "split": metadata["split"],
            "count": metadata["count"],
            **metadata["filter_stats"],
        })
        mlflow.log_artifact(local_metadata_json, artifact_path=version)
        mlflow.log_input(
            mlflow.data.from_pandas(pd.DataFrame([metadata]), source=metadata["hf_repo"]),
            context="data_versioning",
        )

    return run.info.run_id


# ── Pipeline ──────────────────────────────────────────────────────────────────

def create_version(
    version: str,
    max_samples: int | None = None,
    samples: list | None = None,
    offset: int | None = None,
) -> dict:
    """
    Full pipeline: HF Hub → filter → exact_dedup → phash_dedup → MinIO + MLflow.

    Args:
        version:     Version name, e.g. "v1".
        max_samples: Max raw samples to take from HF (ignored when samples provided).
        samples:     Pre-loaded sample list. When provided, skips HF download and
                     offset calculation entirely — caller is responsible for slicing.
                     Useful when loading the full dataset once and iterating in a loop.

    Steps:
        1. Load samples from HF Hub (nbdaaa/all-ocr-data) — skipped if samples given
        2. Filter with is_valid()
        3. exact_dedup()
        4. phash_dedup()
        5. Save dataset.parquet locally
        6. build_metadata() with combined filter_stats
        7. upload_to_minio()
        8. log_to_mlflow()

    Returns metadata dict (includes mlflow_run_id).
    """
    hf_repo = os.environ.get("HF_REPO_DATA", "nbdaaa/all-ocr-data")
    bucket  = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")

    if samples is not None:
        # Pre-loaded path: caller sliced the dataset and tracks offset externally.
        offset = offset if offset is not None else 0
        total  = len(samples)
    else:
        # Auto path: load from HF and calculate offset from existing versions.
        offset      = get_next_offset(bucket)
        all_samples = list(load_dataset(hf_repo, split="train", streaming=False))
        samples     = all_samples[offset:]
        if max_samples:
            samples = samples[:max_samples]
        total = len(samples)

    print(f"  [1/5] filtering invalid samples ({total:,} total) ...", flush=True)
    samples = [s for s in samples if is_valid(s)]
    rejected_invalid = total - len(samples)
    print(f"        → {len(samples):,} valid  ({rejected_invalid:,} rejected)", flush=True)

    print(f"  [2/5] exact dedup ...", flush=True)
    samples, exact_stats = exact_dedup(samples)
    print(f"        → {len(samples):,} remain  ({exact_stats['exact_removed']} removed)", flush=True)

    print(f"  [3/5] phash dedup ({len(samples):,} samples, O(n²)) ...", flush=True)
    samples, phash_stats = phash_dedup(samples)
    print(f"        → {len(samples):,} remain  ({phash_stats['phash_removed']} removed)", flush=True)

    filter_stats = {
        "total": total,
        "rejected_invalid": rejected_invalid,
        "rejected_exact_dup": exact_stats["exact_removed"],
        "rejected_phash_dup": phash_stats["phash_removed"],
        "final_count": len(samples),
    }

    metadata = build_metadata(
        version=version,
        count=len(samples),
        hf_repo=hf_repo,
        filter_stats=filter_stats,
        split="train",
        offset=offset,
    )

    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = os.path.join(tmp, "dataset.parquet")
        meta_path = os.path.join(tmp, "metadata.json")

        print(f"  [4/5] writing parquet ({len(samples):,} rows, pyarrow) ...", flush=True)
        write_parquet(samples, parquet_path)
        size_mb = os.path.getsize(parquet_path) / 1024 / 1024
        print(f"        → {size_mb:.1f} MB", flush=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  [5/5] uploading to MinIO ...", flush=True)
        upload_to_minio(tmp, version)
        print(f"        logging to MLflow ...", flush=True)
        run_id = log_to_mlflow(version, metadata, meta_path)

    metadata["mlflow_run_id"] = run_id
    return metadata
