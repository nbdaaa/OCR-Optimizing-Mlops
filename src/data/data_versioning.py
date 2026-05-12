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
from datasets import load_dataset
from PIL import Image


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
) -> dict:
    """Build metadata dict for a data version (stored alongside dataset.parquet)."""
    return {
        "version": version,
        "count": count,
        "hf_repo": hf_repo,
        "filter_stats": filter_stats,
        "split": split,
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
    hashes = [compute_phash(s["image"]) for s in samples]

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
    for i in range(n):
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


def log_to_mlflow(version: str, metadata: dict, local_parquet: str) -> str:
    """
    Start an MLflow run, log params + log_input + artifact.
    Returns the MLflow run_id.
    """
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
        mlflow.log_artifact(local_parquet, artifact_path=version)
        mlflow.log_input(
            mlflow.data.from_pandas(pd.DataFrame([metadata]), source=metadata["hf_repo"]),
            context="data_versioning",
        )

    return run.info.run_id


# ── Pipeline ──────────────────────────────────────────────────────────────────

def create_version(version: str, max_samples: int | None = None) -> dict:
    """
    Full pipeline: HF Hub → filter → exact_dedup → phash_dedup → MinIO + MLflow.

    Steps:
        1. Load samples from HF Hub (nbdaaa/all-ocr-data)
        2. Filter with is_valid()
        3. exact_dedup()
        4. phash_dedup()
        5. Save dataset.parquet locally
        6. build_metadata() with combined filter_stats
        7. upload_to_minio()
        8. log_to_mlflow()

    Returns metadata dict.
    """
    hf_repo = os.environ.get("HF_REPO_DATA", "nbdaaa/all-ocr-data")

    samples = list(load_dataset(hf_repo, split="train", streaming=False))
    if max_samples:
        samples = samples[:max_samples]
    total = len(samples)

    samples = [s for s in samples if is_valid(s)]
    rejected_invalid = total - len(samples)

    samples, exact_stats = exact_dedup(samples)
    samples, phash_stats = phash_dedup(samples)

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
    )

    with tempfile.TemporaryDirectory() as tmp:
        parquet_path = os.path.join(tmp, "dataset.parquet")
        meta_path = os.path.join(tmp, "metadata.json")

        pd.DataFrame(samples).to_parquet(parquet_path, index=False)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        upload_to_minio(tmp, version)
        run_id = log_to_mlflow(version, metadata, parquet_path)

    metadata["mlflow_run_id"] = run_id
    return metadata
