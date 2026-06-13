"""
PDF → inference-server → data version (self-labeling, append/overflow pipeline).

An ALTERNATIVE source to data_versioning.py (HF Hub). Splits input PDFs into
pages, renders each to an image, sends it to the inference server (serving LB,
OpenAI-compatible /v1/chat/completions) for DocTags, and fills data versions to
a fixed size N:

  - A version with < N samples is OPEN: a later ingest loads it back, dedups
    against its existing samples, and appends until it reaches N.
  - When the accumulated set reaches N the version is CLOSED with exactly N
    samples (existing-first order preserved), and the overflow spills into a new
    auto-numbered version (v20 → v21 → …), which itself fills to N or stays OPEN.

Reuses the same parquet schema, validation, dedup primitives, MinIO upload, and
MLflow lineage as data_versioning.py, so training reads these versions
identically. The original HF-based pipeline is untouched.

CLI: scripts/create_version_from_pdf.py
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone

import boto3
import pandas as pd
import requests
from PIL import Image

from src.data.data_versioning import (
    compute_image_hash,
    compute_phash,
    is_valid,
    upload_to_minio,
    write_parquet,
)

PROMPT = "Convert this page to docling format."
DEFAULT_MODEL = "ocr-adapter"
PHASH_THRESHOLD = 8   # hamming distance for near-duplicate images (matches phash_dedup)


# This model emits Vietnamese tone marks as SPACING chars (´ ` ˜ ˇ) instead of
# combining marks; map to combining + NFC so stored LABELS are clean. Same fix as
# the experiment UI. Does NOT touch <loc_>/element tags.
_SPACING_TONE = str.maketrans({
    "´": "́", "ˊ": "́",
    "`": "̀", "ˋ": "̀",
    "˜": "̃",
    "ˇ": "̉",
})


def _fix_vn(text: str) -> str:
    return unicodedata.normalize("NFC", text.translate(_SPACING_TONE))


# ── PDF → page images ─────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, dpi: int = 200):
    """Yield (page_index, PIL.Image RGB) for each page of the PDF at `dpi`."""
    import fitz  # PyMuPDF — imported lazily so the module imports without it

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            pix = doc.load_page(i).get_pixmap(matrix=mat, alpha=False)
            yield i, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


# ── Inference ───────────────────────────────────────────────────────────────--

def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def infer_doctags(
    img: Image.Image,
    base_url: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    timeout: int = 300,
) -> str:
    """Send one page image to the serving LB; return DocTags (special tokens kept,
    Vietnamese tone marks NFC-fixed)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri(img)}},
            {"type": "text", "text": PROMPT},
        ]}],
        "max_tokens": int(max_tokens), "temperature": 0.0,
        "skip_special_tokens": False,   # keep <loc_>/element tags for training labels
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    return _fix_vn(r.json()["choices"][0]["message"]["content"])


def _image_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── MinIO helpers (read existing versions for resume/append) ──────────────────

def _minio():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
    )


def _version_meta(client, bucket: str, version: str) -> dict | None:
    try:
        obj = client.get_object(Bucket=bucket, Key=f"{version}/metadata.json")
        return json.loads(obj["Body"].read())
    except Exception:  # noqa: BLE001 — absent or unreadable → treat as new
        return None


def _load_samples(client, bucket: str, version: str) -> list[dict]:
    """Read an existing version's parquet back into the sample-dict format."""
    obj = client.get_object(Bucket=bucket, Key=f"{version}/dataset.parquet")
    df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
    out = []
    for _, r in df.iterrows():
        img = r["image"]
        img = img.get("bytes") if isinstance(img, dict) else (bytes(img) if img is not None else None)
        out.append({
            "sample_id": r.get("sample_id"),
            "image": img,
            "output_text": r.get("output_text"),
            "source": r.get("source", "pdf-inference"),
            "img_w": int(r.get("img_w") or 0),
            "img_h": int(r.get("img_h") or 0),
        })
    return out


def _existing_versions(client, bucket: str) -> set[str]:
    try:
        resp = client.list_objects_v2(Bucket=bucket, Delimiter="/")
        return {p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])}
    except Exception:  # noqa: BLE001
        return set()


def _next_free_version(existing: set[str], base: str) -> str:
    """Next unused version name: increment a trailing integer (v20→v21), else _2,_3."""
    m = re.match(r"^(.*?)(\d+)$", base)
    if m:
        prefix, num = m.group(1), int(m.group(2)) + 1
        while f"{prefix}{num}" in existing:
            num += 1
        return f"{prefix}{num}"
    k = 2
    while f"{base}_{k}" in existing:
        k += 1
    return f"{base}_{k}"


# ── MLflow lineage ─────────────────────────────────────────────────────────────

def _log_to_mlflow(version: str, metadata: dict, meta_json_path: str) -> str:
    import mlflow

    os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("MINIO_ACCESS_KEY", ""))
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("MINIO_SECRET_KEY", ""))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("data-versioning")

    with mlflow.start_run(run_name=f"data-{version}") as run:
        mlflow.log_params({
            "version": version,
            "source": metadata["source"],
            "model": metadata["model"],
            "dpi": metadata["dpi"],
            "count": metadata["count"],
            "complete": metadata["complete"],
            **metadata["filter_stats"],
        })
        mlflow.log_artifact(meta_json_path, artifact_path=version)
        mlflow.log_input(
            mlflow.data.from_pandas(
                pd.DataFrame([{"version": version, "count": metadata["count"]}]),
                source=metadata["source"],
            ),
            context="data_versioning",
        )
    return run.info.run_id


def _save_version(client, bucket, version, samples, target, source_url, model, dpi,
                  pages_processed, infer_errors, rejected_benchmark, partial) -> dict:
    """Write one version (parquet + metadata) to MinIO + log MLflow. Overwrites if
    the version already existed (the extend/append case)."""
    metadata = {
        "version": version,
        "count": len(samples),
        "complete": not partial,                 # False → OPEN, can be appended later
        "target_per_version": target,
        "source": f"pdf-inference:{source_url}",
        "model": model,
        "dpi": dpi,
        # run-level stats (this ingest run, not cumulative across runs)
        "filter_stats": {
            "pages_processed": pages_processed,
            "infer_errors": infer_errors,
            "rejected_benchmark": rejected_benchmark,
            "final_count": len(samples),
            "target": target,
        },
        "offset": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        pq = os.path.join(tmp, "dataset.parquet")
        mj = os.path.join(tmp, "metadata.json")
        write_parquet(samples, pq)
        with open(mj, "w") as f:
            json.dump(metadata, f, indent=2)
        upload_to_minio(tmp, version)
        metadata["mlflow_run_id"] = _log_to_mlflow(version, metadata, mj)
    return metadata


# ── Pipeline: fill versions to N, append to OPEN, overflow to new ─────────────--

def create_version_from_pdfs(
    version: str,
    pdf_paths: list[str],
    inference_url: str,
    max_samples: int,
    dpi: int = 200,
    max_tokens: int = 2048,
    model: str = DEFAULT_MODEL,
    phash_threshold: int = PHASH_THRESHOLD,
    exclude_versions: list[str] | None = None,
) -> list[dict]:
    """
    Ingest PDFs into fixed-size (N=max_samples) versions, starting at `version`.

    - If `version` exists and is OPEN (count < N): load it, append new pages
      (deduped against existing) until it reaches N.
    - If `version` is already COMPLETE (count >= N): start at the next free name.
    - Each version closes at exactly N; overflow spills into the next free
      auto-numbered version. The last version may stay OPEN (< N) for next time.

    Leakage guard: pages that match (exact or near-duplicate) any image in
    `exclude_versions` are dropped, so the held-out benchmark never leaks into a
    training version. Defaults to the BENCHMARK_VERSION env (or "benchmark");
    pass [] to disable.

    Returns a list of metadata dicts, one per version written this run.
    """
    base_url = inference_url.rstrip("/")
    bucket = os.environ.get("MINIO_BUCKET_DATA", "ocr-data")
    N = int(max_samples)
    client = _minio()
    existing = _existing_versions(client, bucket)

    if exclude_versions is None:
        exclude_versions = [os.environ.get("BENCHMARK_VERSION", "benchmark")]

    # decide the starting (open) version + preload its samples if extending
    meta0 = _version_meta(client, bucket, version)
    if meta0 is None:
        accepted, cur = [], version
    elif int(meta0.get("count", 0)) < N:
        print(f"  [resume] '{version}' đang OPEN ({meta0.get('count')}/{N}) → append tiếp", flush=True)
        accepted, cur = _load_samples(client, bucket, version), version
    else:
        cur = _next_free_version(existing, version)
        accepted = []
        print(f"  [skip] '{version}' đã COMPLETE ({meta0.get('count')}/{N}) → tràn sang '{cur}'", flush=True)

    # dedup indexes accumulate across the WHOLE run (incl. preloaded existing) so
    # overflow versions don't re-accept a near-duplicate of an earlier page.
    exact = {compute_image_hash(s["image"]) for s in accepted}
    phashes = [compute_phash(s["image"]) for s in accepted]

    # Leakage guard: hashes of benchmark (and any excluded version) images. Kept in
    # a SEPARATE index so matches are reported as benchmark rejections, and so they
    # never get counted as accepted training samples.
    bench_exact: set[str] = set()
    bench_phashes: list = []
    for gv in exclude_versions:
        if not gv or _version_meta(client, bucket, gv) is None:
            continue
        for s in _load_samples(client, bucket, gv):
            if s["image"] is None:
                continue
            bench_exact.add(compute_image_hash(s["image"]))
            bench_phashes.append(compute_phash(s["image"]))
    if bench_exact:
        print(f"  [guard] {len(bench_exact)} benchmark hashes loaded "
              f"({exclude_versions}) → sẽ loại khỏi version train", flush=True)

    added = 0                 # NEW samples added to `cur` since it became current
    pages_processed = 0
    infer_errors = 0
    rejected_benchmark = 0
    written: list[dict] = []

    def _save(ver: str, samples: list[dict], partial: bool) -> None:
        meta = _save_version(client, bucket, ver, samples, N, base_url, model, dpi,
                             pages_processed, infer_errors, rejected_benchmark, partial)
        existing.add(ver)
        written.append(meta)
        print(f"  ✔ saved '{ver}': {len(samples)} samples "
              f"({'OPEN' if partial else 'COMPLETE'}) · run {meta['mlflow_run_id'][:8]}",
              flush=True)

    stop = False
    for pdf_path in pdf_paths:
        if stop:
            break
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        for page_idx, img in pdf_to_images(pdf_path, dpi=dpi):
            pages_processed += 1
            try:
                doctags = infer_doctags(img, base_url, model=model, max_tokens=max_tokens)
            except Exception as exc:  # noqa: BLE001 — skip page, keep going
                infer_errors += 1
                print(f"  [warn] {name} p{page_idx}: inference lỗi ({exc})", flush=True)
                continue

            w, h = img.size
            sample = {
                "sample_id": f"{name}_p{page_idx:04d}",
                "image": _image_bytes(img),
                "output_text": doctags,
                "source": "pdf-inference",
                "img_w": w,
                "img_h": h,
            }
            if not is_valid(sample):
                continue
            h_md5 = compute_image_hash(sample["image"])
            ph = compute_phash(sample["image"])
            # leakage guard: drop pages matching a benchmark image (exact/near)
            if h_md5 in bench_exact or any((ph - p) <= phash_threshold for p in bench_phashes):
                rejected_benchmark += 1
                print(f"  [guard] {sample['sample_id']} trùng benchmark → loại", flush=True)
                continue
            # within-run / current-version dedup
            if h_md5 in exact:
                continue
            if any((ph - p) <= phash_threshold for p in phashes):
                continue

            accepted.append(sample)
            exact.add(h_md5)
            phashes.append(ph)
            added += 1
            print(f"  + {sample['sample_id']} → {cur} ({len(accepted)}/{N})", flush=True)

            if len(accepted) >= N:
                _save(cur, accepted[:N], partial=False)   # close at exactly N
                cur = _next_free_version(existing, cur)
                accepted, added = [], 0                    # exact/phashes keep growing

    # save the trailing OPEN version only if it gained new samples this run
    if added > 0:
        _save(cur, accepted, partial=(len(accepted) < N))

    if not written:
        print("  (không thêm được sample mới nào — không version nào được ghi)", flush=True)
    return written
