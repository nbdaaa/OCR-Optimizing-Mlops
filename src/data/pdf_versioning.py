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
import imagehash
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
# OCR_LAYOUT_PROMPT makes Chandra emit layout HTML; chandra_to_docling converts
# that HTML to DocTags. (bs4 is imported lazily inside the converter, so this
# top-level import stays light for the hash-backfill path.)
from src.data.chandra_converter import OCR_LAYOUT_PROMPT, chandra_to_docling

PROMPT = "Convert this page to docling format."
DEFAULT_MODEL = "chandra"
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
    max_tokens: int = 4096,
    timeout: int = 300,
) -> str:
    """Send one page image to the teacher (Chandra) and return DocTags.

    Chandra is prompted (OCR_LAYOUT_PROMPT) to emit layout HTML — divs with
    data-bbox (0-1000) + data-label — which we convert to DocTags via
    chandra_to_docling. If the endpoint already returns DocTags, it's used as-is.
    Vietnamese tone marks are NFC-fixed at the end.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_uri(img)}},
            {"type": "text", "text": OCR_LAYOUT_PROMPT},
        ]}],
        "max_tokens": int(max_tokens), "temperature": 0.0,
    }
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=timeout)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"] or ""
    if "data-bbox" in raw:                 # Chandra layout HTML → convert
        raw = chandra_to_docling(raw)
    return _fix_vn(raw)


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


def _version_hashes(client, bucket: str, version: str) -> tuple[set[str], list]:
    """Return (md5_set, phash_list) for a version's images.

    Prefers the precomputed hashes stored in metadata.json (cheap: one small
    JSON read); falls back to loading the parquet and hashing the images for
    versions saved before hashes were stored. The phash strings are rebuilt into
    ImageHash objects so the Hamming-distance comparison still works."""
    meta = _version_meta(client, bucket, version)
    hashes = meta.get("image_hashes") if meta else None
    if isinstance(hashes, list) and hashes:
        md5s = {h["md5"] for h in hashes if h.get("md5")}
        phs = [imagehash.hex_to_hash(h["phash"]) for h in hashes if h.get("phash")]
        return md5s, phs
    # fallback: old version without stored hashes → load images and hash them
    md5s, phs = set(), []
    for s in _load_samples(client, bucket, version):
        if s["image"] is None:
            continue
        md5s.add(compute_image_hash(s["image"]))
        phs.append(compute_phash(s["image"]))
    return md5s, phs


def _existing_versions(client, bucket: str) -> set[str]:
    """Top-level version folders in the bucket (e.g. {v1, …, v18, benchmark}).

    Only a genuinely missing bucket counts as "fresh" (→ start at v1). Connection,
    endpoint, or auth failures are raised loudly: silently returning an empty set
    here makes the counter restart at v1 and risks overwriting the real v1."""
    try:
        resp = client.list_objects_v2(Bucket=bucket, Delimiter="/")
    except client.exceptions.NoSuchBucket:
        return set()
    except Exception as exc:  # noqa: BLE001 — surface stale-endpoint / auth issues
        raise RuntimeError(
            f"Không liệt kê được version trong MinIO bucket '{bucket}' tại "
            f"{os.environ.get('MINIO_ENDPOINT')!r}: {exc}. "
            f"Kiểm tra MINIO_ENDPOINT (IP VM có thể đã đổi) / credentials."
        ) from exc
    return {p["Prefix"].rstrip("/") for p in resp.get("CommonPrefixes", [])}


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


def _pdf_versions(client, bucket: str, existing: set[str]) -> dict[str, dict]:
    """{name: metadata} for versions created by THIS PDF pipeline (have
    `target_per_version`) — so auto-resolve never touches HF/other versions."""
    out = {}
    for v in existing:
        m = _version_meta(client, bucket, v)
        if m and "target_per_version" in m:
            out[v] = m
    return out


def _resolve_start_version(client, bucket: str, existing: set[str]) -> str:
    """Auto-pick the version to fill, by counter:
      - resume the latest OPEN PDF version (count < N) if one exists, else
      - the next free vN after the highest existing v-number.
    """
    pdfvers = _pdf_versions(client, bucket, existing)

    def _num(v: str) -> int:
        m = re.search(r"(\d+)$", v)
        return int(m.group(1)) if m else -1

    open_vers = [v for v, m in pdfvers.items() if not m.get("complete", False)]
    if open_vers:
        return max(open_vers, key=_num)        # the still-fillable one

    nums = [int(m.group(1)) for v in existing if (m := re.match(r"^v(\d+)$", v))]
    nxt = (max(nums) + 1) if nums else 1
    while f"v{nxt}" in existing:
        nxt += 1
    return f"v{nxt}"


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
        # log_input is best-effort lineage: pass `name` (not `source`) so MLflow
        # doesn't try to resolve "pdf-inference:http://…" as a dataset-source URI
        # (which raises). The source URL is already captured in params above.
        try:
            mlflow.log_input(
                mlflow.data.from_pandas(
                    pd.DataFrame([{"version": version, "count": metadata["count"]}]),
                    name=f"data-{version}",
                ),
                context="data_versioning",
            )
        except Exception as exc:  # noqa: BLE001 — never let lineage block the save
            print(f"  [mlflow] bỏ qua log_input ({exc})", flush=True)
    return run.info.run_id


def _save_version(client, bucket, version, samples, target, source_url, model, dpi,
                  pages_processed, infer_errors, rejected_benchmark, partial) -> dict:
    """Write one version (parquet + metadata) to MinIO + log MLflow. Overwrites if
    the version already existed (the extend/append case)."""
    # Precompute per-image hashes so later runs can dedup against this version by
    # reading metadata.json instead of reloading and re-hashing every image.
    image_hashes = [
        {"md5": compute_image_hash(s["image"]), "phash": str(compute_phash(s["image"]))}
        for s in samples if s.get("image") is not None
    ]
    metadata = {
        "version": version,
        "count": len(samples),
        "complete": not partial,                 # False → OPEN, can be appended later
        "target_per_version": target,
        "image_hashes": image_hashes,
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
    pdf_paths: list[str],
    inference_url: str,
    max_samples: int,
    version: str | None = None,
    dpi: int = 200,
    max_tokens: int = 4096,
    model: str = DEFAULT_MODEL,
    phash_threshold: int = PHASH_THRESHOLD,
    exclude_versions: list[str] | None = None,
) -> list[dict]:
    """
    Ingest PDFs into fixed-size (N=max_samples) versions.

    - `version=None` (default): auto-pick by counter — resume the latest OPEN PDF
      version if one exists, else the next free vN.
    - If the chosen version is OPEN (count < N): load it, append new pages
      (deduped against existing) until it reaches N.
    - If it is already COMPLETE (count >= N): start at the next free name.
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

    if version is None:
        version = _resolve_start_version(client, bucket, existing)
        print(f"  [auto] version = {version} (theo counter)", flush=True)

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

    # Global dedup: index every image already stored in ANY prior data version, so
    # an incoming page is accepted only if it is genuinely new across the WHOLE
    # dataset — not just unique within the version currently being filled. The
    # version being filled (`cur`) is already covered by `accepted`; benchmark /
    # excluded versions are covered by the leakage guard below.
    prior = [v for v in existing if v != cur and v not in exclude_versions]
    if prior:
        print(f"  [dedup] nạp hash ảnh từ {len(prior)} version trước để chống trùng "
              f"toàn cục (có thể lâu nếu version cũ chưa lưu sẵn hash)…", flush=True)
    n_prior = 0
    for i, pv in enumerate(prior, 1):
        try:
            md5s, phs = _version_hashes(client, bucket, pv)
            exact.update(md5s)
            phashes.extend(phs)
            n_prior += len(md5s)
            print(f"    [dedup {i}/{len(prior)}] '{pv}': +{len(md5s)} hash", flush=True)
        except Exception as exc:  # noqa: BLE001 — skip unreadable/foreign versions
            print(f"    [dedup {i}/{len(prior)}] bỏ qua '{pv}' ({exc})", flush=True)
    if n_prior:
        print(f"  [dedup] nạp {n_prior} hash ảnh từ {len(prior)} version trước "
              f"→ dữ liệu mới phải duy nhất toàn cục", flush=True)

    # Leakage guard: hashes of benchmark (and any excluded version) images. Kept in
    # a SEPARATE index so matches are reported as benchmark rejections, and so they
    # never get counted as accepted training samples.
    bench_exact: set[str] = set()
    bench_phashes: list = []
    for gv in exclude_versions:
        if not gv or _version_meta(client, bucket, gv) is None:
            continue
        md5s, phs = _version_hashes(client, bucket, gv)
        bench_exact.update(md5s)
        bench_phashes.extend(phs)
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
            print(f"  [infer {pages_processed}] {name} p{page_idx} → teacher…", flush=True)
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
                snippet = (sample["output_text"] or "").strip().replace("\n", " ")[:80]
                print(f"  [invalid] {sample['sample_id']}: output không phải DocTags "
                      f"(thiếu <doctag>/<loc_>) → bỏ. Teacher có emit DocTags không? "
                      f"| đầu output: {snippet!r}", flush=True)
                continue
            h_md5 = compute_image_hash(sample["image"])
            ph = compute_phash(sample["image"])
            # leakage guard: drop pages matching a benchmark image (exact/near)
            if h_md5 in bench_exact or any((ph - p) <= phash_threshold for p in bench_phashes):
                rejected_benchmark += 1
                print(f"  [guard] {sample['sample_id']} trùng benchmark → loại", flush=True)
                continue
            # global dedup: reject if the page matches any prior version or any
            # page already accepted this run (exact MD5 or near-duplicate pHash)
            if h_md5 in exact:
                print(f"  [dup] {sample['sample_id']}: trùng tuyệt đối ảnh đã có → bỏ", flush=True)
                continue
            if any((ph - p) <= phash_threshold for p in phashes):
                print(f"  [dup~] {sample['sample_id']}: gần trùng (pHash ≤ {phash_threshold}) "
                      f"ảnh đã có → bỏ", flush=True)
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
