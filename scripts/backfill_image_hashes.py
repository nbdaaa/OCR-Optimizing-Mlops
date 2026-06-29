"""
Backfill per-image hashes (MD5 + pHash) into each data version's metadata.json.

The global dedup in pdf_versioning prefers reading `image_hashes` from a version's
metadata.json over reloading and re-hashing its parquet. Versions created before
that field existed have no hashes, so they fall back to the slow path on every
ingest. This one-off script computes the hashes and patches metadata.json so the
fast path applies everywhere.

Idempotent: a version that already has a non-empty `image_hashes` list is skipped
unless --force. Only metadata.json is rewritten — dataset.parquet and MLflow runs
are left untouched.

Env (from .env): MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET_DATA.

Usage:
    python scripts/backfill_image_hashes.py
    python scripts/backfill_image_hashes.py --only v10,v11 --force
    python scripts/backfill_image_hashes.py --dry-run
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_versioning import compute_image_hash, compute_phash
from src.data.pdf_versioning import (
    _existing_versions,
    _load_samples,
    _minio,
    _version_meta,
)


def _backfill_one(version: str, bucket: str, force: bool) -> tuple[str, str, int]:
    """Worker (runs in a thread): hash one version's images and patch its
    metadata.json. Each call uses its own MinIO client. Returns
    (version, status, count) where status is ok / skip / nometa / 'fail:<msg>'."""
    client = _minio()
    meta = _version_meta(client, bucket, version)
    if meta is None:
        return version, "nometa", 0
    if not force and isinstance(meta.get("image_hashes"), list) and meta["image_hashes"]:
        return version, "skip", len(meta["image_hashes"])

    print(f"  [load] {version}: đang tải parquet + hash ảnh…", flush=True)
    try:
        samples = _load_samples(client, bucket, version)
    except Exception as exc:  # noqa: BLE001 — foreign/unreadable version
        return version, f"fail:{exc}", 0

    image_hashes = [
        {"md5": compute_image_hash(s["image"]), "phash": str(compute_phash(s["image"]))}
        for s in samples if s.get("image") is not None
    ]
    meta["image_hashes"] = image_hashes
    client.put_object(
        Bucket=bucket,
        Key=f"{version}/metadata.json",
        Body=json.dumps(meta, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return version, "ok", len(image_hashes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default=os.environ.get("MINIO_BUCKET_DATA", "ocr-data"))
    ap.add_argument("--only", default=None,
                    help="comma-separated version names to limit to (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if image_hashes is already present")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel threads for the real run (default 8)")
    args = ap.parse_args()

    client = _minio()
    bucket = args.bucket

    versions = sorted(_existing_versions(client, bucket))
    if args.only:
        want = {v.strip() for v in args.only.split(",") if v.strip()}
        versions = [v for v in versions if v in want]
    if not versions:
        sys.exit("Không tìm thấy version nào trong bucket.")

    print(f"[backfill] bucket={bucket} · {len(versions)} version: {versions}"
          + (" · DRY-RUN" if args.dry_run else ""))
    patched = skipped = failed = 0

    # Dry-run is instant (reads only metadata) — keep it sequential.
    if args.dry_run:
        for v in versions:
            meta = _version_meta(client, bucket, v)
            if meta is None:
                print(f"  [skip] {v}: không có metadata.json"); skipped += 1; continue
            if not args.force and isinstance(meta.get("image_hashes"), list) and meta["image_hashes"]:
                print(f"  [skip] {v}: đã có {len(meta['image_hashes'])} hash"); skipped += 1; continue
            print(f"  [dry-run] {v}: would write ~{meta.get('count', '?')} hashes"); patched += 1
        print(f"[backfill] done · patched={patched} skipped={skipped} failed={failed} "
              f"(dry-run, nothing written)")
        return

    # Real run: each version is independent → parallelise across versions. The cost
    # is dominated by downloading the parquets (network I/O), which threads overlap.
    workers = max(1, args.workers)
    print(f"[backfill] {len(versions)} version · {workers} luồng song song…", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_backfill_one, v, bucket, args.force): v for v in versions}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                ver, status, n = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  [fail] {v}: {exc}", flush=True); failed += 1; continue
            if status == "ok":
                print(f"  [ok] {ver}: wrote {n} hashes", flush=True); patched += 1
            elif status == "skip":
                print(f"  [skip] {ver}: đã có {n} hash", flush=True); skipped += 1
            elif status == "nometa":
                print(f"  [skip] {ver}: không có metadata.json", flush=True); skipped += 1
            else:  # "fail:<msg>"
                print(f"  [fail] {ver}: {status[5:]}", flush=True); failed += 1

    print(f"[backfill] done · patched={patched} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    main()
