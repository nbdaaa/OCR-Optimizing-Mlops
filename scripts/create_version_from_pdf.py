"""
Create a data version by running PDFs through the inference server (self-labeling).

Splits each PDF into pages → renders to images → asks the inference server for
DocTags → accumulates (image, output_text) until `--max-samples` VALID, deduped
samples are collected → writes the version to MinIO + MLflow (same format as the
HF-based pipeline, so training reads it identically).

Does NOT touch the original HF-based scripts (create_all_versions.py etc.).

Usage:
    python scripts/create_version_from_pdf.py \
        --version v20 \
        --inference-url http://34.142.198.19 \
        --max-samples 200 \
        --pdf doc1.pdf doc2.pdf
    # or ingest every *.pdf in a folder:
    python scripts/create_version_from_pdf.py \
        --version v20 --inference-url http://34.142.198.19 \
        --max-samples 200 --pdf-dir ./pdfs

Env (from .env): MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
                 MINIO_BUCKET_DATA, MLFLOW_TRACKING_URI.
"""
import argparse
import glob
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.pdf_versioning import create_version_from_pdfs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="Version name, e.g. v20")
    ap.add_argument("--inference-url", required=True,
                    help="Serving LB base URL, e.g. http://34.142.198.19")
    ap.add_argument("--max-samples", type=int, required=True,
                    help="FINAL sample count for this version (after filter + dedup)")
    ap.add_argument("--pdf", nargs="*", default=[], help="PDF file path(s)")
    ap.add_argument("--pdf-dir", default=None, help="Directory of *.pdf to ingest")
    ap.add_argument("--dpi", type=int, default=200, help="Render DPI (default 200)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model", default="ocr-adapter")
    ap.add_argument("--benchmark-version", default=None,
                    help="Version to exclude as leakage guard (default: BENCHMARK_VERSION env / 'benchmark')")
    ap.add_argument("--no-exclude-benchmark", action="store_true",
                    help="Disable the benchmark leakage guard (NOT recommended)")
    args = ap.parse_args()

    pdfs = list(args.pdf)
    if args.pdf_dir:
        pdfs += sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
    if not pdfs:
        ap.error("cần --pdf <files> hoặc --pdf-dir <folder>")
    missing = [p for p in pdfs if not os.path.isfile(p)]
    if missing:
        ap.error(f"không tìm thấy PDF: {missing}")

    if args.no_exclude_benchmark:
        exclude = []
    elif args.benchmark_version:
        exclude = [args.benchmark_version]
    else:
        exclude = None   # → module default (BENCHMARK_VERSION env / "benchmark")

    print(f"[pdf-version] {len(pdfs)} PDF → {args.inference_url} → bắt đầu từ "
          f"'{args.version}' (N={args.max_samples}/version, dpi={args.dpi})")
    written = create_version_from_pdfs(
        version=args.version,
        pdf_paths=pdfs,
        inference_url=args.inference_url,
        max_samples=args.max_samples,
        dpi=args.dpi,
        max_tokens=args.max_tokens,
        model=args.model,
        exclude_versions=exclude,
    )
    if not written:
        print("\n(không version nào được ghi — không có sample mới)")
        return
    print(f"\n✅ Đã ghi {len(written)} version:")
    for m in written:
        state = "COMPLETE" if m["complete"] else "OPEN (còn append được)"
        print(f"   - {m['version']}: {m['count']}/{m['target_per_version']} · {state} "
              f"· run {m['mlflow_run_id'][:8]}")
    total_bench = written[-1]["filter_stats"].get("rejected_benchmark", 0)
    print(f"   🛡️  loại do trùng benchmark: {total_bench} trang")


if __name__ == "__main__":
    main()
