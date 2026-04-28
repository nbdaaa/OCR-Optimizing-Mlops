import sys
import argparse
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from common.logging import get_logger
from ingestion.pdf_splitter import pdf_to_images
from ingestion.chandra_client import call_chandra
from ingestion.writer import save_sample

load_dotenv()
log = get_logger(__name__)

SUPPORTED      = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
BATCH_SIZE     = 30   # số requests gửi song song cùng lúc


def process_one(img_path: str, idx: int, total: int) -> tuple[str, int] | None:
    try:
        chandra_html = call_chandra(img_path)
        new_count    = save_sample(img_path, chandra_html)
        if new_count is None:
            print(f"[{idx}/{total}] ⏭️  Duplicate — skipped")
            return None
        print(f"[{idx}/{total}] ✅ Saved — total samples: {new_count}")
        return img_path, new_count
    except Exception as e:
        log.info(f"[{idx}/{total}] ❌ Failed: {e}")  # ← use idx not i
        print(f"[{idx}/{total}] ❌ Failed: {e}")
        return None


def ingest_file(file_path: str, batch_size: int = BATCH_SIZE) -> None:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED}")

    # split PDF hoặc dùng trực tiếp ảnh
    if ext == ".pdf":
        log.info(f"Splitting PDF: {file_path}")
        image_paths = pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    total     = len(image_paths)
    succeeded = 0

    log.info(f"Processing {total} image(s) with batch_size={batch_size}")

    # chia thành các batch 30 ảnh
    for batch_start in range(0, total, batch_size):
        batch       = image_paths[batch_start:batch_start + batch_size]
        batch_end   = batch_start + len(batch)
        print(f"\n🚀 Firing batch [{batch_start+1}–{batch_end}] / {total} ({len(batch)} requests)...")

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(
                    process_one,
                    img_path,
                    batch_start + i + 1,
                    total,
                ): img_path
                for i, img_path in enumerate(batch)
            }

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    succeeded += 1

        print(f"   Batch done — {succeeded}/{batch_end} succeeded so far")

    print(f"\n🎉 Done. {succeeded}/{total} page(s) saved from {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF/image through Chandra OCR-2 and save to HuggingFace"
    )
    parser.add_argument("file", help="Path to PDF or image file")
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Number of concurrent requests per batch (default: {BATCH_SIZE})"
    )
    args = parser.parse_args()
    ingest_file(args.file, batch_size=args.batch_size)