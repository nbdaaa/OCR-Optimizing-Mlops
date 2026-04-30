import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from common.logging import get_logger
from ingestion.pdf_splitter import pdf_to_images
from ingestion.chandra_client import call_chandra
from ingestion.writer import save_sample

load_dotenv()
log = get_logger(__name__)

SUPPORTED  = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
BATCH_SIZE = 100


def process_one(img_path: str, idx: int, total: int) -> bool:
    try:
        chandra_html = call_chandra(img_path)
        saved        = save_sample(img_path, chandra_html)
        if not saved:
            print(f"[{idx}/{total}] ⏭️  Duplicate — skipped")
            return False
        print(f"[{idx}/{total}] ✅ Saved")
        return True
    except Exception as e:
        log.info(f"[{idx}/{total}] ❌ Failed: {e}")
        print(f"[{idx}/{total}] ❌ Failed: {e}")
        return False


def process_image_batch(image_paths: list[str], offset: int = 0, total: int = 0) -> int:
    """
    Fire all image_paths concurrently and wait for every request to finish.
    Returns the number of successfully saved samples.
    """
    n     = len(image_paths)
    total = total or n

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(process_one, img, offset + i + 1, total): img
            for i, img in enumerate(image_paths)
        }
        succeeded = sum(1 for f in as_completed(futures) if f.result())

    return succeeded


def ingest_file(file_path: str, batch_size: int = BATCH_SIZE) -> int:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext}")

    image_paths = pdf_to_images(file_path) if ext == ".pdf" else [file_path]
    total       = len(image_paths)
    succeeded   = 0

    log.info(f"Processing {total} image(s) with batch_size={batch_size}")

    for batch_start in range(0, total, batch_size):
        batch     = image_paths[batch_start:batch_start + batch_size]
        batch_end = batch_start + len(batch)

        print(f"\n🚀 Firing batch [{batch_start+1}–{batch_end}] / {total} ({len(batch)} requests)...")

        batch_saved = process_image_batch(batch, offset=batch_start, total=total)
        succeeded  += batch_saved

        print(f"   Batch done — saved {batch_saved}/{len(batch)}")

    print(f"\n🎉 Done. {succeeded}/{total} page(s) saved from {path.name}")
    return succeeded


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
