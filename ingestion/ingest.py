import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from common.logging import get_logger
from ingestion.pdf_splitter import pdf_to_images
from ingestion.chandra_client import call_chandra
from ingestion.writer import save_sample

load_dotenv()
log = get_logger(__name__)

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def ingest_file(file_path: str) -> None:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {SUPPORTED}"
        )

    # split PDF into pages or treat as single image
    if ext == ".pdf":
        log.info(f"Splitting PDF: {file_path}")
        image_paths = pdf_to_images(file_path)
    else:
        image_paths = [file_path]

    total = len(image_paths)
    log.info(f"Processing {total} image(s) from {file_path}")

    succeeded = 0
    for i, img_path in enumerate(image_paths, 1):
        log.info(f"[{i}/{total}] Processing {img_path}")
        try:
            chandra_html = call_chandra(img_path)
            new_count    = save_sample(img_path, chandra_html)
            print(f"[{i}/{total}] ✅ Saved — total samples: {new_count}")
            succeeded   += 1
        except Exception as e:
            log.info(f"[{i}/{total}] ❌ Failed: {e}")
            print(f"[{i}/{total}] ❌ Failed: {e}")
            continue

    print(f"\n🎉 Done. {succeeded}/{total} page(s) saved from {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a PDF or image through Chandra OCR-2 and save to HuggingFace"
    )
    parser.add_argument("file", help="Path to PDF or image file")
    args = parser.parse_args()
    ingest_file(args.file)
