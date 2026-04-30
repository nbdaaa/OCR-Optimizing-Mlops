import time
import signal
import shutil
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from common.logging import get_logger
from common.storage import (
    get_local_sample_count, push_to_hf, clear_local_repo, LOCAL_REPO, _ensure_repo,
)
from ingestion.pdf_splitter import pdf_to_images
from ingestion.ingest import process_image_batch, BATCH_SIZE

load_dotenv()
log = get_logger(__name__)

WATCH_DIR     = Path("./data")
DONE_DIR      = Path("./data/done")
FAILED_DIR    = Path("./data/failed")
SUPPORTED     = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
POLL_INTERVAL = 5
COMMIT_EVERY  = 1000

_running       = True
_shutdown_once = False
_image_queue: list[tuple[str, Path]] = []  # (img_path, source_file)
_source_remaining: dict[Path, int]   = {}  # source_file → images not yet processed


def setup_dirs() -> None:
    WATCH_DIR.mkdir(exist_ok=True)
    DONE_DIR.mkdir(exist_ok=True)
    FAILED_DIR.mkdir(exist_ok=True)


def scan_once() -> list[Path]:
    return [
        f for f in WATCH_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED
    ]


def _move(src: Path, dest_dir: Path) -> None:
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.move(str(src), dest)
    print(f"   → {dest_dir.name}/{dest.name}")


def _try_push() -> None:
    count = get_local_sample_count()
    if count >= COMMIT_EVERY:
        print(f"\n🚀 {count} samples in local repo — pushing to HF...")
        pushed = push_to_hf(f"Add {count} samples (auto-batch)")
        clear_local_repo()
        print(f"✅ Pushed {pushed} samples to HF. Local repo cleared.\n")


def _fire_batch(batch: list[tuple[str, Path]]) -> None:
    global _source_remaining

    image_paths = [img for img, _ in batch]
    sources     = [src for _, src in batch]

    print(f"\n🚀 Firing batch of {len(batch)} image(s)...")
    succeeded = process_image_batch(image_paths)
    print(f"   Batch done — {succeeded}/{len(batch)} saved | local total: {get_local_sample_count()}")

    # Decrement remaining count; move source file to done/ when all images processed
    for src, dispatched in Counter(sources).items():
        _source_remaining[src] -= dispatched
        if _source_remaining[src] <= 0:
            del _source_remaining[src]
            _move(src, DONE_DIR)

    _try_push()


def _shutdown(signum, frame) -> None:
    global _running, _shutdown_once
    if _shutdown_once:
        print("\n   Already shutting down — please wait...")
        return
    _shutdown_once = True
    _running       = False
    print("\n\n🛑 Shutdown signal received — finishing current batch first...")
    print("   (press Ctrl+C again ONLY if completely stuck)\n")


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


def _final_push() -> None:
    count = get_local_sample_count()
    if count == 0:
        print("   No remaining samples to push.")
        return
    print(f"   Pushing {count} remaining sample(s) to HF...")
    try:
        pushed = push_to_hf(f"Add {count} samples (shutdown flush)")
        clear_local_repo()
        print(f"   ✅ Pushed {pushed} samples. Done.")
    except Exception as e:
        print(f"   ❌ Push failed: {e}")
        print(f"   ⚠️  {count} samples are safe in {LOCAL_REPO} — push manually with:")
        print(f"       cd data/repo && git add . && git commit -m 'manual push' && git push")


def main() -> None:
    global _image_queue

    setup_dirs()

    print("🔄 Initializing local repo...")
    _ensure_repo()
    print(f"✅ Local repo ready at {LOCAL_REPO.resolve()}\n")

    print(f"👀 Watcher started — drop PDF/images into {WATCH_DIR.resolve()}")
    print(f"   Batch size      : {BATCH_SIZE} images")
    print(f"   HF push every   : {COMMIT_EVERY} samples")
    print(f"   Poll interval   : {POLL_INTERVAL}s")
    print(f"   Local repo      : ./data/repo/")
    print(f"   Ctrl+C once to stop gracefully\n")

    while _running:
        for file_path in scan_once():
            if file_path in _source_remaining:
                continue  # already split and queued

            print(f"\n📄 Splitting: {file_path.name}")
            try:
                imgs = (
                    pdf_to_images(str(file_path))
                    if file_path.suffix.lower() == ".pdf"
                    else [str(file_path)]
                )
                _source_remaining[file_path] = len(imgs)
                _image_queue.extend((img, file_path) for img in imgs)
                print(f"   {len(imgs)} image(s) queued — total pending: {len(_image_queue)}")
            except Exception as e:
                log.info(f"❌ Split failed {file_path.name}: {e}")
                print(f"❌ Split failed: {file_path.name} — {e}")
                _move(file_path, FAILED_DIR)

        # Fire complete batches
        while len(_image_queue) >= BATCH_SIZE:
            batch        = _image_queue[:BATCH_SIZE]
            _image_queue = _image_queue[BATCH_SIZE:]
            _fire_batch(batch)

        time.sleep(POLL_INTERVAL)

    # ── graceful shutdown ─────────────────────────────────────────────────────
    print("\n🛑 Main loop exited. Finishing in-progress work...")

    if _image_queue:
        print(f"   Flushing {len(_image_queue)} remaining image(s)...")
        _fire_batch(_image_queue)
        _image_queue = []

    _final_push()
    print("\n👋 Watcher stopped.")


if __name__ == "__main__":
    main()
