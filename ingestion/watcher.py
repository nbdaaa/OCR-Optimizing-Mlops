import os
import time
import signal
import shutil
from pathlib import Path
from dotenv import load_dotenv
from common.logging import get_logger
from common.storage import get_local_sample_count, push_to_hf, clear_local_repo, LOCAL_REPO, _ensure_repo
from ingestion.ingest import ingest_file

load_dotenv()
log = get_logger(__name__)

WATCH_DIR     = Path("./data")
DONE_DIR      = Path("./data/done")
FAILED_DIR    = Path("./data/failed")
SUPPORTED     = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
POLL_INTERVAL = 5
COMMIT_EVERY  = 1000

_running       = True
_shutdown_once = False   # ← prevent multiple shutdowns


def setup_dirs():
    WATCH_DIR.mkdir(exist_ok=True)
    DONE_DIR.mkdir(exist_ok=True)
    FAILED_DIR.mkdir(exist_ok=True)


def scan_once() -> list[Path]:
    return [
        f for f in WATCH_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED
    ]


def _try_push():
    count = get_local_sample_count()
    if count >= COMMIT_EVERY:
        print(f"\n🚀 {count} samples in local repo — pushing to HF...")
        pushed = push_to_hf(f"Add {count} samples (auto-batch)")
        clear_local_repo()
        print(f"✅ Pushed {pushed} samples to HF. Local repo cleared.\n")


def _shutdown(signum, frame):
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


def _final_push():
    """Push remaining samples to HF. Called after main loop exits."""
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


def main():
    setup_dirs()

    # ensure repo is cloned before any processing starts
    from common.storage import _ensure_repo
    print("🔄 Initializing local repo...")
    _ensure_repo()
    print(f"✅ Local repo ready at {(LOCAL_REPO).resolve()}\n")

    print(f"👀 Watcher started — drop PDF/images into {WATCH_DIR.resolve()}")
    print(f"   HF push every   : {COMMIT_EVERY} samples")
    print(f"   Poll interval   : {POLL_INTERVAL}s")
    print(f"   Local repo      : ./data/repo/")
    print(f"   Ctrl+C once to stop gracefully\n")

    while _running:
        files = scan_once()

        for file_path in files:
            if not _running:
                break

            print(f"\n📄 Processing: {file_path.name}")
            try:
                pages_saved = ingest_file(str(file_path))
                count       = get_local_sample_count()

                print(f"   Pages saved     : {pages_saved}")
                print(f"   Local samples   : {count} / {COMMIT_EVERY}")

                _try_push()

                dest = DONE_DIR / file_path.name
                if dest.exists():
                    dest = DONE_DIR / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
                shutil.move(str(file_path), dest)
                print(f"✅ Moved → {dest.name}")

            except Exception as e:
                log.info(f"❌ Failed {file_path.name}: {e}")
                print(f"❌ Failed: {file_path.name} — {e}")
                dest = FAILED_DIR / file_path.name
                if dest.exists():
                    dest = FAILED_DIR / f"{file_path.stem}_{int(time.time())}{file_path.suffix}"
                shutil.move(str(file_path), dest)

        if not _running:
            break

        time.sleep(POLL_INTERVAL)

    # ── graceful shutdown ─────────────────────────────────────────────────────
    print("\n🛑 Main loop exited. Waiting for in-flight requests to finish...")
    # give threads 10s to finish current batch
    time.sleep(10)

    _final_push()
    print("\n👋 Watcher stopped.")


if __name__ == "__main__":
    main()