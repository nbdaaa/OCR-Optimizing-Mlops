import os
import json
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from datasets import Dataset
from common.logging import get_logger
from common.exceptions import StorageError

load_dotenv()
log = get_logger(__name__)

LOCAL_REPO   = Path("./data/repo")
HF_TOKEN     = lambda: os.environ.get("HF_TOKEN", "")
DATASET_REPO = lambda: os.environ.get("HF_DATASET_REPO", "")
MODEL_REPO   = lambda: os.environ.get("HF_MODEL_REPO", "")


def _get_api() -> HfApi:
    return HfApi(token=HF_TOKEN())


_repo_initialized = False

def _ensure_repo() -> None:
    global _repo_initialized

    if _repo_initialized:
        return

    if (LOCAL_REPO / ".git").exists():
        _repo_initialized = True
        return

    # only remove if truly not a git repo AND no sample files exist yet
    if LOCAL_REPO.exists():
        has_samples = (
            any((LOCAL_REPO / "chandra_raw").glob("*.html"))
            if (LOCAL_REPO / "chandra_raw").exists() else False
        )
        if has_samples:
            raise StorageError(
                f"{LOCAL_REPO} has sample files but no .git — "
                "do NOT delete. Run: git init && git remote add origin ..."
            )
        import shutil as _shutil
        _shutil.rmtree(LOCAL_REPO)
        log.info(f"Removed empty non-git directory {LOCAL_REPO}")

    repo  = DATASET_REPO()
    token = HF_TOKEN()
    url   = f"https://oauth2:{token}@huggingface.co/datasets/{repo}"

    LOCAL_REPO.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Cloning {repo} → {LOCAL_REPO}")

    result = subprocess.run(
        ["git", "clone", url, str(LOCAL_REPO)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise StorageError(f"git clone failed: {result.stderr}")

    subprocess.run(["git", "-C", str(LOCAL_REPO), "config",
                    "user.email", "bot@chandra-mlops"], check=True)
    subprocess.run(["git", "-C", str(LOCAL_REPO), "config",
                    "user.name",  "chandra-mlops-bot"], check=True)
    subprocess.run(["git", "-C", str(LOCAL_REPO), "lfs", "install"],
                   capture_output=True)

    _repo_initialized = True
    log.info("Repo ready")


def _repo_path(subfolder: str) -> Path:
    _ensure_repo()
    p = LOCAL_REPO / subfolder
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_local_count() -> None:
    """Write count.json to local repo based on real file count."""
    count = get_local_sample_count()
    dest  = LOCAL_REPO / "count.json"
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"count": count}, f)
    log.info(f"count.json → {count}")

# ── write functions — local only, no HF push ─────────────────────────────────

def upload_image(local_path: str, repo_path: str) -> None:
    """Copy image to local repo. No HF push."""
    dest = LOCAL_REPO / repo_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dest)
    log.info(f"Wrote image → {dest}")


def upload_json(data: dict, repo_path: str) -> None:
    """Write JSON to local repo. No HF push."""
    dest = LOCAL_REPO / repo_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote JSON → {dest}")


def upload_raw_html(html_content: str, repo_path: str) -> None:
    """Write raw HTML to local repo. No HF push."""
    dest = LOCAL_REPO / repo_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info(f"Wrote HTML → {dest}")


# ── count based on real files ─────────────────────────────────────────────────

def get_local_sample_count() -> int:
    """
    Count samples by counting files in chandra_raw/.
    This is the real ground truth — no counter file needed.
    """
    raw_dir = LOCAL_REPO / "chandra_raw"
    if not raw_dir.exists():
        return 0
    return len(list(raw_dir.glob("*.html")))


# ── push to HF ────────────────────────────────────────────────────────────────

def push_to_hf(commit_message: str = None) -> int:
    """
    Push all local changes to HF dataset repo.
    Returns number of samples pushed.
    """
    _ensure_repo()
    count   = get_local_sample_count()
    message = commit_message or f"Add {count} samples (auto-batch)"

    log.info(f"Pushing {count} samples to HF...")

    try:
        # stage all changes
        subprocess.run(
            ["git", "-C", str(LOCAL_REPO), "add", "."],
            check=True, capture_output=True,
        )

        # check if there's anything to commit
        status = subprocess.run(
            ["git", "-C", str(LOCAL_REPO), "status", "--porcelain"],
            capture_output=True, text=True,
        )
        if not status.stdout.strip():
            log.info("Nothing to commit — repo is clean")
            return 0

        # commit
        subprocess.run(
            ["git", "-C", str(LOCAL_REPO), "commit", "-m", message],
            check=True, capture_output=True,
        )

        # push
        result = subprocess.run(
            ["git", "-C", str(LOCAL_REPO), "push"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise StorageError(f"git push failed: {result.stderr}")

        log.info(f"✅ Pushed {count} samples to {DATASET_REPO()}")
        return count

    except subprocess.CalledProcessError as e:
        raise StorageError(f"git operation failed: {e.stderr}") from e


def clear_local_repo() -> None:
    """
    Clear all sample files from local repo after successful push.
    Keeps .git/ intact so next batch can push incrementally.
    Removes: images/, manifests/, chandra_raw/
    """
    for folder in ["images", "manifests", "chandra_raw"]:
        p = LOCAL_REPO / folder
        if p.exists():
            shutil.rmtree(p)
            p.mkdir()
            log.info(f"Cleared {folder}/")


# ── model storage — unchanged ─────────────────────────────────────────────────

def push_model(local_dir: str, revision: str) -> None:
    try:
        _get_api().upload_folder(
            folder_path=local_dir,
            repo_id=MODEL_REPO(),
            repo_type="model",
            commit_message=f"Add checkpoint {revision}",
            revision=revision,
            create_pr=False,
        )
        log.info(f"Pushed model revision {revision} → {MODEL_REPO()}")
    except Exception as e:
        raise StorageError(f"Failed to push model: {e}") from e


def download_model(revision: str, local_dir: str) -> str:
    try:
        path = snapshot_download(
            repo_id=MODEL_REPO(),
            revision=revision,
            local_dir=local_dir,
            token=HF_TOKEN(),
        )
        log.info(f"Downloaded model {revision} → {path}")
        return path
    except Exception as e:
        raise StorageError(f"Failed to download model {revision}: {e}") from e


def download_json(repo_path: str, repo_id: str = None,
                  repo_type: str = "dataset") -> dict:
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(
            repo_id=repo_id or DATASET_REPO(),
            filename=repo_path,
            repo_type=repo_type,
            token=HF_TOKEN(),
        )
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        raise StorageError(f"Failed to download {repo_path}: {e}") from e


def push_dataset(dataset: Dataset, config_name: str = "default") -> None:
    try:
        dataset.push_to_hub(DATASET_REPO(), config_name=config_name)
        log.info(f"Pushed dataset config '{config_name}' → {DATASET_REPO()}")
    except Exception as e:
        raise StorageError(f"Failed to push dataset: {e}") from e