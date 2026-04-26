import os
import json
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from datasets import Dataset
from common.logging import get_logger
from common.exceptions import StorageError

log          = get_logger(__name__)
_api         = HfApi(token=os.environ.get("HF_TOKEN"))
DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")
MODEL_REPO   = os.environ.get("HF_MODEL_REPO", "")


def upload_image(local_path: str, repo_path: str) -> None:
    try:
        _api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        log.info(f"Uploaded image → {DATASET_REPO}/{repo_path}")
    except Exception as e:
        raise StorageError(f"Failed to upload image: {e}") from e


def upload_json(data: dict, repo_path: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp = f.name
    try:
        _api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=repo_path,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )
        log.info(f"Uploaded JSON → {DATASET_REPO}/{repo_path}")
    finally:
        Path(tmp).unlink(missing_ok=True)


def download_json(repo_path: str, repo_id: str = None, repo_type: str = "dataset") -> dict:
    try:
        local = hf_hub_download(
            repo_id=repo_id or DATASET_REPO,
            filename=repo_path,
            repo_type=repo_type,
            token=os.environ.get("HF_TOKEN"),
        )
        with open(local) as f:
            return json.load(f)
    except Exception as e:
        raise StorageError(f"Failed to download {repo_path}: {e}") from e


def push_dataset(dataset: Dataset, config_name: str = "default") -> None:
    try:
        dataset.push_to_hub(DATASET_REPO, config_name=config_name)
        log.info(f"Pushed dataset config '{config_name}' → {DATASET_REPO}")
    except Exception as e:
        raise StorageError(f"Failed to push dataset: {e}") from e


def push_model(local_dir: str, revision: str) -> None:
    try:
        _api.upload_folder(
            folder_path=local_dir,
            repo_id=MODEL_REPO,
            repo_type="model",
            commit_message=f"Add checkpoint {revision}",
            revision=revision,
            create_pr=False,
        )
        log.info(f"Pushed model revision {revision} → {MODEL_REPO}")
    except Exception as e:
        raise StorageError(f"Failed to push model: {e}") from e


def download_model(revision: str, local_dir: str) -> str:
    try:
        path = snapshot_download(
            repo_id=MODEL_REPO,
            revision=revision,
            local_dir=local_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        log.info(f"Downloaded model {revision} → {path}")
        return path
    except Exception as e:
        raise StorageError(f"Failed to download model {revision}: {e}") from e
