"""Download the trained model files from the Hugging Face Hub if they are not on disk yet.

Model weights are NOT in the git repository (R7) - they live on the Hugging Face Hub.
If you trained the models yourself (src/train_*.py), the local files are used instead.
"""
from huggingface_hub import snapshot_download

from src.config import HF_MODEL_REPO, MODELS_DIR
from src.predictors import ALL_PREDICTORS


def missing_files():
    return [f for cls in ALL_PREDICTORS for f in cls.files if not (MODELS_DIR / f).exists()]


def ensure_models():
    if not missing_files():
        return
    print(f"Downloading models from https://huggingface.co/{HF_MODEL_REPO} ...")
    snapshot_download(repo_id=HF_MODEL_REPO, local_dir=MODELS_DIR)
    still_missing = missing_files()
    if still_missing:
        raise FileNotFoundError(f"Model files missing after download: {still_missing}")


if __name__ == "__main__":
    ensure_models()
    print("All model files present in", MODELS_DIR)
