"""Shared paths and constants. Every script imports from here so paths live in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

LABELS = ["clean", "offensive"]  # 0 = clean, 1 = offensive
SEED = 42

# Where the trained model files are published (weights are not stored in git)
HF_MODEL_REPO = "Nurlss/chat-moderation-models"
