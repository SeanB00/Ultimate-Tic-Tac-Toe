"""Shared filesystem paths for the Ultimate Tic-Tac-Toe project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QTABLE_DIR = PROJECT_ROOT / "data" / "qtable"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CNN_RUNS_DIR = PROJECT_ROOT / "models" / "cnn_runs_final"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FIXED_QTABLE_PATH = QTABLE_DIR / "fixed_qtable.lmdb"
SHRUNK_QTABLE_PATH = QTABLE_DIR / "qtable_shrink.lmdb"

EXPANDED_X_PATH = PROCESSED_DATA_DIR / "expanded_X_min2.npy"
EXPANDED_Y_PATH = PROCESSED_DATA_DIR / "expanded_y_min2.npy"
EXPANDED_META_PATH = PROCESSED_DATA_DIR / "expanded_meta_min2.json"

MIXED_X_PATH = PROCESSED_DATA_DIR / "mixed_X_v1.npy"
MIXED_Y_PATH = PROCESSED_DATA_DIR / "mixed_y_v1.npy"
MIXED_META_PATH = PROCESSED_DATA_DIR / "mixed_meta_v1.json"


def ensure_project_dirs() -> None:
    """Create the standard writable project directories if they are missing."""
    for path in (
        QTABLE_DIR,
        PROCESSED_DATA_DIR,
        CNN_RUNS_DIR,
        ARTIFACTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
