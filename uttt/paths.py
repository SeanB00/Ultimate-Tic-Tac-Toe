"""shared filesystem paths for the project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QTABLE_DIR = PROJECT_ROOT / "data" / "qtable"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CNN_RUNS_DIR = PROJECT_ROOT / "models" / "cnn_runs_final"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

FIXED_QTABLE_PATH = QTABLE_DIR / "fixed_qtable.lmdb"
SHRUNK_QTABLE_PATH = QTABLE_DIR / "qtable_shrink.lmdb"

FILTERED_X_PATH = PROCESSED_DATA_DIR / "filtered_X.npy"
FILTERED_Y_PATH = PROCESSED_DATA_DIR / "filtered_y.npy"
FILTERED_META_PATH = PROCESSED_DATA_DIR / "filtered_meta.json"

MIXED_X_PATH = PROCESSED_DATA_DIR / "mixed_X.npy"
MIXED_Y_PATH = PROCESSED_DATA_DIR / "mixed_y.npy"
MIXED_META_PATH = PROCESSED_DATA_DIR / "mixed_meta.json"


def ensure_project_dirs() -> None:
    """create the standard project directories."""
    for path in (
        QTABLE_DIR,
        PROCESSED_DATA_DIR,
        CNN_RUNS_DIR,
        ARTIFACTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
