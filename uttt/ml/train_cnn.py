"""Unified entrypoint for CNN training jobs."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uttt.ml.cnn_core as cnn_core
from uttt.paths import EXPANDED_X_PATH, EXPANDED_Y_PATH, MIXED_X_PATH, MIXED_Y_PATH



BATCH_SIZE = 4096
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42
NUM_WORKERS = 0


def main() -> None:
    """Run the selected CNN training job."""
    if cnn_core.TRAINING_NAME == "mixed":
        x_path = MIXED_X_PATH
        y_path = MIXED_Y_PATH
    elif cnn_core.TRAINING_NAME == "filtered":
        x_path = EXPANDED_X_PATH
        y_path = EXPANDED_Y_PATH

    else:
        raise ValueError(f"Unknown TRAINING_NAME: {cnn_core.TRAINING_NAME}")

    device = cnn_core.pick_device()
    train_loader, val_loader, test_loader = cnn_core.build_npy_dataloaders(
        x_path=x_path,
        y_path=y_path,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        device=device,
    )

    print(f"{cnn_core.TRAINING_NAME} training:")
    print(
        f">>> Train rows: {len(train_loader.dataset):,} | "
        f"Val rows: {len(val_loader.dataset):,} | "
        f"Test rows: {len(test_loader.dataset):,}"
    )
    print(
        f">>> Settings: model={cnn_core.TRAIN_MODEL_OPTION}, epochs={cnn_core.TRAIN_EPOCHS}, "
        f"batch={BATCH_SIZE}, lr={cnn_core.TRAIN_LR}, "
        f"split=train-rest/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}, "
        f"auto_resume={cnn_core.DEFAULT_AUTO_RESUME}, "
        f"early_stop_patience={cnn_core.DEFAULT_EARLY_STOPPING_PATIENCE}, "
        f"early_stop_min_delta={cnn_core.DEFAULT_EARLY_STOPPING_MIN_DELTA}"
    )

    result = cnn_core.train_supervised_model(
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    print(f">>> [{cnn_core.TRAIN_MODEL_OPTION}] Done")
    print(f">>> [{cnn_core.TRAIN_MODEL_OPTION}] Last: {result['model_path']}")
    print(f">>> [{cnn_core.TRAIN_MODEL_OPTION}] Best: {result['best_model_path']}")
    print(
        f">>> [{cnn_core.TRAIN_MODEL_OPTION}] Plots: "
        f"{result['train_plot_path']} | {result['val_plot_path']}"
    )


if __name__ == "__main__":
    main()
