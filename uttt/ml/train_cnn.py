"""entrypoint for cnn training jobs."""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import uttt.ml.cnn_core as cnn_core
from uttt.paths import FILTERED_X_PATH, FILTERED_Y_PATH, MIXED_X_PATH, MIXED_Y_PATH



BATCH_SIZE = 4096
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42
NUM_WORKERS = 0


def main() -> None:
    """run the selected cnn training job."""
    if cnn_core.TRAINING_NAME == "mixed":
        x_path = MIXED_X_PATH
        y_path = MIXED_Y_PATH
    elif cnn_core.TRAINING_NAME == "filtered":
        x_path = FILTERED_X_PATH
        y_path = FILTERED_Y_PATH

    else:
        raise ValueError(f"unknown training_name: {cnn_core.TRAINING_NAME}")

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

    print(f"{cnn_core.TRAINING_NAME} training")
    print(
        f"train rows: {len(train_loader.dataset):,} | "
        f"val rows: {len(val_loader.dataset):,} | "
        f"test rows: {len(test_loader.dataset):,}"
    )
    print(
        f"settings: model={cnn_core.TRAIN_MODEL_OPTION}, epochs={cnn_core.TRAIN_EPOCHS}, "
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

    print(f"done: {cnn_core.TRAIN_MODEL_OPTION}")
    print(f"last: {result['model_path']}")
    print(f"best: {result['best_model_path']}")
    print(
        f"plots: "
        f"{result['train_plot_path']} | {result['val_plot_path']}"
    )

def test(model):
    cnn_core.TRAIN_MODEL_OPTION = model
    main()

if __name__ == "__main__":
    for model in ['A','B','C','D']:
        test(model)