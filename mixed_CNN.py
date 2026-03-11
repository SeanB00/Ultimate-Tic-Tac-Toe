"""Train CNN models on mixed NumPy dataset."""

from CNN_core import (
    DEFAULT_AUTO_RESUME,
    DEFAULT_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_SAVE_PLOTS,
    DEFAULT_USE_CUDA_IF_AVAILABLE,
    DEFAULT_USE_MEMMAP,
    build_npy_dataloaders,
    pick_device,
    train_supervised_model,
)

# Training configuration
X_PATH = "mixed_X_v1.npy"
Y_PATH = "mixed_y_v1.npy"

MODEL_OPTIONS = ["D"]
EPOCHS = 5
BATCH_SIZE = 2048
LR = 1e-3
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
SEED = 42
NUM_WORKERS = 0

USE_CUDA_IF_AVAILABLE = DEFAULT_USE_CUDA_IF_AVAILABLE
USE_MEMMAP = DEFAULT_USE_MEMMAP


def train_one_model(option, loaders, device):
    """Train one model option on the shared mixed-data split."""
    option = option.upper()

    print(f"\n>>> [{option}] Training start")
    result = train_supervised_model(
        model_option=option,
        device=device,
        train_loader=loaders.train_loader,
        val_loader=loaders.val_loader,
        test_loader=loaders.test_loader,
        epochs=EPOCHS,
        lr=LR,
        title_prefix=f"Mixed CNN {option}",
        log_prefix=f"[{option}] ",
    )

    print(f">>> [{option}] Done")
    print(f">>> [{option}] Model: {result['model_path']}")
    if DEFAULT_SAVE_PLOTS:
        print(f">>> [{option}] Plots: {result['train_plot_path']} | {result['val_plot_path']}")


def main():
    """Train each configured model option on mixed NumPy data."""
    device = pick_device(USE_CUDA_IF_AVAILABLE)

    loaders = build_npy_dataloaders(
        x_path=X_PATH,
        y_path=Y_PATH,
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        device=device,
        use_memmap=USE_MEMMAP,
    )

    print(
        f">>> Train rows: {loaders.train_size:,} | Val rows: {loaders.val_size:,} | "
        f"Test rows: {loaders.test_size:,}"
    )
    print(
        f">>> Settings: models={MODEL_OPTIONS}, epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, "
        f"split={TRAIN_RATIO:.2f}/{VAL_RATIO:.2f}/{TEST_RATIO:.2f}, "
        f"memmap={USE_MEMMAP}, auto_resume={DEFAULT_AUTO_RESUME}, "
        f"early_stop_patience={DEFAULT_EARLY_STOPPING_PATIENCE}, "
        f"early_stop_min_delta={DEFAULT_EARLY_STOPPING_MIN_DELTA}, "
        f"save_plots={DEFAULT_SAVE_PLOTS}"
    )

    for opt in MODEL_OPTIONS:
        train_one_model(opt, loaders, device)


if __name__ == "__main__":
    main()
