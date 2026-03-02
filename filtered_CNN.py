"""
Epoch-based CNN training on filtered/expanded NumPy data.

Expected files from `filter_lmdb_to_numpy.py`:
- X file: (N, 9, 9), float32
- y file: (N,), float32

This script is intentionally simple:
1) Load X/y
2) Train/val split
3) Epoch loop
4) Save best and last checkpoints
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# CONFIG (edit here, no argparse)
# ============================================================
X_PATH = "expanded_X_min2.npy"
Y_PATH = "expanded_y_min2.npy"
OUT_DIR = "filtered_cnn_runs"

EPOCHS = 25
BATCH_SIZE = 1024
LR = 1e-4
VAL_RATIO = 0.10
SEED = 42
NUM_WORKERS = 0
LOG_EVERY = 100
USE_CUDA_IF_AVAILABLE = True


def pick_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        print(">>> Using CUDA GPU")
        return torch.device("cuda")
    print(">>> Using CPU")
    return torch.device("cpu")


def act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "softplus":
        return nn.Softplus()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class FilteredModelE(nn.Module):
    """
    Similar to ModelE in your CNN.py.
    Input is single-channel 9x9 board.
    """

    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            act("relu"),
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            act("elu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 9 * 9, 128),
            act("softplus"),
            nn.Linear(128, 64),
            act("relu"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


class NpyDataset(Dataset):
    """
    X is expected as (N, 9, 9).
    We add channel dimension on-the-fly -> (1, 9, 9).
    """

    def __init__(self, x_path: str, y_path: str):
        self.X = np.load(x_path)  # full RAM load (simple path)
        self.y = np.load(y_path)

        if self.X.ndim != 3 or self.X.shape[1:] != (9, 9):
            raise ValueError(f"Unexpected X shape: {self.X.shape}, expected (N,9,9)")
        if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError(f"Unexpected y shape: {self.y.shape}, expected ({self.X.shape[0]},)")

        print(f">>> Dataset loaded: N={self.X.shape[0]:,}")
        print(f">>> X dtype/shape: {self.X.dtype} / {self.X.shape}")
        print(f">>> y dtype/shape: {self.y.dtype} / {self.y.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32)
        x = np.expand_dims(x, axis=0)  # (1, 9, 9)
        y = np.float32(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb, reduction="sum")
            total_loss += float(loss.item())
            total_count += int(yb.numel())
    if total_count == 0:
        return float("inf")
    return total_loss / total_count


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = pick_device(USE_CUDA_IF_AVAILABLE)

    dataset = NpyDataset(X_PATH, Y_PATH)
    n = len(dataset)
    if n < 10:
        raise RuntimeError(f"Dataset too small: N={n}")

    # Reproducible split
    rng = np.random.default_rng(SEED)
    indices = np.arange(n)
    rng.shuffle(indices)

    val_n = max(1, int(n * VAL_RATIO))
    train_idx = indices[val_n:]
    val_idx = indices[:val_n]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_mem,
        drop_last=False,
    )

    model = FilteredModelE(in_ch=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    history = {"train_mse": [], "val_mse": []}
    best_path = os.path.join(OUT_DIR, "filtered_model_best.pt")
    last_path = os.path.join(OUT_DIR, "filtered_model_last.pt")
    history_path = os.path.join(OUT_DIR, "filtered_history.json")

    print(">>> Training start")
    print(f">>> Train rows: {len(train_ds):,} | Val rows: {len(val_ds):,}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_count = 0
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss = F.mse_loss(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_n = int(yb.numel())
            running_loss += float(loss.item()) * batch_n
            running_count += batch_n

            if LOG_EVERY > 0 and step % LOG_EVERY == 0:
                train_mse_so_far = running_loss / max(running_count, 1)
                print(
                    f"[epoch {epoch:>3}/{EPOCHS}] step={step:>6} "
                    f"train_mse={train_mse_so_far:.6f}"
                )

        train_mse = running_loss / max(running_count, 1)
        val_mse = evaluate(model, val_loader, device)
        history["train_mse"].append(float(train_mse))
        history["val_mse"].append(float(val_mse))

        # Save last checkpoint every epoch.
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
            },
            last_path,
        )

        # Save best checkpoint by validation MSE.
        best_tag = ""
        if val_mse < best_val:
            best_val = val_mse
            best_tag = " [BEST]"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_mse": float(train_mse),
                    "val_mse": float(val_mse),
                },
                best_path,
            )

        dt = time.time() - t0
        print(
            f"[epoch {epoch:>3}/{EPOCHS}] "
            f"train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
            f"time={dt:.2f}s{best_tag}"
        )

        with open(history_path, "w", encoding="ascii") as f:
            json.dump(history, f, indent=2)

    print(">>> Training done")
    print(f">>> Best checkpoint: {best_path}")
    print(f">>> Last checkpoint: {last_path}")
    print(f">>> History: {history_path}")


if __name__ == "__main__":
    main()
