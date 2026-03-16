"""Shared CNN components for Ultimate Tic-Tac-Toe training/inference."""

# OpenMP duplicate runtime workaround (Windows + torch/numpy/mkl)
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from uttt.paths import CNN_RUNS_DIR, ensure_project_dirs

DEFAULT_USE_MEMMAP = False
DEFAULT_LOG_EVERY = 100
DEFAULT_AUTO_RESUME = True
DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_EARLY_STOPPING_MIN_DELTA = 5e-5

TRAINING_NAME = "Mixed"
TRAIN_MODEL_OPTION = "D"
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-4

def pick_device():
    """Execute pick device."""
    if torch.cuda.is_available():
        print(">>> Using CUDA GPU")
        return torch.device("cuda")

    print(">>> Using CPU")
    return torch.device("cpu")

def act(name):
    """Execute act."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.LeakyReLU(0.05, inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name in {"silu", "swish", "mish", "elu"}:
        return nn.ELU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    if name == "prelu":
        return nn.PReLU()
    if name == "hardswish":
        return nn.Hardswish()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(f"Unknown activation: {name}")

class ModelA(nn.Module):
    def __init__(self, in_ch = 1):
        """Initialize the ModelA instance."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            act("elu"),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            act("relu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 512),
            act("elu"),
            nn.Linear(512, 128),
            act("prelu"),
            nn.Linear(128, 1),
            act("tanh"),
        )

    def forward(self, x):
        """Execute forward."""
        x = self.conv(x)
        return self.fc(x).squeeze(-1)

class ModelB(nn.Module):
    def __init__(self, in_ch = 1):
        """Initialize the ModelB instance."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            act("elu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            act("relu"),
            nn.Linear(256, 64),
            act("leakyrelu"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        """Execute forward."""
        x = self.conv(x)
        return self.fc(x).squeeze(-1)

class ModelC(nn.Module):
    def __init__(self, in_ch = 1):
        """Initialize the ModelC instance."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            act("elu"),
            nn.Conv2d(96, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            act("relu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 9 * 9, 256),
            act("leakyrelu"),
            nn.Linear(256, 64),
            act("elu"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        """Execute forward."""
        x = self.conv(x)
        return self.fc(x).squeeze(-1)

class ModelD(nn.Module):
    def __init__(self, in_ch = 1):
        """Initialize the ModelD instance."""
        super().__init__()
        self.mb = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, stride=3),
            nn.BatchNorm2d(128),
            act("elu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            act("leakyrelu"),
            nn.Linear(256, 64),
            act("elu"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        """Execute forward."""
        x = self.mb(x)
        return self.fc(x).squeeze(-1)

class ModelE(nn.Module):
    def __init__(self, in_ch = 1):
        """Initialize the ModelE instance."""
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
            act("prelu"),
            nn.Linear(128, 64),
            act("elu"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        """Execute forward."""
        x = self.conv(x)
        return self.fc(x).squeeze(-1)

def build_model(option, in_ch = 1):
    """Build model."""
    option = option.upper()
    if option == "A":
        return ModelA(in_ch=in_ch)
    if option == "B":
        return ModelB(in_ch=in_ch)
    if option == "C":
        return ModelC(in_ch=in_ch)
    if option == "D":
        return ModelD(in_ch=in_ch)
    if option == "E":
        return ModelE(in_ch=in_ch)
    raise ValueError(f"Unknown model option: {option}")


def load_trained_model(
    model_option,
):
    """
    Load a trained model, preferring the best checkpoint.
    """
    device = pick_device()
    print(">>> DEVICE:", device)

    model = build_model(model_option)
    paths = model_artifact_paths(model_option)
    path = paths["best_model_path"] if os.path.isfile(paths["best_model_path"]) else paths["model_path"]
    print(">>> Loading model from:", path)
    obj = torch.load(path, map_location=device)
    model.load_state_dict(obj["model_state"])
    step = obj.get("step")
    if step is not None:
        print(">>> Loaded checkpoint step:", step)

    model.to(device)
    model.eval()
    return model, device

class NpyDataset(Dataset):
    """Loads (N,9,9) board arrays and scalar targets from NumPy files."""

    def __init__(self, x_path, y_path, use_memmap = True):
        """Initialize the NpyDataset instance."""
        mmap_mode = "r" if use_memmap else None
        self.X = np.load(x_path, mmap_mode=mmap_mode)
        self.y = np.load(y_path, mmap_mode=mmap_mode)

        if self.X.ndim != 3 or self.X.shape[1:] != (9, 9):
            raise ValueError(f"Unexpected X shape: {self.X.shape}, expected (N, 9, 9)")
        if self.y.ndim != 1 or self.y.shape[0] != self.X.shape[0]:
            raise ValueError(f"Unexpected y shape: {self.y.shape}, expected ({self.X.shape[0]},)")

        print(f">>> Dataset loaded: N={self.X.shape[0]:,}")
        print(f">>> X dtype/shape: {self.X.dtype} / {self.X.shape}")
        print(f">>> y dtype/shape: {self.y.dtype} / {self.y.shape}")

    def __len__(self):
        """Return the number of items."""
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        """Return an item for the given index."""
        x = np.array(self.X[idx], dtype=np.float32, copy=True)
        x = np.expand_dims(x, axis=0)
        y = np.float32(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

def build_npy_dataloaders(
    x_path,
    y_path,
    batch_size,
    val_ratio,
    test_ratio,
    seed,
    num_workers,
    device,
    use_memmap = True,
):
    """Build npy dataloaders."""
    dataset = NpyDataset(x_path, y_path, use_memmap=use_memmap)
    n = len(dataset)
    if n < 10:
        raise RuntimeError(f"Dataset too small: N={n}")

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"val_ratio + test_ratio must be less than 1, got {val_ratio + test_ratio}"
        )

    val_n = max(1, int(n * val_ratio))
    test_n = max(1, int(n * test_ratio))
    if val_n + test_n >= n:
        raise RuntimeError(f"Split too large for dataset size N={n}")

    rng = np.random.default_rng(seed)
    indices = np.arange(n, dtype=np.int64)
    rng.shuffle(indices)

    val_idx = indices[:val_n]
    test_idx = indices[val_n:val_n + test_n]
    train_idx = indices[val_n + test_n:]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader

def evaluate_mse(model, loader, device):
    """Execute evaluate mse."""
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

def save_mse_plots(
    history,
    train_plot_path,
    val_plot_path,
):
    """Save mse plots."""
    plot_title = f"{TRAINING_NAME} {TRAIN_MODEL_OPTION.upper()}"

    if history.get("train_mse"):
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_mse"], label="Train MSE")
        plt.legend()
        plt.title(f"{plot_title} Training")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(train_plot_path)
        plt.close()

    if history.get("val_mse"):
        plt.figure(figsize=(10, 5))
        plt.plot(history["val_mse"], label="Validation MSE")
        plt.legend()
        plt.title(f"{plot_title} Validation")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.tight_layout()
        plt.savefig(val_plot_path)
        plt.close()

def model_artifact_paths(model_option):
    """Return model and plot output paths for a model option."""
    model_option = model_option.upper()
    run_dir = os.fspath(CNN_RUNS_DIR)
    return {
        "model_path": os.path.join(run_dir, f"model_{model_option}.pt"),
        "best_model_path": os.path.join(run_dir, f"model_{model_option}_best.pt"),
        "train_plot_path": os.path.join(run_dir, f"train_loss_{model_option}.png"),
        "val_plot_path": os.path.join(run_dir, f"val_loss_{model_option}.png"),
    }

def train_supervised_model(
    *,
    device,
    train_loader,
    val_loader,
    test_loader,
):
    """Train supervised model."""
    model_option = TRAIN_MODEL_OPTION.upper()
    epochs = TRAIN_EPOCHS
    lr = TRAIN_LR
    ensure_project_dirs()
    paths = model_artifact_paths(model_option)
    os.makedirs(os.path.dirname(paths["model_path"]) or ".", exist_ok=True)

    model = build_model(model_option).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs_no_improve = 0
    start_epoch = 1
    best_val_mse = float("inf")
    best_epoch = 0
    history = {"train_mse": [], "val_mse": []}

    def load_history_from_dict(data):
        """Load history from dict."""
        if not isinstance(data, dict):
            return
        # Accept both current keys (train_mse/val_mse) and legacy keys (train_loss/val_loss).
        history["train_mse"] = list(data.get("train_mse", data.get("train_loss", [])))
        history["val_mse"] = list(data.get("val_mse", data.get("val_loss", [])))

    def best_stats_from_history():
        """Return best val_mse, its epoch index, and no-improve streak since then."""
        vals = history["val_mse"]
        if not vals:
            return float("inf"), 0, 0
        replay_best = float("inf")
        replay_best_epoch = 0
        replay_no_improve = 0
        for epoch_idx, raw_val in enumerate(vals, start=1):
            val = float(raw_val)
            if val < (replay_best - DEFAULT_EARLY_STOPPING_MIN_DELTA):
                replay_best = val
                replay_best_epoch = epoch_idx
                replay_no_improve = 0
            else:
                replay_no_improve += 1
        return replay_best, replay_best_epoch, replay_no_improve

    if DEFAULT_AUTO_RESUME and os.path.isfile(paths["model_path"]):
        print(f">>> Found checkpoint: {paths['model_path']}")
        ckpt = torch.load(paths["model_path"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])


        # Prefer step-based schema;
        # keep epoch fallback for older checkpoints.
        last_step = int(ckpt.get("step", ckpt.get("epoch", 0)))
        start_epoch = last_step + 1
        load_history_from_dict(ckpt.get("history"))
        saved_best_val_mse = ckpt.get("best_val_mse")
        saved_best_epoch = ckpt.get("best_epoch")
        if saved_best_val_mse is not None and saved_best_epoch is not None:
            best_val_mse = float(saved_best_val_mse)
            best_epoch = int(saved_best_epoch)
            completed_epochs = len(history["val_mse"]) or last_step
            epochs_no_improve = max(0, completed_epochs - best_epoch)
        else:
            best_val_mse, best_epoch, epochs_no_improve = best_stats_from_history()

        print(
            f">>> Resumed at epoch {start_epoch} "
            f"(last_step={last_step}, best_epoch={best_epoch}, no_improve={epochs_no_improve})"
        )

    save_mse_plots(
        history,
        paths["train_plot_path"],
        paths["val_plot_path"],
    )

    if start_epoch > epochs:
        print(
            f">>> No training needed: checkpoint already at epoch {start_epoch - 1}, "
            f"EPOCHS={epochs}."
        )
    else:
        for epoch in range(start_epoch, epochs + 1):
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

                if DEFAULT_LOG_EVERY > 0 and step % DEFAULT_LOG_EVERY == 0:
                    train_mse_so_far = running_loss / max(running_count, 1)
                    print(
                        f"[epoch {epoch:>3}/{epochs}] "
                        f"step={step:>6} train_mse={train_mse_so_far:.6f}"
                    )

            train_mse = running_loss / max(running_count, 1)
            val_mse = evaluate_mse(model, val_loader, device)
            history["train_mse"].append(float(train_mse))
            history["val_mse"].append(float(val_mse))

            improved_tag = ""
            if val_mse < (best_val_mse - DEFAULT_EARLY_STOPPING_MIN_DELTA):
                best_val_mse = float(val_mse)
                best_epoch = epoch
                epochs_no_improve = 0
                improved_tag = " [BEST]"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "step": int(epoch),
                        "train_mse": float(train_mse),
                        "val_mse": float(val_mse),
                        "history": history,
                        "best_epoch": int(best_epoch),
                        "best_val_mse": float(best_val_mse),
                    },
                    paths["best_model_path"],
                )
            else:
                epochs_no_improve += 1

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": int(epoch),
                    "train_mse": float(train_mse),
                    "val_mse": float(val_mse),
                    "history": history,
                    "best_epoch": int(best_epoch),
                    "best_val_mse": float(best_val_mse),
                },
                paths["model_path"],
            )

            save_mse_plots(
                history,
                paths["train_plot_path"],
                paths["val_plot_path"],
            )

            dt = time.time() - t0
            print(
                f"[epoch {epoch:>3}/{epochs}] "
                f"train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
                f"best_val_mse={best_val_mse:.6f} best_epoch={best_epoch} "
                f"time={dt:.2f}s{improved_tag} "
                f"no_improve={epochs_no_improve}/{DEFAULT_EARLY_STOPPING_PATIENCE}"
            )

            if 0 < DEFAULT_EARLY_STOPPING_PATIENCE <= epochs_no_improve:
                print(
                    f">>> Early stopping at epoch {epoch}: "
                    f"no validation improvement for {epochs_no_improve} epochs."
                )
                break

    model.eval()
    last_test_mse = evaluate_mse(model, test_loader, device)
    best_test_mse = last_test_mse
    if os.path.isfile(paths["best_model_path"]):
        best_obj = torch.load(paths["best_model_path"], map_location=device)
        best_model = build_model(model_option).to(device)
        best_model.load_state_dict(best_obj["model_state"])
        best_model.eval()
        best_test_mse = evaluate_mse(best_model, test_loader, device)

    print(f">>> Test MSE (last): {last_test_mse:.6f}")
    print(f">>> Test MSE (best): {best_test_mse:.6f}")

    history["test_mse_last"] = float(last_test_mse)
    history["test_mse_best"] = float(best_test_mse)

    return {
        "history": history,
        "model_path": paths["model_path"],
        "best_model_path": paths["best_model_path"],
        "train_plot_path": paths["train_plot_path"],
        "val_plot_path": paths["val_plot_path"],
    }
