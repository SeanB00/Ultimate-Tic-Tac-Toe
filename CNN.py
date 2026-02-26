"""CNN training for Ultimate Tic-Tac-Toe value approximation.

This file merges the "good parts" of both versions you showed:
- 3-channel encoding: X-plane / O-plane / Empty-plane (fast, stable)
- LMDB random seek sampling (reduces cursor-walk bias; usually faster)
- Always-on symmetry augmentation (keeps your newer improvement)
- Single checkpoint file overwritten (like your run-based approach)
  storing: model_state, optimizer_state, step, history
- Loss plots saved to png (train + val) and updated during training
- Nicer logging, including sampler batch time + tries

LMDB value format: struct.pack("di", q_value(double), visits(int)).
"""

# --- OpenMP duplicate runtime workaround (Windows + torch/numpy/mkl)
# Put BEFORE importing torch/numpy in scripts that crash with libiomp5md.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import random
import struct
from dataclasses import dataclass

import numpy as np
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import hashing
from logic import UltimateTicTacToeGame


# ============================================================
# CONFIG (global variables instead of TrainConfig)
# ============================================================

LMDB_PATH = "fixed_qtable.lmdb"

BATCH_SIZE = 1024
VAL_SIZE = 2048
STEPS = 50_000
LR = 1e-4

LOG_EVERY = 100
VAL_EVERY = 500
SAVE_EVERY = 1000

OUT_DIR = "cnn_runs"
MODEL_OPTION = "A"  # A / B / C / D / E

MIN_COUNT = 1

# random LMDB sampling: how many key bits are actually used
KEY_BITS = 129

PREFER_CUDA = True


# ============================================================
# DEVICE
# ============================================================

def pick_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        print(">>> Using CUDA GPU")
        return torch.device("cuda")
    print(">>> Using CPU")
    return torch.device("cpu")


# ============================================================
# MODELS
# ============================================================

def act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    # Avoid disallowed/undesired activations by mapping them to safe alternatives
    if name == "gelu":
        # Map GELU to LeakyReLU to avoid using GELU while keeping smooth-ish nonlinearity
        return nn.LeakyReLU(0.05, inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "silu" or name == "swish":
        # Replace SiLU/Swish with ELU to avoid SiLU
        return nn.ELU(inplace=True)
    if name == "mish":
        # Replace Mish with ELU to avoid Mish
        return nn.ELU(inplace=True)
    if name == "elu":
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
    def __init__(self, in_ch: int = 1):
        super().__init__()
        # Stronger CNN: 3 conv blocks + bigger FC head
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            act("elu"),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            act("elu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 9 * 9, 512),
            act("elu"),
            nn.Linear(512, 128),
            act("prelu"),
            nn.Linear(128, 1),
            act("tanh"),  # output in [-1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


class ModelB(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),  # single conv, 64 filters
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
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


class ModelC(nn.Module):
    def __init__(self, in_ch: int = 1):
        super().__init__()
        # Two conv layers, moderate width
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
            act("hardswish"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


class ModelD(nn.Module):
    """Stride=3 model to capture each 3x3 mini-board as a "patch" with a single conv."""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        self.mb = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, stride=3),  # single conv, 128 filters
            act("relu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            act("leakyrelu"),
            nn.Linear(256, 64),
            act("relu"),
            nn.Linear(64, 1),
            act("tanh"),
        )

    def forward(self, x):
        x = self.mb(x)
        return self.fc(x).squeeze(-1)


class ModelE(nn.Module):
    """Small multi-conv + FC head (freestyle)."""

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


def build_model(option: str) -> nn.Module:
    option = option.upper()
    if option == "A":
        return ModelA(in_ch=1)
    if option == "B":
        return ModelB(in_ch=1)
    if option == "C":
        return ModelC(in_ch=1)
    if option == "D":
        return ModelD(in_ch=1)
    if option == "E":
        return ModelE(in_ch=1)
    raise ValueError("Unknown model option")


# ============================================================
# SAMPLER
# ============================================================

class LmdbSampler:
    def __init__(self, game: UltimateTicTacToeGame):
        self.game = game

        # readonly=True + lock=False is typical for pure reading
        self.env = lmdb.open(LMDB_PATH, readonly=True, lock=False, subdir=False, readahead=True)
        self.txn = self.env.begin(write=False)
        self.cursor = self.txn.cursor()

        self.unpack = struct.unpack
        self.value_size = struct.calcsize("di")

        if not self.cursor.first():
            raise RuntimeError("LMDB appears empty")
        self.key_len = len(self.cursor.key())

    def _random_seek(self):
        # sample within correct keyspace, seek to nearest key
        r = random.getrandbits(KEY_BITS)
        key = r.to_bytes(self.key_len, "big", signed=False)
        if not self.cursor.set_range(key):
            self.cursor.first()

    def sample_batch(self, batch_size: int):
        X = np.empty((batch_size, 1, 9, 9), dtype=np.float32)
        y = np.empty(batch_size, dtype=np.float32)

        accepted = 0
        tries = 0
        t0 = time.time()

        while accepted < batch_size:
            tries += 1

            # random seek each attempt (your previous working style)
            self._random_seek()

            k = self.cursor.key()
            v = self.cursor.value()

            if len(v) != self.value_size:
                continue

            q, visits = self.unpack("di", v)
            if visits < MIN_COUNT:
                continue

            state_int = int.from_bytes(k, "big", signed=False)
            board = np.array(hashing.decode_board_from_int(state_int), dtype=np.int8).reshape(9, 9)

            # symmetry augmentation (always)
            syms = self.game.all_symmetries_fast(board)
            board = syms[random.randrange(0, 8)]

            # single-channel encoding with values in {-1,0,1}
            X[accepted, 0] = board.astype(np.float32)

            y[accepted] = float(q)
            accepted += 1

        dt = time.time() - t0
        return X, y, tries, dt


# ============================================================
# TRAIN
# ============================================================

from typing import Optional

def _pretty_step_line(step: int, total: int, train_mse: float, val_mse: Optional[float], batch_dt: float, tries: int, sps: float):
    if val_mse is None:
        return (
            f"[Step {step:>6}/{total}] "
            f"Train MSE={train_mse:.6f} | "
            f"Sampler {batch_dt:.3f}s (tries={tries}) | "
            f"~{sps:,.0f} samples/s"
        )
    return (
        f"[Step {step:>6}/{total}] "
        f"Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f} | "
        f"Sampler {batch_dt:.3f}s (tries={tries}) | "
        f"~{sps:,.0f} samples/s"
    )


def train():
    device = pick_device(PREFER_CUDA)

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, f"model_{MODEL_OPTION}.pt")
    train_plot_path = os.path.join(OUT_DIR, f"train_loss_{MODEL_OPTION}.png")
    val_plot_path = os.path.join(OUT_DIR, f"val_loss_{MODEL_OPTION}.png")

    game = UltimateTicTacToeGame(q_table={}, training=False)
    sampler = LmdbSampler(game)

    model = build_model(MODEL_OPTION).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_step = 0
    history = {"train_loss": [], "val_loss": []}

    def save_checkpoint(step: int):
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "history": history,
            },
            model_path,
        )

    def save_plots():
        if history["train_loss"]:
            plt.figure(figsize=(10, 5))
            plt.plot(history["train_loss"], label="Train Loss")
            plt.legend()
            plt.title("UTTT CNN Training")
            plt.xlabel(f"Logged every {LOG_EVERY} steps")
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.savefig(train_plot_path)
            plt.close()

        if history["val_loss"]:
            plt.figure(figsize=(10, 5))
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.legend()
            plt.title("UTTT CNN Validation")
            plt.xlabel(f"Evaluated every {VAL_EVERY} steps")
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.savefig(val_plot_path)
            plt.close()

    # ----------------- RESUME -----------------
    if os.path.exists(model_path):
        print(">>> Loading existing checkpoint:", model_path)
        ckpt = torch.load(model_path, map_location=device)

        # support several key styles (your old script used model/optim)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("step", 0))
            history = ckpt.get("history", history)
        elif isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optim" in ckpt:
                optimizer.load_state_dict(ckpt["optim"])
            start_step = int(ckpt.get("step", 0))
        else:
            # raw state_dict
            model.load_state_dict(ckpt)
            start_step = 0

        print(f">>> Resumed from step {start_step}")
        save_plots()
    else:
        print(">>> Created new model")

    # ----------------- VALIDATION SET -----------------
    print(">>> Building validation set...")
    Xv_np, yv_np, vtries, vdt = sampler.sample_batch(VAL_SIZE)
    Xv = torch.from_numpy(Xv_np).to(device)
    yv = torch.from_numpy(yv_np).to(device)
    print(f">>> Val batch ready in {vdt:.3f}s (tries={vtries})")

    # ----------------- TRAIN LOOP -----------------
    last_log_t = time.time()
    last_log_samples = 0

    for step in range(start_step + 1, STEPS + 1):
        X_np, y_np, tries, batch_dt = sampler.sample_batch(BATCH_SIZE)

        X = torch.from_numpy(X_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        pred = model(X)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging cadence like your old script
        if step % LOG_EVERY == 0:
            history["train_loss"].append(float(loss.item()))

            now = time.time()
            last_log_samples += BATCH_SIZE * LOG_EVERY
            elapsed = now - last_log_t
            sps = (BATCH_SIZE * LOG_EVERY) / max(elapsed, 1e-9)
            last_log_t = now

            # optional val info (printed on val steps)
            line = _pretty_step_line(step, STEPS, float(loss.item()), None, batch_dt, tries, sps)
            print(line)

        if step % VAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = float(F.mse_loss(val_pred, yv).item())
            model.train()

            history["val_loss"].append(val_loss)
            print(f"   ↳ Validation MSE={val_loss:.6f}")

        # save cadence: single file overwritten (your run-style)
        if step % SAVE_EVERY == 0:
            save_checkpoint(step)
            save_plots()
            print(f"    Saved checkpoint (step {step}) -> {model_path}")

    # final save
    save_checkpoint(STEPS)
    save_plots()

    print("\n>>> Training finished")
    print(">>> Model:", model_path)
    print(">>> Plots:", train_plot_path, "|", val_plot_path)


def run_experiments():
    global MODEL_OPTION, STEPS, LR, MIN_COUNT

    # Define a schedule: tuple(model, steps, lr, min_count)
    schedule = [
        ("A", 100_000, 1e-4, 1),     # Best model A, longest run
        ("B", 60_000, 5e-4, 1),      # Different LR
        ("C", 50_000, 1e-4, 1),      # Freestyle C standard run
        ("D", 50_000, 2e-4, 1),      # Keep stride model with adjusted LR
        ("E", 30_000, 1e-4, 2),      # Shortest and MIN_COUNT=2 as requested
    ]

    for m, steps, lr, mc in schedule:
        MODEL_OPTION = m
        STEPS = steps
        LR = lr
        MIN_COUNT = mc
        print(f"\n=== Running model {m} | steps={steps:,} | lr={lr} | min_count={mc} ===")
        train()


if __name__ == "__main__":
    run_experiments()

