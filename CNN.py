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
# CONFIG
# ============================================================

@dataclass
class TrainConfig:
    lmdb_path: str = "fixed_qtable.lmdb"

    batch_size: int = 1024
    val_size: int = 2048
    steps: int = 50_000
    lr: float = 1e-4

    log_every: int = 100
    val_every: int = 500
    save_every: int = 1000

    out_dir: str = "cnn_runs"
    model_option: str = "A"   # A / B / C / D / E

    min_count: int = 1

    # random LMDB sampling: how many key bits are actually used
    key_bits: int = 129

    prefer_cuda: bool = True


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
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class ModelA(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1),
            act("relu"),
            nn.Conv2d(64, 64, 3, padding=1),
            act("relu"),
            nn.Conv2d(64, 64, 3, padding=1),
            act("relu"),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            act("relu"),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ModelB(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        layers = []
        for i in range(6):
            layers.append(nn.Conv2d(in_ch if i == 0 else 96, 96, 3, padding=1))
            layers.append(act("gelu"))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 9 * 9, 512),
            act("gelu"),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


class ModelC(nn.Module):
    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, 64, 3, padding=1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                act("relu"),
                nn.Conv2d(64, 64, 3, padding=1),
            )
            for _ in range(6)
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 128),
            act("relu"),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = act("relu")(self.stem(x))
        for blk in self.blocks:
            x = act("relu")(x + blk(x))
        return self.head(x).squeeze(-1)


class ModelD(nn.Module):
    """Stride=3 model to capture each 3x3 mini-board as a "patch"."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.mb = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=3),
            act("relu"),
            nn.Conv2d(64, 64, 3, padding=1),
            act("relu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            act("relu"),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.mb(x)
        return self.fc(x).squeeze(-1)


class ModelE(nn.Module):
    """Pure MLP baseline."""

    def __init__(self, in_ch: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(81 * in_ch, 512),
            act("tanh"),
            nn.Linear(512, 512),
            act("tanh"),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(option: str) -> nn.Module:
    option = option.upper()
    if option == "A":
        return ModelA(in_ch=3)
    if option == "B":
        return ModelB(in_ch=3)
    if option == "C":
        return ModelC(in_ch=3)
    if option == "D":
        return ModelD(in_ch=3)
    if option == "E":
        return ModelE(in_ch=3)
    raise ValueError("Unknown model option")


# ============================================================
# SAMPLER
# ============================================================

class LmdbSampler:
    def __init__(self, cfg: TrainConfig, game: UltimateTicTacToeGame):
        self.cfg = cfg
        self.game = game

        # readonly=True + lock=False is typical for pure reading
        self.env = lmdb.open(cfg.lmdb_path, readonly=True, lock=False, subdir=False, readahead=True)
        self.txn = self.env.begin(write=False)
        self.cursor = self.txn.cursor()

        self.unpack = struct.unpack
        self.value_size = struct.calcsize("di")

        if not self.cursor.first():
            raise RuntimeError("LMDB appears empty")
        self.key_len = len(self.cursor.key())

    def _random_seek(self):
        # sample within correct keyspace, seek to nearest key
        r = random.getrandbits(self.cfg.key_bits)
        key = r.to_bytes(self.key_len, "big", signed=False)
        if not self.cursor.set_range(key):
            self.cursor.first()

    def sample_batch(self, batch_size: int):
        X = np.empty((batch_size, 3, 9, 9), dtype=np.float32)
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
            if visits < self.cfg.min_count:
                continue

            state_int = int.from_bytes(k, "big", signed=False)
            board = np.array(hashing.decode_board_from_int(state_int), dtype=np.int8).reshape(9, 9)

            # symmetry augmentation (always)
            syms = self.game.all_symmetries_fast(board)
            board = syms[random.randrange(0, 8)]

            # 3-channel encoding
            x_plane = (board == 1).astype(np.float32)
            o_plane = (board == -1).astype(np.float32)
            e_plane = (board == 0).astype(np.float32)
            X[accepted] = np.stack([x_plane, o_plane, e_plane], axis=0)

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


def train(cfg: TrainConfig):
    device = pick_device(cfg.prefer_cuda)

    os.makedirs(cfg.out_dir, exist_ok=True)
    model_path = os.path.join(cfg.out_dir, f"model_{cfg.model_option}.pt")
    train_plot_path = os.path.join(cfg.out_dir, f"train_loss_{cfg.model_option}.png")
    val_plot_path = os.path.join(cfg.out_dir, f"val_loss_{cfg.model_option}.png")

    game = UltimateTicTacToeGame(q_table={}, training=False)
    sampler = LmdbSampler(cfg, game)

    model = build_model(cfg.model_option).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

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
            plt.xlabel(f"Logged every {cfg.log_every} steps")
            plt.ylabel("MSE")
            plt.tight_layout()
            plt.savefig(train_plot_path)
            plt.close()

        if history["val_loss"]:
            plt.figure(figsize=(10, 5))
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.legend()
            plt.title("UTTT CNN Validation")
            plt.xlabel(f"Evaluated every {cfg.val_every} steps")
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
    Xv_np, yv_np, vtries, vdt = sampler.sample_batch(cfg.val_size)
    Xv = torch.from_numpy(Xv_np).to(device)
    yv = torch.from_numpy(yv_np).to(device)
    print(f">>> Val batch ready in {vdt:.3f}s (tries={vtries})")

    # ----------------- TRAIN LOOP -----------------
    last_log_t = time.time()
    last_log_samples = 0

    for step in range(start_step + 1, cfg.steps + 1):
        X_np, y_np, tries, batch_dt = sampler.sample_batch(cfg.batch_size)

        X = torch.from_numpy(X_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        pred = model(X)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging cadence like your old script
        if step % cfg.log_every == 0:
            history["train_loss"].append(float(loss.item()))

            now = time.time()
            last_log_samples += cfg.batch_size * cfg.log_every
            elapsed = now - last_log_t
            sps = (cfg.batch_size * cfg.log_every) / max(elapsed, 1e-9)
            last_log_t = now

            # optional val info (printed on val steps)
            line = _pretty_step_line(step, cfg.steps, float(loss.item()), None, batch_dt, tries, sps)
            print(line)

        if step % cfg.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = float(F.mse_loss(val_pred, yv).item())
            model.train()

            history["val_loss"].append(val_loss)
            print(f"   ↳ Validation MSE={val_loss:.6f}")

        # save cadence: single file overwritten (your run-style)
        if step % cfg.save_every == 0:
            save_checkpoint(step)
            save_plots()
            print(f"    Saved checkpoint (step {step}) -> {model_path}")

    # final save
    save_checkpoint(cfg.steps)
    save_plots()

    print("\n>>> Training finished")
    print(">>> Model:", model_path)
    print(">>> Plots:", train_plot_path, "|", val_plot_path)


if __name__ == "__main__":
    cfg = TrainConfig(model_option="A")
    train(cfg)
    cfg = TrainConfig(model_option="B")
    train(cfg)
    cfg = TrainConfig(model_option="C", lr=1e-5)
    train(cfg)
    cfg = TrainConfig(model_option="D")
    train(cfg)
    cfg = TrainConfig(model_option="E", lr=1e-5,steps=30_000)
    train(cfg)

