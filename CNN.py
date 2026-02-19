# cnn_train.py
import os
import time
import random
import struct
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import lmdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import hashing
from logic import UltimateTicTacToeGame


# ============================================================
# ============================ CONFIG =========================
# ============================================================

@dataclass
class TrainConfig:
    lmdb_path: str = "fixed_qtable.lmdb"

    batch_size: int = 256
    val_size: int = 2048
    steps: int = 20_000
    lr: float = 1e-4

    log_every: int = 100
    val_every: int = 500
    save_every: int = 2000

    out_dir: str = "cnn_runs"
    model_option: str = "A"   # A / B / C / D / E

    min_count: int = 1

    prefer_cuda: bool = True


# ============================================================
# ======================= DEVICE ==============================
# ============================================================

def pick_device(prefer_cuda=True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# ======================== MODELS =============================
# ============================================================

def act(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError


class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            act("relu"),
            nn.Conv2d(64, 64, 3, padding=1),
            act("relu"),
            nn.Conv2d(64, 64, 3, padding=1),
            act("relu"),
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 256),
            act("relu"),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for _ in range(6):
            layers.append(nn.Conv2d(1 if len(layers)==0 else 96, 96, 3, padding=1))
            layers.append(act("gelu"))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 9 * 9, 512),
            act("gelu"),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze(-1)


class ModelC(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 64, 3, padding=1)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                act("relu"),
                nn.Conv2d(64, 64, 3, padding=1)
            ) for _ in range(6)
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, 128),
            act("relu"),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = act("relu")(self.stem(x))
        for block in self.blocks:
            x = act("relu")(x + block(x))
        return self.head(x).squeeze(-1)


class ModelD(nn.Module):
    # stride=3 to capture mini-boards
    def __init__(self):
        super().__init__()
        self.mb = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=3),
            act("relu"),
            nn.Conv2d(64, 64, 3, padding=1),
            act("relu"),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            act("relu"),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.mb(x)
        return self.fc(x).squeeze(-1)


class ModelE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(81, 512),
            act("tanh"),
            nn.Linear(512, 512),
            act("tanh"),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_model(option):
    if option == "A":
        return ModelA()
    if option == "B":
        return ModelB()
    if option == "C":
        return ModelC()
    if option == "D":
        return ModelD()
    if option == "E":
        return ModelE()
    raise ValueError("Unknown model option")


# ============================================================
# ======================== SAMPLER ============================
# ============================================================

class LmdbSampler:
    def __init__(self, cfg, game):
        self.cfg = cfg
        self.game = game
        self.env = lmdb.open(cfg.lmdb_path, readonly=True, lock=False, subdir=False)
        self.txn = self.env.begin()
        self.cursor = self.txn.cursor()
        self.unpack = struct.unpack
        self.value_size = struct.calcsize("di")

    def sample_batch(self, batch_size):
        X = np.zeros((batch_size, 1, 9, 9), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.float32)

        accepted = 0
        while accepted < batch_size:
            if not self.cursor.next():
                self.cursor.first()

            k = self.cursor.key()
            v = self.cursor.value()

            if len(v) != self.value_size:
                continue

            q, visits = self.unpack("di", v)
            if visits < self.cfg.min_count:
                continue

            state_int = int.from_bytes(k, "big", signed=False)
            board = np.array(hashing.decode_board_from_int(state_int)).reshape(9, 9)

            # ALWAYS apply random symmetry
            syms = self.game.all_symmetries_fast(board)
            board = syms[random.randrange(0, 8)]

            X[accepted, 0] = board
            y[accepted] = q
            accepted += 1

        return X, y


# ============================================================
# ============================ TRAIN ==========================
# ============================================================

def train(cfg):
    device = pick_device(cfg.prefer_cuda)
    print("DEVICE:", device)

    game = UltimateTicTacToeGame(q_table={}, training=False)
    sampler = LmdbSampler(cfg, game)

    model = build_model(cfg.model_option).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"train_loss": [], "val_loss": []}

    Xv_np, yv_np = sampler.sample_batch(cfg.val_size)
    Xv = torch.tensor(Xv_np).to(device)
    yv = torch.tensor(yv_np).to(device)

    for step in range(1, cfg.steps + 1):
        X_np, y_np = sampler.sample_batch(cfg.batch_size)

        X = torch.tensor(X_np).to(device)
        y = torch.tensor(y_np).to(device)

        pred = model(X)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.log_every == 0:
            print(f"[{step}] train_mse={loss.item():.6f}")
            history["train_loss"].append(loss.item())

        if step % cfg.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(Xv)
                val_loss = F.mse_loss(val_pred, yv).item()
            print(f"    VAL mse={val_loss:.6f}")
            history["val_loss"].append(val_loss)
            model.train()

    os.makedirs(cfg.out_dir, exist_ok=True)

    model_path = os.path.join(cfg.out_dir, f"model_{cfg.model_option}.pt")
    torch.save(model.state_dict(), model_path)

    plt.plot(history["train_loss"])
    plt.title("Train Loss")
    plt.savefig(os.path.join(cfg.out_dir, f"train_loss_{cfg.model_option}.png"))
    plt.close()

    plt.plot(history["val_loss"])
    plt.title("Validation Loss")
    plt.savefig(os.path.join(cfg.out_dir, f"val_loss_{cfg.model_option}.png"))
    plt.close()

    print("Saved model to:", model_path)
    print("Training complete.")


if __name__ == "__main__":
    cfg = TrainConfig(model_option="A")
    train(cfg)
