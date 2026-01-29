# train_cnn_from_lmdb_verified.py

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import hashing
from lmdb_qtable import LMDBQTable


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")


# ============================================================
# UTTT SYMMETRY
# ============================================================

def apply_uttt_symmetry(board9, k):
    def R90(x): return np.rot90(x, 1)
    def R180(x): return np.rot90(x, 2)
    def R270(x): return np.rot90(x, 3)
    def FLIP_LR(x): return np.fliplr(x)
    def FLIP_UD(x): return np.flipud(x)
    def DIAG(x): return x.T
    def ADIAG(x): return R180(x).T

    funcs = [lambda x: x, R90, R180, R270, FLIP_LR, FLIP_UD, DIAG, ADIAG]
    f = funcs[k]

    blk = board9.reshape(3, 3, 3, 3)
    big = blk.reshape(3, 3, 9)
    big = f(big)
    blk = big.reshape(3, 3, 3, 3)

    out = np.empty_like(blk)
    for i in range(3):
        for j in range(3):
            out[i, j] = f(blk[i, j])

    return out.reshape(9, 9)


# ============================================================
# LMDB SAMPLER
# ============================================================

class LMDBSampler:
    def __init__(self, lmdb_path):
        self.qtable = LMDBQTable(lmdb_path)
        self.env = self.qtable.env
        self.txn = self.env.begin(write=False)
        self.cursor = self.txn.cursor()

    def sample_batch(self, batch_size):
        X = np.empty((batch_size, 9, 9), dtype=np.int8)
        Y = np.empty(batch_size, dtype=np.float32)

        i = 0
        while i < batch_size:
            r = random.getrandbits(256)
            key = r.to_bytes(32, "big")
            if not self.cursor.set_range(key):
                self.cursor.first()

            state_int = int.from_bytes(self.cursor.key(), "big")
            val, _ = self.qtable._decode(self.cursor.value())

            board = hashing.decode_board_from_int(state_int)
            X[i] = np.asarray(board, dtype=np.int8).reshape(9, 9)
            Y[i] = val
            i += 1

        return X, Y


# ============================================================
# MODEL (RESIDUAL VALUE CNN)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        r = x
        x = F.relu(self.c1(x))
        x = self.c2(x)
        return F.relu(x + r)


class ValueCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (B,1,9,9)
        self.inp = nn.Conv2d(1, 64, 3, padding=1)

        # Deep reasoning trunk
        self.res = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])

        # Value head
        self.out = nn.Conv2d(64, 1, 1)
        self.fc = nn.Linear(81, 1)

    def forward(self, x):
        x = F.relu(self.inp(x))
        x = self.res(x)
        x = self.out(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze(1)


# ============================================================
# SAVE / LOAD
# ============================================================

MODEL_PATH = "cnn_value_model.pt"

def load_or_create(device):
    model = ValueCNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        step = ckpt["step"]
        samples = ckpt["samples"]
        print(f"Resumed from step {step}, samples {samples}")
    else:
        step = 0
        samples = 0
        print("Starting fresh model")

    return model, optim, step, samples


def save(model, optim, step, samples):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "samples": samples,
        },
        MODEL_PATH,
    )


# ============================================================
# TRAINING WITH VERIFICATION
# ============================================================

def train(
    lmdb="fixed_qtable.lmdb",
    batch_size=1024,
    total_steps=20_000,
    log_every=100,
    val_every=500,
    val_batch=2048,
):
    device = get_device()
    sampler = LMDBSampler(lmdb)
    model, optim, step0, samples0 = load_or_create(device)
    model.train()

    # ---- fixed validation batch ----
    X_val, Y_val = sampler.sample_batch(val_batch)
    X_val_t = torch.from_numpy(X_val).float().unsqueeze(1).to(device)
    Y_val_t = torch.from_numpy(Y_val).float().to(device)

    samples_seen = samples0
    start_time = time.time()
    last_log = start_time

    for step in range(step0 + 1, total_steps + 1):
        X, Y = sampler.sample_batch(batch_size)

        for i in range(batch_size):
            X[i] = apply_uttt_symmetry(X[i], random.randrange(8))

        X_t = torch.from_numpy(X).float().unsqueeze(1).to(device)
        Y_t = torch.from_numpy(Y).float().to(device)

        pred = model(X_t)
        loss = F.mse_loss(pred, Y_t)

        optim.zero_grad()
        loss.backward()
        optim.step()

        samples_seen += batch_size

        if step % log_every == 0:
            dt = time.time() - last_log
            sps = log_every / dt
            print(
                f"[step {step}/{total_steps}] "
                f"train_loss={loss.item():.6f} | "
                f"samples={samples_seen:,} | "
                f"{sps:.2f} steps/s"
            )
            last_log = time.time()
            save(model, optim, step, samples_seen)

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                pv = model(X_val_t)
                mse = F.mse_loss(pv, Y_val_t).item()
                corr = torch.corrcoef(torch.stack([pv, Y_val_t]))[0, 1].item()
            model.train()
            print(f"   ↳ VALIDATION: mse={mse:.6f}, corr={corr:.4f}")

    save(model, optim, total_steps, samples_seen)
    print("Training finished.")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    train()
