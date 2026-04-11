"""
Build a mixed dataset from expanded min2 data plus visits==1 LMDB states.

This version is tuned to the original v1 split from your metadata JSONs.

Pipeline:
1) Keep all rows from expanded_X_min2 / expanded_y_min2 (base_min2 bucket).
2) Single sequential LMDB scan for visits==1 states.
3) Disjoint LMDB bucket classification with fixed precedence:
   true_final -> hard_endgame -> onecount_high_pos -> onecount_high_neg -> onecount_random_deep
4) Reservoir-sample states per bucket.
5) Symmetry-expand sampled states and write exact per-bucket row targets.
"""

import math
import random
import sys
import time
import gc
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uttt.game.logic import UltimateTicTacToeGame
from uttt.game.lmdb_qtable import LMDBQTable
from uttt.paths import (
    EXPANDED_X_PATH,
    EXPANDED_Y_PATH,
    FIXED_QTABLE_PATH,
    MIXED_X_PATH,
    MIXED_Y_PATH,
    ensure_project_dirs,
)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
# Exact split from previous run metadata.
SPLIT_RATIOS = {
    "base_min2": 0.45,
    "onecount_high_pos": 0.175,
    "onecount_high_neg": 0.175,
    "onecount_random_deep": 0.15,
    "true_final": 0.01,
    "hard_endgame": 0.04,
}

# Bucket precedence is disjoint and order-sensitive.
LMDB_BUCKET_ORDER = [
    "true_final",
    "hard_endgame",
    "onecount_high_pos",
    "onecount_high_neg",
    "onecount_random_deep",
]

# Thresholds from previous run metadata.
ONECOUNT_OCC_MIN = 30
HARD_ENDGAME_OCC_MIN = 40
HARD_ENDGAME_ABS_SCORE_MIN = 0.75
HIGH_POS_MIN = 0.5
HIGH_NEG_MAX = -0.5
SCORE_ONE_ATOL = 1e-9

# LMDB schema
ONECOUNT_VISITS = 1

# Runtime controls
SEED = 42
PROGRESS_EVERY = 500_000
CHUNK_ROWS = 65_536


def split_counts(total, weights):
    """Split integer `total` into integer bucket counts by fractional weights."""
    if total <= 0:
        return {k: 0 for k in weights}

    keys = list(weights.keys())
    norm = sum(max(0.0, float(weights[k])) for k in keys)
    if norm <= 0:
        even = total // len(keys)
        rem = total - even * len(keys)
        out = {k: even for k in keys}
        for i in range(rem):
            out[keys[i % len(keys)]] += 1
        return out

    raw = {k: total * (max(0.0, float(weights[k])) / norm) for k in keys}
    flo = {k: int(math.floor(raw[k])) for k in keys}
    rem = total - sum(flo.values())

    by_rem = sorted(keys, key=lambda k: (raw[k] - flo[k]), reverse=True)
    for i in range(rem):
        flo[by_rem[i % len(by_rem)]] += 1

    return flo


def check_win_3x3(board3x3):
    """Return winner for a 3x3 board: 1, -1, or 0."""
    for i in range(3):
        s = int(np.sum(board3x3[i, :]))
        if s == 3:
            return 1
        if s == -3:
            return -1
    for j in range(3):
        s = int(np.sum(board3x3[:, j]))
        if s == 3:
            return 1
        if s == -3:
            return -1

    d1 = int(np.trace(board3x3))
    if d1 == 3:
        return 1
    if d1 == -3:
        return -1

    d2 = int(np.trace(np.fliplr(board3x3)))
    if d2 == 3:
        return 1
    if d2 == -3:
        return -1

    return 0


def is_terminal_uttt(board9x9):
    """Return True if the UTTT board is terminal (meta win or fully resolved tie)."""
    sub = np.zeros((3, 3), dtype=np.int8)
    resolved = np.zeros((3, 3), dtype=bool)

    for bi in range(3):
        for bj in range(3):
            b = board9x9[bi * 3:(bi + 1) * 3, bj * 3:(bj + 1) * 3]
            w = check_win_3x3(b)
            if w != 0:
                sub[bi, bj] = w
                resolved[bi, bj] = True
            elif np.count_nonzero(b == 0) == 0:
                resolved[bi, bj] = True

    if check_win_3x3(sub) != 0:
        return True
    return bool(np.all(resolved))


def is_score_pm_one(score):
    """Return True for values numerically equal to +/-1 within tolerance."""
    s = float(score)
    return abs(s - 1.0) <= SCORE_ONE_ATOL or abs(s + 1.0) <= SCORE_ONE_ATOL


def classify_bucket(board, q_value, occupancy):
    """Classify a visits==1 state into one disjoint bucket by precedence."""
    if occupancy < ONECOUNT_OCC_MIN:
        return None

    if is_terminal_uttt(board) and is_score_pm_one(q_value):
        return "true_final"
    if occupancy >= HARD_ENDGAME_OCC_MIN and abs(float(q_value)) >= HARD_ENDGAME_ABS_SCORE_MIN:
        return "hard_endgame"
    if float(q_value) >= HIGH_POS_MIN:
        return "onecount_high_pos"
    if float(q_value) <= HIGH_NEG_MAX:
        return "onecount_high_neg"
    return "onecount_random_deep"


@dataclass
class ReservoirStates:
    """Uniform reservoir sampler for a stream with fixed max capacity."""

    capacity: int
    seed: int

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.items = []
        self.seen = 0

    def add(self, item):
        self.seen += 1
        if self.capacity <= 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        j = self.rng.randint(0, self.seen - 1)
        if j < self.capacity:
            self.items[j] = item


def write_batch(X_out, y_out, write_idx, boards_batch, targets_batch):
    """Write buffered rows into the in-memory output arrays."""
    n = len(boards_batch)
    if n == 0:
        return write_idx
    X_out[write_idx:write_idx + n] = np.asarray(boards_batch, dtype=np.float32)
    y_out[write_idx:write_idx + n] = np.asarray(targets_batch, dtype=np.float32)
    return write_idx + n


def write_base_rows(base_x, base_y, X_out, y_out):
    """Copy all base rows in chunks and return next write index."""
    n = int(base_x.shape[0])
    w = 0
    while w < n:
        j = min(w + CHUNK_ROWS, n)
        X_out[w:j] = np.asarray(base_x[w:j], dtype=np.float32)
        y_out[w:j] = np.asarray(base_y[w:j], dtype=np.float32)
        w = j
    return n


def main():
    """Build mixed_X_v1.npy and mixed_y_v1.npy using the configured split."""
    t0 = time.time()
    ensure_project_dirs()
    print("=== Build mixed dataset v1 (retuned, RAM mode) ===")

    base_x = np.load(EXPANDED_X_PATH)
    base_y = np.load(EXPANDED_Y_PATH)

    if base_x.ndim != 3 or base_x.shape[1:] != (9, 9):
        raise ValueError(f"Unexpected base X shape: {base_x.shape}")
    if base_y.ndim != 1 or base_y.shape[0] != base_x.shape[0]:
        raise ValueError(f"Unexpected base y shape: {base_y.shape}")

    base_rows = int(base_x.shape[0])
    total_rows = int(math.ceil(base_rows / SPLIT_RATIOS["base_min2"]))
    target_rows = split_counts(total_rows, SPLIT_RATIOS)

    # Base bucket is fixed to all available base rows.
    target_rows["base_min2"] = base_rows
    delta = total_rows - sum(target_rows.values())
    if delta != 0:
        target_rows["onecount_random_deep"] += delta

    # Per-bucket target states (before symmetry expansion).
    state_targets_exact = {
        b: int(math.ceil(target_rows[b] / 8.0))
        for b in LMDB_BUCKET_ORDER
    }

    print(f"base_rows={base_rows:,}")
    print(f"total_rows_target={total_rows:,}")
    print(f"target_rows={target_rows}")
    print(f"state_targets_exact={state_targets_exact}")

    reservoirs = {
        b: ReservoirStates(capacity=state_targets_exact[b], seed=SEED + i)
        for i, b in enumerate(LMDB_BUCKET_ORDER)
    }

    scanned = 0
    onecount_entries = 0

    state_candidates_seen = {b: 0 for b in LMDB_BUCKET_ORDER}
    lmdb_hits_by_priority = {b: 0 for b in LMDB_BUCKET_ORDER}

    qtable = LMDBQTable(path=FIXED_QTABLE_PATH, readonly=True, lock=False, max_readers=1)

    try:
        with qtable.begin_read() as txn:
            for k, v in qtable.iter_raw_items(txn=txn):
                scanned += 1
                if PROGRESS_EVERY and scanned % PROGRESS_EVERY == 0:
                    elapsed = max(time.time() - t0, 1e-9)
                    sampled_states = sum(len(reservoirs[b].items) for b in LMDB_BUCKET_ORDER)
                    print(
                        f"[scan] scanned={scanned:,} onecount={onecount_entries:,} "
                        f"sampled_states={sampled_states:,} speed={scanned/elapsed:,.0f}/s"
                    )

                state_int = qtable.state_int_from_key(k)
                q, visits = qtable.decode_value(v)
                if int(visits) != ONECOUNT_VISITS:
                    continue

                onecount_entries += 1
                board = np.array(qtable.decode_board_from_state_int(state_int), dtype=np.int8).reshape(9, 9)
                occ = int(np.count_nonzero(board))

                bucket = classify_bucket(board, float(q), occ)
                if bucket is None:
                    continue

                state_candidates_seen[bucket] += 1
                lmdb_hits_by_priority[bucket] += 1
                reservoirs[bucket].add((state_int, float(q), occ))
    finally:
        qtable.close()

    sampled_states = {b: reservoirs[b].items for b in LMDB_BUCKET_ORDER}
    state_selected = {b: len(sampled_states[b]) for b in LMDB_BUCKET_ORDER}

    # Initial row targets for LMDB buckets.
    lmdb_row_targets = {b: int(target_rows[b]) for b in LMDB_BUCKET_ORDER}

    # If a strict bucket is short, move its shortage to random_deep.
    shortage_rows_moved_to_random = {}
    strict = ["true_final", "hard_endgame", "onecount_high_pos", "onecount_high_neg"]
    for b in strict:
        avail_rows = 8 * state_selected[b]
        target = lmdb_row_targets[b]
        if avail_rows < target:
            shortage = target - avail_rows
            shortage_rows_moved_to_random[b] = shortage
            lmdb_row_targets[b] = avail_rows
            lmdb_row_targets["onecount_random_deep"] += shortage
        else:
            shortage_rows_moved_to_random[b] = 0

    # Clamp random_deep target to available rows.
    random_avail = 8 * state_selected["onecount_random_deep"]
    if lmdb_row_targets["onecount_random_deep"] > random_avail:
        lmdb_row_targets["onecount_random_deep"] = random_avail

    final_rows = base_rows + sum(lmdb_row_targets[b] for b in LMDB_BUCKET_ORDER)

    print(f"state_selected={state_selected}")
    print(f"lmdb_row_targets={lmdb_row_targets}")
    print(f"final_rows={final_rows:,}")

    X_out = np.empty((final_rows, 9, 9), dtype=np.float32)
    y_out = np.empty((final_rows,), dtype=np.float32)

    write_idx = write_base_rows(base_x, base_y, X_out, y_out)
    del base_x, base_y
    gc.collect()

    game = UltimateTicTacToeGame(q_table={}, training=False, multiprocess=False)

    rows_written_bucket = {b: 0 for b in LMDB_BUCKET_ORDER}
    boards_batch = []
    targets_batch = []

    for bucket in LMDB_BUCKET_ORDER:
        row_limit = lmdb_row_targets[bucket]
        if row_limit <= 0:
            continue

        for state_int, q, _occ in sampled_states[bucket]:
            if rows_written_bucket[bucket] >= row_limit:
                break

            board = np.array(qtable.decode_board_from_state_int(state_int), dtype=np.int8).reshape(9, 9)
            syms = game.all_symmetries_fast(board)

            for sym in syms:
                if rows_written_bucket[bucket] >= row_limit:
                    break
                boards_batch.append(sym.astype(np.float32))
                targets_batch.append(np.float32(q))
                rows_written_bucket[bucket] += 1

                if len(boards_batch) >= CHUNK_ROWS:
                    write_idx = write_batch(X_out, y_out, write_idx, boards_batch, targets_batch)
                    boards_batch.clear()
                    targets_batch.clear()

    write_idx = write_batch(X_out, y_out, write_idx, boards_batch, targets_batch)

    if write_idx != final_rows:
        raise RuntimeError(f"Row mismatch: wrote {write_idx}, expected {final_rows}")

    np.save(MIXED_X_PATH, X_out)
    np.save(MIXED_Y_PATH, y_out)

    elapsed = time.time() - t0

    print(f"[done] wrote {MIXED_X_PATH}")
    print(f"[done] wrote {MIXED_Y_PATH}")
    print(f"[done] scanned={scanned:,}, onecount_entries={onecount_entries:,}")
    print(f"[done] state_candidates_seen={state_candidates_seen}")
    print(f"[done] lmdb_hits_by_priority={lmdb_hits_by_priority}")
    print(f"[done] shortage_rows_moved_to_random={shortage_rows_moved_to_random}")
    print(f"[done] elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
