"""
Filter LMDB states by minimum visit count, expand each kept state by all 8
UTTT symmetries, and save two NumPy files:

- X.npy -> shape (N, 9, 9), dtype float32
- y.npy -> shape (N,), dtype float32

This version is intentionally straightforward:
1) One cursor pass over LMDB
2) Keep expanded boards/targets in Python lists
3) Convert to numpy arrays
4) Save at the end
"""

import os
import json
import struct
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lmdb
import numpy as np

from uttt.game import hashing
from uttt.game.logic import UltimateTicTacToeGame
from uttt.paths import (
    EXPANDED_META_PATH,
    EXPANDED_X_PATH,
    EXPANDED_Y_PATH,
    FIXED_QTABLE_PATH,
    ensure_project_dirs,
)


MIN_VISITS = 2
PROGRESS_EVERY = 250_000

KEY_BYTES = 32
VALUE_FMT = "dI"  # (q_value: float64, visits: uint32)
VALUE_SIZE = struct.calcsize(VALUE_FMT)


def main():
    ensure_project_dirs()
    print("=== LMDB -> Expanded NumPy (simple RAM mode) ===")
    print(f"LMDB_PATH      : {FIXED_QTABLE_PATH}")
    print(f"OUT_X_PATH     : {EXPANDED_X_PATH}")
    print(f"OUT_Y_PATH     : {EXPANDED_Y_PATH}")
    print(f"MIN_VISITS     : {MIN_VISITS}")
    print(f"VALUE_FMT      : {VALUE_FMT}")

    env = lmdb.open(
        os.fspath(FIXED_QTABLE_PATH),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=False,
    )

    game = UltimateTicTacToeGame(q_table={}, training=False, multiprocess=False)
    unpack = struct.unpack

    boards = []
    targets = []

    scanned = 0
    kept_states = 0
    expanded_rows = 0
    bad_key_len = 0
    bad_val_len = 0

    t0 = time.time()
    with env.begin() as txn:
        cursor = txn.cursor()
        for k, v in cursor:
            scanned += 1
            if PROGRESS_EVERY > 0 and scanned % PROGRESS_EVERY == 0:
                dt = max(time.time() - t0, 1e-9)
                print(
                    f"[progress] scanned={scanned:,} kept_states={kept_states:,} "
                    f"expanded_rows={expanded_rows:,} speed={scanned/dt:,.0f} rows/sec"
                )

            if len(k) != KEY_BYTES:
                bad_key_len += 1
                continue
            if len(v) != VALUE_SIZE:
                bad_val_len += 1
                continue

            q, visits = unpack(VALUE_FMT, v)
            if int(visits) < MIN_VISITS:
                continue

            state_int = int.from_bytes(k, "big", signed=False)
            board = np.array(hashing.decode_board_from_int(state_int), dtype=np.int8).reshape(9, 9)
            syms = game.all_symmetries_fast(board)

            # Keep each expanded symmetry as an independent sample row.
            for sym in syms:
                boards.append(sym.astype(np.float32))
                targets.append(np.float32(q))
                expanded_rows += 1

            kept_states += 1

    env.close()

    if expanded_rows == 0:
        print("[done] No rows passed filtering. Nothing saved.")
        return

    print("[save] Converting python lists to numpy arrays...")
    X = np.array(boards, dtype=np.float32)  # (N, 9, 9)
    y = np.array(targets, dtype=np.float32)  # (N,)

    print("[save] Writing X/y .npy files...")
    np.save(EXPANDED_X_PATH, X)
    np.save(EXPANDED_Y_PATH, y)

    dt = time.time() - t0
    print(
        f"[done] scanned={scanned:,}, kept_states={kept_states:,}, expanded_rows={expanded_rows:,}, "
        f"bad_key_len={bad_key_len:,}, bad_val_len={bad_val_len:,}, secs={dt:.2f}"
    )
    print(f"[done] X shape={X.shape}, y shape={y.shape}")

    meta = {
        "lmdb_path": os.fspath(FIXED_QTABLE_PATH),
        "out_x_path": os.fspath(EXPANDED_X_PATH),
        "out_y_path": os.fspath(EXPANDED_Y_PATH),
        "min_visits": MIN_VISITS,
        "value_fmt": VALUE_FMT,
        "key_bytes": KEY_BYTES,
        "kept_states_before_symmetry": kept_states,
        "expanded_rows": expanded_rows,
        "x_shape": list(X.shape),
        "y_shape": list(y.shape),
        "x_dtype": str(X.dtype),
        "y_dtype": str(y.dtype),
    }
    with open(EXPANDED_META_PATH, "w", encoding="ascii") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] metadata written: {EXPANDED_META_PATH}")


if __name__ == "__main__":
    main()
