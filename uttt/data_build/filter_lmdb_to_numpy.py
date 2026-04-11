"""
Filter LMDB states by minimum visit count, expand each kept state by all 8
UTTT symmetries, and save two NumPy files:

- X.npy -> shape (N, 9, 9), dtype float32
- y.npy -> shape (N,), dtype float32

This version preallocates the final arrays in RAM:
1) First pass counts states with visits >= MIN_VISITS
2) Allocate the exact final NumPy arrays
3) Second pass fills them directly
4) Save at the end
"""

import sys
import time
from pathlib import Path
import numpy as np

from uttt.game.logic import UltimateTicTacToeGame
from uttt.game.lmdb_qtable import VALUE_FMT, LMDBQTable
from uttt.paths import (
    EXPANDED_X_PATH,
    EXPANDED_Y_PATH,
    FIXED_QTABLE_PATH,
    ensure_project_dirs,
)


MIN_VISITS = 2
PROGRESS_EVERY = 250_000


def main():
    ensure_project_dirs()
    print("=== LMDB -> Expanded NumPy (preallocated RAM mode) ===")
    print(f"LMDB_PATH      : {FIXED_QTABLE_PATH}")
    print(f"OUT_X_PATH     : {EXPANDED_X_PATH}")
    print(f"OUT_Y_PATH     : {EXPANDED_Y_PATH}")
    print(f"MIN_VISITS     : {MIN_VISITS}")
    print(f"VALUE_FMT      : {VALUE_FMT}")

    qtable = LMDBQTable(path=FIXED_QTABLE_PATH, readonly=True, lock=False, max_readers=1)

    game = UltimateTicTacToeGame(q_table={}, training=False, multiprocess=False)

    scanned = 0
    kept_states = 0

    t0 = time.time()
    try:
        with qtable.begin_read() as txn:
            for k, v in qtable.iter_raw_items(txn=txn):
                scanned += 1
                if PROGRESS_EVERY > 0 and scanned % PROGRESS_EVERY == 0:
                    dt = max(time.time() - t0, 1e-9)
                    print(
                        f"[count] scanned={scanned:,} kept_states={kept_states:,} "
                        f"speed={scanned/dt:,.0f} rows/sec"
                    )

                _state_int = qtable.state_int_from_key(k)
                _q, visits = qtable.decode_value(v)
                if int(visits) < MIN_VISITS:
                    continue

                kept_states += 1
    finally:
        qtable.close()

    expanded_rows = kept_states * 8
    if expanded_rows == 0:
        print("[done] No rows passed filtering. Nothing saved.")
        return

    print(f"[alloc] Allocating X/y for {expanded_rows:,} expanded rows...")
    X = np.empty((expanded_rows, 9, 9), dtype=np.float32)
    y = np.empty((expanded_rows,), dtype=np.float32)

    qtable = LMDBQTable(path=FIXED_QTABLE_PATH, readonly=True, lock=False, max_readers=1)
    write_idx = 0
    scanned_fill = 0
    try:
        with qtable.begin_read() as txn:
            for k, v in qtable.iter_raw_items(txn=txn):
                scanned_fill += 1
                if PROGRESS_EVERY > 0 and scanned_fill % PROGRESS_EVERY == 0:
                    dt = max(time.time() - t0, 1e-9)
                    print(
                        f"[fill] scanned={scanned_fill:,} written_rows={write_idx:,} "
                        f"speed={scanned_fill/dt:,.0f} rows/sec"
                    )

                state_int = qtable.state_int_from_key(k)
                q, visits = qtable.decode_value(v)
                if int(visits) < MIN_VISITS:
                    continue

                board = np.array(qtable.decode_board_from_state_int(state_int), dtype=np.int8).reshape(9, 9)
                for sym in game.all_symmetries_fast(board):
                    X[write_idx] = np.asarray(sym, dtype=np.float32)
                    y[write_idx] = np.float32(q)
                    write_idx += 1
    finally:
        qtable.close()

    if write_idx != expanded_rows:
        raise RuntimeError(f"Row mismatch: wrote {write_idx}, expected {expanded_rows}")

    print("[save] Writing X/y .npy files...")
    np.save(EXPANDED_X_PATH, X)
    np.save(EXPANDED_Y_PATH, y)

    dt = time.time() - t0
    print(
        f"[done] scanned={scanned:,}, kept_states={kept_states:,}, expanded_rows={expanded_rows:,}, "
        f"secs={dt:.2f}"
    )
    print(f"[done] X shape={X.shape}, y shape={y.shape}")


if __name__ == "__main__":
    main()
