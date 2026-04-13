"""build the filtered dataset from lmdb states with enough visits."""

import sys
import time
from pathlib import Path

import numpy as np

from uttt.game.logic import UltimateTicTacToeGame
from uttt.game.lmdb_qtable import VALUE_FMT, LMDBQTable
from uttt.paths import (
    FILTERED_X_PATH,
    FILTERED_Y_PATH,
    FIXED_QTABLE_PATH,
    ensure_project_dirs,
)


MIN_VISITS = 2
PROGRESS_EVERY = 250_000


def main():
    """build and save the filtered numpy arrays."""
    ensure_project_dirs()
    print("building filtered numpy data")
    print(f"lmdb_path: {FIXED_QTABLE_PATH}")
    print(f"out_x_path: {FILTERED_X_PATH}")
    print(f"out_y_path: {FILTERED_Y_PATH}")
    print(f"min_visits: {MIN_VISITS}")
    print(f"value_fmt: {VALUE_FMT}")

    qtable = LMDBQTable(path=FIXED_QTABLE_PATH, readonly=True, lock=False, max_readers=1)

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
                        f"count scan: scanned={scanned:,} kept_states={kept_states:,} "
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
        print("no rows passed filtering")
        return

    print(f"allocating x and y for {expanded_rows:,} expanded rows")
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
                        f"fill scan: scanned={scanned_fill:,} written_rows={write_idx:,} "
                        f"speed={scanned_fill/dt:,.0f} rows/sec"
                    )

                state_int = qtable.state_int_from_key(k)
                q, visits = qtable.decode_value(v)
                if int(visits) < MIN_VISITS:
                    continue

                board = np.array(qtable.decode_board_from_state_int(state_int), dtype=np.int8).reshape(9, 9)
                for sym in UltimateTicTacToeGame.all_symmetries_fast(board):
                    X[write_idx] = np.asarray(sym, dtype=np.float32)
                    y[write_idx] = np.float32(q)
                    write_idx += 1
    finally:
        qtable.close()

    if write_idx != expanded_rows:
        raise RuntimeError(f"row mismatch: wrote {write_idx}, expected {expanded_rows}")

    np.save(FILTERED_X_PATH, X)
    np.save(FILTERED_Y_PATH, y)

    dt = time.time() - t0
    print(f"saved x to {FILTERED_X_PATH}")
    print(f"saved y to {FILTERED_Y_PATH}")
    print(
        f"scanned={scanned:,}, kept_states={kept_states:,}, expanded_rows={expanded_rows:,}, "
        f"secs={dt:.2f}"
    )
    print(f"x shape={X.shape}, y shape={y.shape}")


if __name__ == "__main__":
    main()
