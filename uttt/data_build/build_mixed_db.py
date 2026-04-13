"""build a mixed dataset from filtered rows and lmdb samples."""

import gc
import math
import random
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from uttt.game.logic import UltimateTicTacToeGame
from uttt.game.lmdb_qtable import LMDBQTable
from uttt.paths import (
    FILTERED_X_PATH,
    FILTERED_Y_PATH,
    FIXED_QTABLE_PATH,
    MIXED_X_PATH,
    MIXED_Y_PATH,
    ensure_project_dirs,
)


# config
SPLIT_RATIOS = {
    "base_min2": 0.45,
    "onecount_high_pos": 0.175,
    "onecount_high_neg": 0.175,
    "onecount_random_deep": 0.15,
    "true_final": 0.01,
    "hard_endgame": 0.04,
}

LMDB_BUCKET_ORDER = [
    "true_final",
    "hard_endgame",
    "onecount_high_pos",
    "onecount_high_neg",
    "onecount_random_deep",
]

ONECOUNT_OCC_MIN = 30
HARD_ENDGAME_OCC_MIN = 40
HARD_ENDGAME_ABS_SCORE_MIN = 0.75
HIGH_POS_MIN = 0.5
HIGH_NEG_MAX = -0.5
SCORE_ONE_TOLERANCE = 1e-5

SEED = 42
PROGRESS_EVERY = 500_000
CHUNK_ROWS = 65_536
SYMMETRY_COUNT = 8
ONECOUNT_VISITS = 1


def states_needed(row_target):
    """convert a row target into symmetry seeds."""
    return math.ceil(row_target / SYMMETRY_COUNT)


def compute_target_state_counts(base_rows):
    """compute approximate row and state targets."""
    total_rows = math.ceil(base_rows / SPLIT_RATIOS["base_min2"])
    target_rows = {}

    allocated_rows = 0
    for bucket in LMDB_BUCKET_ORDER[:-1]:
        rows = int(total_rows * SPLIT_RATIOS[bucket])
        target_rows[bucket] = rows
        allocated_rows += rows

    target_rows["onecount_random_deep"] = max(0, total_rows - base_rows - allocated_rows)
    target_states = {
        bucket: states_needed(target_rows[bucket])
        for bucket in LMDB_BUCKET_ORDER
    }
    return total_rows, target_rows, target_states


def is_score_endgame(score):
    """check whether a score is effectively plus or minus one."""
    score = abs(float(score))
    return abs(score - 1) <= SCORE_ONE_TOLERANCE


def classify_bucket(board, q_value, occupancy):
    """assign a visits-one state to one bucket."""
    if occupancy < ONECOUNT_OCC_MIN:
        return None
    if is_score_endgame(q_value):
        return "true_final"
    if occupancy >= HARD_ENDGAME_OCC_MIN and abs(float(q_value)) >= HARD_ENDGAME_ABS_SCORE_MIN:
        return "hard_endgame"
    if float(q_value) >= HIGH_POS_MIN:
        return "onecount_high_pos"
    if float(q_value) <= HIGH_NEG_MAX:
        return "onecount_high_neg"
    return "onecount_random_deep"


class ReservoirStates:
    """uniform reservoir sampler with fixed capacity."""

    def __init__(self, capacity, seed):
        """set up the reservoir state."""
        self.capacity = capacity
        self.seed = seed
        self.rng = random.Random(seed)
        self.items = []
        self.seen = 0

    def add(self, item):
        """add one item to the reservoir."""
        self.seen += 1
        if self.capacity <= 0:
            return
        if len(self.items) < self.capacity:
            self.items.append(item)
            return
        idx = self.rng.randint(0, self.seen - 1)
        if idx < self.capacity:
            self.items[idx] = item


def sample_lmdb_states(path, target_states):
    """sample visits-one states into bucket reservoirs."""
    reservoirs = {
        bucket: ReservoirStates(capacity=target_states[bucket], seed=SEED + idx)
        for idx, bucket in enumerate(LMDB_BUCKET_ORDER)
    }
    sampled_candidates = {bucket: 0 for bucket in LMDB_BUCKET_ORDER}
    start = time.time()

    qtable = LMDBQTable(path=path, readonly=True, lock=False, max_readers=1)
    try:
        with qtable.begin_read() as txn:
            for entry_index, (state_int, q_value, visits) in enumerate(
                qtable.iter_entries(txn=txn),
                start=1,
            ):
                if PROGRESS_EVERY and entry_index % PROGRESS_EVERY == 0:
                    elapsed = max(time.time() - start, 1e-9)
                    sampled_total = sum(len(r.items) for r in reservoirs.values())
                    print(
                        f"scan: entries={entry_index:,} "
                        f"sampled_states={sampled_total:,} speed={entry_index / elapsed:,.0f}/s"
                    )

                if int(visits) != ONECOUNT_VISITS:
                    continue

                board = np.array(qtable.decode_board_from_state_int(state_int), dtype=np.int8).reshape(9, 9)
                occupancy = int(np.count_nonzero(board))
                bucket = classify_bucket(board, float(q_value), occupancy)
                if bucket is None:
                    continue

                sampled_candidates[bucket] += 1
                reservoirs[bucket].add((state_int, float(q_value)))
    finally:
        qtable.close()

    sampled_states = {bucket: reservoirs[bucket].items for bucket in LMDB_BUCKET_ORDER}
    return sampled_states, sampled_candidates


def expanded_rows_by_bucket(sampled_states):
    """count rows contributed by each bucket after expansion."""
    return {
        bucket: len(sampled_states[bucket]) * SYMMETRY_COUNT
        for bucket in LMDB_BUCKET_ORDER
    }


def write_batch(X_out, y_out, write_idx, boards_batch, targets_batch):
    """flush one buffered batch into the output arrays."""
    batch_size = len(boards_batch)
    if batch_size == 0:
        return write_idx
    X_out[write_idx:write_idx + batch_size] = np.asarray(boards_batch, dtype=np.float32)
    y_out[write_idx:write_idx + batch_size] = np.asarray(targets_batch, dtype=np.float32)
    return write_idx + batch_size


def write_base_rows(base_x, base_y, X_out, y_out):
    """copy the filtered base rows into the output arrays."""
    base_rows = int(base_x.shape[0])
    write_idx = 0
    while write_idx < base_rows:
        next_idx = min(write_idx + CHUNK_ROWS, base_rows)
        X_out[write_idx:next_idx] = np.asarray(base_x[write_idx:next_idx], dtype=np.float32)
        y_out[write_idx:next_idx] = np.asarray(base_y[write_idx:next_idx], dtype=np.float32)
        write_idx = next_idx
    return base_rows


def write_lmdb_rows(sampled_states, X_out, y_out, write_idx):
    """expand sampled lmdb rows and append them."""
    rows_written = expanded_rows_by_bucket(sampled_states)
    boards_batch = []
    targets_batch = []

    for bucket in LMDB_BUCKET_ORDER:
        for state_int, q_value in sampled_states[bucket]:
            board = np.array(UltimateTicTacToeGame.get_board_from_int(state_int), dtype=np.int8)
            for symmetry in UltimateTicTacToeGame.all_symmetries_fast(board):
                boards_batch.append(symmetry)
                targets_batch.append(np.float32(q_value))

                if len(boards_batch) >= CHUNK_ROWS:
                    write_idx = write_batch(X_out, y_out, write_idx, boards_batch, targets_batch)
                    boards_batch.clear()
                    targets_batch.clear()

    write_idx = write_batch(X_out, y_out, write_idx, boards_batch, targets_batch)
    return write_idx, rows_written


def main():
    """build the mixed dataset using the configured split."""
    start = time.time()
    ensure_project_dirs()
    print("building mixed dataset")

    base_x = np.load(FILTERED_X_PATH)
    base_y = np.load(FILTERED_Y_PATH)
    base_rows = int(base_x.shape[0])
    total_rows_target, target_rows, target_states = compute_target_state_counts(base_rows)

    print(f"base_rows={base_rows:,}")
    print(f"total_rows_target={total_rows_target:,}")
    print(f"target_rows={target_rows}")
    print(f"target_states={target_states}")

    sampled_states, sampled_candidates = sample_lmdb_states(
        FIXED_QTABLE_PATH,
        target_states,
    )
    sampled_counts = {bucket: len(sampled_states[bucket]) for bucket in LMDB_BUCKET_ORDER}
    rows_written = expanded_rows_by_bucket(sampled_states)
    final_rows = base_rows + sum(rows_written.values())

    print(f"sampled_candidates={sampled_candidates}")
    print(f"sampled_counts={sampled_counts}")
    print(f"rows_written={rows_written}")
    print(f"final_rows={final_rows}")

    X_out = np.empty((final_rows, 9, 9), dtype=np.float32)
    y_out = np.empty((final_rows,), dtype=np.float32)

    write_idx = write_base_rows(base_x, base_y, X_out, y_out)
    del base_x, base_y
    gc.collect()

    write_idx, rows_written = write_lmdb_rows(sampled_states, X_out, y_out, write_idx)
    if write_idx != final_rows:
        raise RuntimeError(f"row mismatch: wrote {write_idx}, expected {final_rows}")

    np.save(MIXED_X_PATH, X_out)
    np.save(MIXED_Y_PATH, y_out)

    elapsed = time.time() - start
    print(f"saved x to {MIXED_X_PATH}")
    print(f"saved y to {MIXED_Y_PATH}")
    print(f"rows_written={rows_written}")
    print(f"elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
