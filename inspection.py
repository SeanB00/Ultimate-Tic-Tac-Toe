import math

import lmdb
import os
import random
import struct
import time

import numpy as np

import hashing

from logic import UltimateTicTacToeGame

# ===== CONFIG =====
LMDB_PATH = "fixed_qtable.lmdb"
KEY_BYTES = 32
VALUE_FMT = "di"            # (double q_value, int count)
VALUE_SIZE = struct.calcsize(VALUE_FMT)
NUM_PLAYS = 10

PROGRESS_EVERY = 500_000     # print progress every N entries
MAX_ENTRIES = 1_000_000
SAMPLE_SIZE = 12_000
PLOT_SAMPLE_SIZE = 20_000
SAMPLE_PROGRESS_EVERY = 250_000
PLOT_OUTPUT_DIR = "inspection_plots"
TEST_REPORT_PATH = "inspection_test_report.txt"
TEST_EXPLANATIONS_PATH = "inspection_test_explanations.txt"
# ==================


def decode_state(key_bytes: bytes):
    state_int = int.from_bytes(key_bytes, "big", signed=False)

    return hashing.decode_board_from_int(state_int)


def iter_lmdb_entries(path, max_entries=None):
    env = lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=False,
    )
    try:
        with env.begin() as txn:
            cursor = txn.cursor()
            for idx, (k, v) in enumerate(cursor):
                if max_entries is not None and idx >= max_entries:
                    break
                yield k, v
    finally:
        env.close()


def collect_samples(path, sample_size, max_entries=None, seed=1337):
    rng = random.Random(seed)
    unpack = struct.unpack
    samples = []
    seen = 0
    start = time.time()

    for k, v in iter_lmdb_entries(path, max_entries=max_entries):
        if len(k) != KEY_BYTES or len(v) != VALUE_SIZE:
            continue

        q, visits = unpack(VALUE_FMT, v)
        board = np.array(decode_state(k)).reshape((9, 9))
        flat = board.ravel()
        x_count = int(np.count_nonzero(flat == 1))
        o_count = int(np.count_nonzero(flat == -1))
        empty_count = int(np.count_nonzero(flat == 0))

        sample = {
            "board": board,
            "q": float(q),
            "visits": int(visits),
            "x_count": x_count,
            "o_count": o_count,
            "empty_count": empty_count,
            "non_empty": x_count + o_count,
        }

        seen += 1
        if SAMPLE_PROGRESS_EVERY and seen % SAMPLE_PROGRESS_EVERY == 0:
            elapsed = time.time() - start
            rate = seen / elapsed if elapsed > 0 else 0
            print(f"[sampling] {seen:,} valid entries scanned ({rate:,.0f} entries/sec)")
        if len(samples) < sample_size:
            samples.append(sample)
        else:
            j = rng.randint(0, seen - 1)
            if j < sample_size:
                samples[j] = sample

    return samples, seen


def run_data_tests(samples, game):
    results = []

    def record(name, passed, details):
        results.append({"name": name, "passed": passed, "details": details})

    def test_board_values_valid():
        for sample in samples:
            if not np.isin(sample["board"], [-1, 0, 1]).all():
                return False, "Found board values outside {-1,0,1}."
        return True, f"Checked {len(samples)} sampled boards."

    def test_turn_balance():
        for sample in samples:
            if abs(sample["x_count"] - sample["o_count"]) > 1:
                return False, "Found board with move imbalance > 1."
        return True, "All sampled boards respect turn balance."

    def test_piece_count_consistency():
        for sample in samples:
            if sample["x_count"] + sample["o_count"] + sample["empty_count"] != 81:
                return False, "Piece counts do not sum to 81 cells."
        return True, "Piece counts match 81 cells for all samples."

    def test_visits_nonnegative():
        for sample in samples:
            if sample["visits"] < 0:
                return False, "Found negative visit count."
        return True, "All sampled visit counts are non-negative."

    def test_q_finite():
        for sample in samples:
            if not np.isfinite(sample["q"]):
                return False, "Found non-finite Q value."
        return True, "All sampled Q values are finite."

    def test_symmetry_shapes():
        for sample in samples:
            syms = game.all_symmetries_fast(sample["board"])
            if any(sym.shape != (9, 9) for sym in syms):
                return False, "Symmetry transform produced wrong shape."
        return True, "All symmetry transforms kept 9x9 shape."

    def test_symmetry_piece_counts():
        for sample in samples:
            syms = game.all_symmetries_fast(sample["board"])
            for sym in syms:
                flat = sym.ravel()
                if np.count_nonzero(flat == 1) != sample["x_count"]:
                    return False, "X count changed under symmetry."
                if np.count_nonzero(flat == -1) != sample["o_count"]:
                    return False, "O count changed under symmetry."
        return True, "All symmetry transforms preserved piece counts."

    def test_symmetry_canonical_invariance():
        for sample in samples:
            canonical = game.canonical_board_int(sample["board"])
            for sym in game.all_symmetries_fast(sample["board"]):
                if game.canonical_board_int(sym) != canonical:
                    return False, "Canonical hash changed under symmetry."
        return True, "Canonical hash invariant across symmetries."

    def test_canonical_equals_min_symmetry_hash():
        for sample in samples:
            syms = game.all_symmetries_fast(sample["board"])
            sym_hashes = [hashing.encode_board_to_int(sym.ravel()) for sym in syms]
            if game.canonical_board_int(sample["board"]) != min(sym_hashes):
                return False, "Canonical hash is not the minimum symmetry hash."
        return True, "Canonical hash equals minimum symmetry hash."

    def test_symmetry_unique_count():
        for sample in samples:
            syms = game.all_symmetries_fast(sample["board"])
            unique = {
                hashing.encode_board_to_int(sym.ravel()) for sym in syms
            }
            if not (1 <= len(unique) <= 8):
                return False, "Symmetry unique count outside [1, 8]."
        return True, "Symmetry variants count is within expected range."

    tests = [
        ("Board values are valid", test_board_values_valid),
        ("Move counts obey turn balance", test_turn_balance),
        ("Piece counts sum to 81 cells", test_piece_count_consistency),
        ("Visits are non-negative", test_visits_nonnegative),
        ("Q values are finite", test_q_finite),
        ("Symmetry transforms keep 9x9 shape", test_symmetry_shapes),
        ("Symmetry transforms preserve piece counts", test_symmetry_piece_counts),
        ("Canonical hash invariant across symmetries", test_symmetry_canonical_invariance),
        ("Canonical hash equals minimum symmetry hash", test_canonical_equals_min_symmetry_hash),
        ("Symmetry unique count within expected range", test_symmetry_unique_count),
    ]

    for name, func in tests:
        passed, details = func()
        record(name, passed, details)

    return results


TEST_EXPLANATIONS = {
    "Board values are valid": (
        "Verifies that sampled boards only contain -1, 0, or 1. "
        "This confirms there are no corrupted cell values in the state data."
    ),
    "Move counts obey turn balance": (
        "Checks that the number of X and O moves differ by at most one. "
        "This ensures sampled states follow valid turn order."
    ),
    "Piece counts sum to 81 cells": (
        "Verifies that X, O, and empty counts always sum to 81 cells, "
        "so the board encoding stays internally consistent."
    ),
    "Visits are non-negative": (
        "Ensures visit counters never go negative, which would indicate "
        "corrupted training statistics."
    ),
    "Q values are finite": (
        "Confirms Q values are real numbers (not NaN or infinity), "
        "indicating stable numeric data."
    ),
    "Symmetry transforms keep 9x9 shape": (
        "Validates that symmetry operations preserve the 9x9 board structure, "
        "so the symmetry logic does not distort board dimensions."
    ),
    "Symmetry transforms preserve piece counts": (
        "Checks that rotations/reflections never change how many X or O pieces "
        "are on the board, preserving game-state integrity."
    ),
    "Canonical hash invariant across symmetries": (
        "Ensures all symmetric variants map to the same canonical hash, which "
        "is critical for symmetry-aware hashing of board states."
    ),
    "Canonical hash equals minimum symmetry hash": (
        "Confirms canonical hashing selects the minimum integer hash among all "
        "symmetry variants."
    ),
    "Symmetry unique count within expected range": (
        "Verifies that a board produces between 1 and 8 unique symmetry variants, "
        "matching the D4 symmetry group for UTTT boards."
    ),
}


def write_test_explanations(path):
    lines = ["Inspection test explanations:", ""]
    for name, description in TEST_EXPLANATIONS.items():
        lines.append(f"- {name}: {description}")
    lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_test_report(path, results, sample_count, seen_count):
    lines = [
        "Inspection test report",
        f"Sampled entries: {sample_count} (from {seen_count} valid entries scanned)",
        "",
    ]

    passed = 0
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1
        lines.append(f"[{status}] {result['name']}: {result['details']}")

    lines.append("")
    lines.append(f"Passed {passed}/{len(results)} tests.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def plot_sample_distributions(samples, output_dir):
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    def log1p_normalized(x):
        res =  math.log1p(abs(x))  / math.log(2)
        if x > 0:
            return res
        return -1*res
    q_values = np.array(
        [log1p_normalized(sample["q"]) for sample in samples],
        dtype=float
    )


    visits = np.array([sample["visits"] for sample in samples], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(q_values, bins=60, color="#4C72B0", edgecolor="white")
    ax.set_title("Sampled Q-value distribution (with log)")
    ax.set_xlabel("Q value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "q_value_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(visits, bins=60, color="#55A868", edgecolor="white", log=True)
    ax.set_title("Sampled visit-count distribution")
    ax.set_xlabel("Visit count")
    ax.set_ylabel("Count (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "visit_distribution.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hexbin(visits, q_values, gridsize=50, cmap="viridis", mincnt=1, xscale="log")
    ax.set_title("Q value vs visit count (log scale)")
    ax.set_xlabel("Visit count (log scale)")
    ax.set_ylabel("Q value")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "q_vs_visits.png"), dpi=150)
    plt.close(fig)


def run_inspection_suite(path):
    game = UltimateTicTacToeGame(q_table={}, training=False, multiprocess=False)

    samples, seen = collect_samples(
        path,
        sample_size=SAMPLE_SIZE,
        max_entries=MAX_ENTRIES,
    )

    plot_samples, _ = collect_samples(
        path,
        sample_size=min(PLOT_SAMPLE_SIZE, SAMPLE_SIZE),
        max_entries=MAX_ENTRIES,
        seed=2024,
    )

    results = run_data_tests(samples, game)
    write_test_report(TEST_REPORT_PATH, results, len(samples), seen)
    write_test_explanations(TEST_EXPLANATIONS_PATH)
    plot_sample_distributions(plot_samples, PLOT_OUTPUT_DIR)

    return results


def inspect_lmdb(path):
    env = lmdb.open(
        path,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
        subdir=False,
    )

    # ---- running statistics ----
    entries = 0

    min_q = float("inf")
    max_q = float("-inf")
    sum_q = 0.0

    min_visits = float("inf")
    max_visits = -1
    sum_visits = 0

    zero_visits = 0
    positive_q = 0
    negative_q = 0

    max_visit_key = None
    max_visit_q = None

    bad_key_len = 0
    bad_val_len = 0

    start = time.time()

    with env.begin() as txn:
        info = env.info()
        stat = txn.stat()

        print("===== LMDB INFO =====")
        print(f"Entries      : {stat['entries']}")
        print(f"Page size    : {env.stat()['psize']} bytes")
        print(f"Map size     : {info['map_size'] / (1024**3):.2f} GB")
        print(f"Last page no : {info['last_pgno']}")
        print()

        cursor = txn.cursor()
        unpack = struct.unpack
        used_entries = 0
        for k, v in cursor:
            entries += 1

            if entries > 15_000_000:
                break
            # ---- progress ----
            if entries % PROGRESS_EVERY == 0:
                elapsed = time.time() - start
                print(f"[{entries:,}] scanned ({entries/elapsed:,.0f} entries/sec)")

            # ---- key/value sanity ----
            if len(k) != KEY_BYTES:
                bad_key_len += 1
                continue

            if len(v) != VALUE_SIZE:
                bad_val_len += 1
                continue

            q, visits = unpack(VALUE_FMT, v)

            # ---- visit stats ----

            b = decode_state(k)
            non_empty = 0
            for _ in b:
                if _ != 0:
                    non_empty += 1

            if non_empty < NUM_PLAYS:

                used_entries += 1
                sum_visits += visits
                min_visits = min(min_visits, visits)
                if visits == 0:
                    zero_visits += 1

                if visits > max_visits:
                    max_visits = visits
                    max_visit_key = k
                    max_visit_q = q

            # ---- Q stats ----
            sum_q += q
            min_q = min(min_q, q)
            max_q = max(max_q, q)

            if q > 0:
                positive_q += 1
            elif q < 0:
                negative_q += 1

    env.close()

    print("\n===== SUMMARY =====")
    print(f"Total entries        : {entries}")
    print(f"Effective entries   : {used_entries}" )
    print(f"Bad key length       : {bad_key_len}")
    print(f"Bad value length     : {bad_val_len}")
    print()

    print("===== VISIT STATS =====")
    print(f"Min visits           : {min_visits}")
    print(f"Max visits           : {max_visits}")
    print(f"Avg visits Under {NUM_PLAYS}         : {sum_visits / used_entries:.3f}")
    print(f"Zero-visit states    : {zero_visits}")
    print()

    print("===== Q STATS =====")
    print(f"Min Q                : {min_q}")
    print(f"Max Q                : {max_q}")
    print(f"Avg Q                : {sum_q / entries:.6f}")
    print(f"Positive Q states    : {positive_q}")
    print(f"Negative Q states    : {negative_q}")
    print()

    print("===== MOST VISITED STATE =====")
    print(f"Visits : {max_visits}")
    print(f"Q-value: {max_visit_q}")
    print()

    board = decode_state(max_visit_key)
    print(board)
    print(len(board))
    print_ultimate_board(board)



def print_ultimate_board(board):
    def sym(x):
        return {0: "-", 1: "X", -1: "O"}.get(x, str(x))
    board = [[board[9*i+j] for j in range(9)] for i in range(9)]
    s = ""
    for row in range(9):
        s += "|"
        for col in range(9):
            s += sym(board[row][col]) + "|"
            if col % 3 == 2 and col != 8:
                s += "  |"
        s += "\n"
        if row % 3 == 2:
            s += "\n"
    print(s)
    print("-" * 9)


if __name__ == "__main__":
    #run_inspection_suite(LMDB_PATH)
    inspect_lmdb(LMDB_PATH)