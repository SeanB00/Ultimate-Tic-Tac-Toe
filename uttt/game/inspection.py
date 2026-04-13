import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import matplotlib.pyplot as plt

from uttt.game import hashing
from uttt.game.logic import UltimateTicTacToeGame
from uttt.game.lmdb_qtable import LMDBQTable
from uttt.paths import ARTIFACTS_DIR, FIXED_QTABLE_PATH, ensure_project_dirs

# config

# minimum number of non-empty cells used for "effective entries"
NUM_PLAYS = 10
MIN_VISITS = 2

# progress interval for full lmdb scans
PROGRESS_EVERY = 500_000

# optional max scan limit
MAX_ENTRIES = None

# sample size for tests
SAMPLE_SIZE = 25_000_000

# sample size for plots
PLOT_SAMPLE_SIZE = 1_000_000

# progress interval for sample scans
SAMPLE_PROGRESS_EVERY = 250_000

def collect_samples(path, sample_size, max_entries=None):
    """collect sample entries from the front of the lmdb scan."""
    qtable = LMDBQTable(path=path, readonly=True, lock=False, max_readers=1)
    samples = []
    seen = 0
    start = time.time()

    try:
        for state_int, q, visits in qtable.iter_entries(max_entries=max_entries):
            board = np.array(qtable.decode_board_from_state_int(state_int)).reshape((9, 9))

            flat = board.ravel()
            x_count = int(np.count_nonzero(flat == 1))
            o_count = int(np.count_nonzero(flat == -1))

            sample = {
                "board": board,
                "q": float(q),
                "visits": int(visits),
                "x_count": x_count,
                "o_count": o_count,
                "non_empty": x_count + o_count,
            }

            seen += 1

            if SAMPLE_PROGRESS_EVERY and (seen % SAMPLE_PROGRESS_EVERY == 0):
                elapsed = time.time() - start
                rate = seen / elapsed if elapsed > 0 else 0
                print(f"sampling: {seen:,} valid entries scanned ({rate:,.0f} entries/sec)")

            samples.append(sample)
            if len(samples) >= sample_size:
                break
    finally:
        qtable.close()

    return samples, seen


def check_board_values(samples, _game, _path=None):
    """check that sampled boards use only valid values."""
    if not samples:
        return False, "no sampled boards"
    for sample in samples:
        if not np.isin(sample["board"], [-1, 0, 1]).all():
            return False, "found board values outside {-1, 0, 1}"
    return True, f"checked {len(samples)} sampled boards"


def check_turn_balance(samples, _game, _path=None):
    """check that move counts stay balanced."""
    if not samples:
        return False, "no sampled boards"
    for sample in samples:
        if abs(sample["x_count"] - sample["o_count"]) > 1:
            return False, "found board with move imbalance > 1"
    return True, "all sampled boards respect turn balance"


def check_visits_nonnegative(samples, _game, _path=None):
    """check that visit counts are non-negative."""
    if not samples:
        return False, "no sampled boards"
    for sample in samples:
        if sample["visits"] < 0:
            return False, "found negative visit count"
    return True, "all sampled visit counts are non-negative"


def check_q_value_signal(samples, _game, _path=None):
    """check that sampled q values have some spread."""
    if not samples:
        return False, "no sampled boards"
    q_values = np.array([sample["q"] for sample in samples], dtype=float)
    spread = float(np.ptp(q_values))
    passed = spread > 1e-9 and np.any(np.abs(q_values) > 1e-6)
    return (
        passed,
        (
            f"min={q_values.min():.6f}, max={q_values.max():.6f}, "
            f"mean={q_values.mean():.6f}, std={q_values.std():.6f}"
        ),
    )


def check_visit_signal(samples, _game, _path=None):
    """check that sampled visit counts have some spread."""
    if not samples:
        return False, "no sampled boards"
    visits = np.array([sample["visits"] for sample in samples], dtype=int)
    revisited = int(np.count_nonzero(visits >= MIN_VISITS))
    return (
        revisited > 0,
        (
            f"min={visits.min()}, max={visits.max()}, mean={visits.mean():.2f}, "
            f"states with visits >= {MIN_VISITS}: {revisited}"
        ),
    )


def check_symmetry_consistency(samples, game, _path=None):
    """check symmetry and canonical-hash consistency."""
    if not samples:
        return False, "no sampled boards"
    symmetry_samples = samples[: min(len(samples), PLOT_SAMPLE_SIZE)]
    for sample in symmetry_samples:
        syms = UltimateTicTacToeGame.all_symmetries_fast(sample["board"])
        if len(syms) != 8:
            return False, f"expected 8 symmetries, got {len(syms)}"

        canonical = UltimateTicTacToeGame.canonical_board_int(sample["board"])
        sym_hashes = []
        for sym in syms:
            if sym.shape != (9, 9):
                return False, "symmetry transform produced wrong shape"
            flat = sym.ravel()
            if np.count_nonzero(flat == 1) != sample["x_count"]:
                return False, "x count changed under symmetry"
            if np.count_nonzero(flat == -1) != sample["o_count"]:
                return False, "o count changed under symmetry"
            sym_hashes.append(hashing.encode_board_to_int(flat))
            if UltimateTicTacToeGame.canonical_board_int(sym) != canonical:
                return False, "canonical hash changed under symmetry"

        if canonical != min(sym_hashes):
            return False, "canonical hash is not the minimum symmetry hash"

    return True, f"all symmetry checks passed on {len(symmetry_samples)} sampled boards"


def check_opening_preference(samples, game, path=None):
    """check whether the q-table likes the center opening."""
    del samples
    if path is None:
        return False, "missing q-table path for opening inspection"

    qtable = LMDBQTable(path=path, readonly=True, lock=False, max_readers=1)
    best_board = None
    best_row = None
    best_col = None
    best_q = None
    best_visits = -1
    center_q = None
    center_visits = 0

    try:
        for row in range(9):
            for col in range(9):
                board = np.zeros((9, 9), dtype=int)
                board[row, col] = game.agent_symbol
                state_int = UltimateTicTacToeGame.canonical_board_int(board)
                entry = qtable.get(state_int)
                if entry is None:
                    continue

                q_value = float(entry[0])
                visits = int(entry[1])

                if row == 4 and col == 4:
                    center_q = q_value
                    center_visits = visits

                if (
                    best_q is None
                    or q_value > best_q
                    or (q_value == best_q and visits > best_visits)
                ):
                    best_board = board
                    best_row = row
                    best_col = col
                    best_q = q_value
                    best_visits = visits
    finally:
        qtable.close()

    print("\nopening preference")
    if best_q is None:
        print("no opening positions were found in the q-table")
        return False, "no opening positions were found in the q-table"

    print(f"best opening: global ({best_row}, {best_col}) q={best_q:.6f}, visits={best_visits}")
    center_is_best = False
    if center_q is not None:
        center_gap = best_q - center_q
        center_is_best = abs(center_gap) <= 1e-12
        print(f"center global (4, 4) q={center_q:.6f}, visits={center_visits}")
        if center_is_best:
            print("center opening is tied for best")
        else:
            print(f"center gap from best: {center_gap:.6f}")
    else:
        print("center opening is missing from the q-table")

    print("best opening board:")
    print_ultimate_board(best_board)

    center_text = "missing"
    if center_q is not None:
        center_text = f"{center_q:.6f}"

    return (
        center_is_best,
        f"best opening is global ({best_row}, {best_col}) with q={best_q:.6f}; center q={center_text}",
    )


def run_data_tests(samples, game, path=FIXED_QTABLE_PATH):
    """run the compact inspection checks."""
    results = []
    checks = [
        ("board values are valid", check_board_values),
        ("move counts obey turn balance", check_turn_balance),
        ("visits are non-negative", check_visits_nonnegative),
        ("sampled q values have signal", check_q_value_signal),
        ("sampled visit counts have signal", check_visit_signal),
        ("symmetry pipeline is consistent", check_symmetry_consistency),
        ("opening preference favors the center", check_opening_preference),
    ]

    print("\ninspection tests")
    passed = 0
    for name, func in checks:
        ok, details = func(samples, game, path)
        results.append({"name": name, "passed": ok, "details": details})
        print(f"{'pass' if ok else 'fail'}: {name}: {details}")
        if ok:
            passed += 1

    print(f"\npassed {passed}/{len(checks)} tests\n")
    return results

def plot_sample_distributions(samples, output_dir):
    """save simple plots for the sampled data."""
    if not samples:
        print("no plot samples collected")
        return

    ensure_project_dirs()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_values = np.array([sample["q"] for sample in samples], dtype=float)
    visits = np.array([sample["visits"] for sample in samples], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(q_values, bins=60, edgecolor="black")
    ax.set_title("Sampled Q-value distribution")
    ax.set_xlabel("Q value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "q_value_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(visits, bins=60, edgecolor="black")
    ax.set_title("Sampled visit-count distribution")
    ax.set_xlabel("Visit count")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "visit_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(visits, q_values, s=5, alpha=0.3)
    ax.set_title("Q value vs visit count")
    ax.set_xlabel("Visit count")
    ax.set_ylabel("Q value")
    fig.tight_layout()
    fig.savefig(output_dir / "q_vs_visits.png", dpi=150)
    plt.close(fig)

    print(f"plots saved to: {output_dir}")

def run_inspection_suite(path=FIXED_QTABLE_PATH):
    """run the sample-based inspection suite."""
    game = UltimateTicTacToeGame(q_table={}, training=False, multiprocess=False)

    print("collecting samples for tests and plots")
    samples, seen = collect_samples(
        path,
        sample_size=SAMPLE_SIZE,
        max_entries=MAX_ENTRIES,
    )
    print(f"samples collected: {len(samples)} (from {seen} valid scanned entries)\n")

    plot_samples = samples[: min(len(samples), PLOT_SAMPLE_SIZE)]
    print(f"plot samples collected: {len(plot_samples)}\n")

    results = run_data_tests(samples, game, path=path)
    plot_sample_distributions(plot_samples, ARTIFACTS_DIR)

    return results


def inspect_lmdb(path):
    """print aggregate lmdb statistics."""
    qtable = LMDBQTable(path=path, readonly=True, lock=False, max_readers=1)

    entries = 0
    used_entries = 0

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
    valid = 0
    start = time.time()

    try:
        with qtable.begin_read() as txn:
            info = qtable.info()
            stat = txn.stat()

            print("lmdb info")
            print(f"entries: {stat['entries']}")
            print(f"page size: {qtable.page_size()} bytes")
            print(f"map size: {info['map_size'] / (1024**3):.2f} gb")
            print(f"last page no: {info['last_pgno']}")
            print()

            for k, v in qtable.iter_raw_items(txn=txn):
                entries += 1
                if MAX_ENTRIES is not None and entries >= MAX_ENTRIES:
                    break

                if PROGRESS_EVERY and (entries % PROGRESS_EVERY == 0):
                    elapsed = time.time() - start
                    print(f"{entries:,} scanned ({entries/elapsed:,.0f} entries/sec)")

                if not qtable.key_is_valid(k):
                    bad_key_len += 1
                    continue

                if not qtable.value_is_valid(v):
                    bad_val_len += 1
                    continue

                _state_int, q, visits = qtable.decode_entry(k, v)

                b = qtable.decode_board_from_key(k)
                non_empty = 0
                for cell in b:
                    if cell != 0:
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

                if visits >= MIN_VISITS:
                    valid += 1

                sum_q += q
                min_q = min(min_q, q)
                max_q = max(max_q, q)

                if q > 0:
                    positive_q += 1
                elif q < 0:
                    negative_q += 1

        print("\nsummary")
        print(f"total entries: {entries}")
        print(f"effective entries: {used_entries}")
        print(f"usable entries: {valid}")
        print(f"bad key length: {bad_key_len}")
        print(f"bad value length: {bad_val_len}")
        print()

        print("visit stats")
        print(f"min visits: {min_visits if used_entries > 0 else 'n/a'}")
        print(f"max visits: {max_visits if used_entries > 0 else 'n/a'}")
        if used_entries > 0:
            print(f"avg visits: {sum_visits / used_entries:.3f}")
        else:
            print("avg visits: n/a")
        print(f"zero-visit states: {zero_visits}")
        print()

        print("q stats")
        print(f"min q: {min_q}")
        print(f"max q: {max_q}")
        print(f"avg q: {sum_q / entries:.6f}")
        print(f"positive q states: {positive_q}")
        print(f"negative q states: {negative_q}")
        print()

        if max_visit_key is not None:
            print("most visited state")
            print(f"visits: {max_visits}")
            print(f"q-value: {max_visit_q}")
            print()
            board = qtable.decode_board_from_key(max_visit_key)
            print_ultimate_board(board)
    finally:
        qtable.close()


def print_ultimate_board(board):
    """print a 9x9 board with subboard separators."""
    def sym(x):
        return {0: "-", 1: "X", -1: "O"}.get(x, str(x))

    board = np.array(board, dtype=int).reshape((9, 9))
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
    # quick db stats and sample suite
    run_inspection_suite()
