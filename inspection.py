import lmdb
import struct
import hashing
import time

# ===== CONFIG =====
LMDB_PATH = "qtable.lmdb"
KEY_BYTES = 32
VALUE_FMT = "di"            # (double q_value, int count)
VALUE_SIZE = struct.calcsize(VALUE_FMT)

PROGRESS_EVERY = 500_000     # print progress every N entries
# ==================


def decode_state(key_bytes: bytes):
    state_int = int.from_bytes(key_bytes, "big", signed=False)
    return hashing.decode_board_from_int(state_int)


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

        for k, v in cursor:
            entries += 1

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
    print(f"Bad key length       : {bad_key_len}")
    print(f"Bad value length     : {bad_val_len}")
    print()

    print("===== VISIT STATS =====")
    print(f"Min visits           : {min_visits}")
    print(f"Max visits           : {max_visits}")
    print(f"Avg visits           : {sum_visits / entries:.3f}")
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
    print_ultimate_board(board)


def print_ultimate_board(board):
    def sym(x):
        return {0: ".", 1: "X", -1: "O"}.get(x, str(x))

    for big_r in range(3):
        for small_r in range(3):
            row = []
            for big_c in range(3):
                idx = big_r * 3 + big_c
                sub = board[idx]
                row.append(" ".join(sym(sub[small_r][c]) for c in range(3)))
            print(" | ".join(row))
        if big_r < 2:
            print("-" * 29)


if __name__ == "__main__":
    inspect_lmdb(LMDB_PATH)
