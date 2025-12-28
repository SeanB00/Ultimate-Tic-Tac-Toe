import lmdb
import struct

import numpy as np

import hashing
import time

from logic import UltimateTicTacToeGame

# ===== CONFIG =====
LMDB_PATH = "qtable.lmdb"
KEY_BYTES = 32
VALUE_FMT = "di"            # (double q_value, int count)
VALUE_SIZE = struct.calcsize(VALUE_FMT)
NUM_PLAYS = 10

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


def check_symmetries():
    import logic
    game = UltimateTicTacToeGame(q_table={}, training=False, multiprocess=False)
    import numpy as np

    b =board_9x9 = np.array([
        [ 1,  0, -1,   0,  0,  0,   -1,  0,  1],
        [ 0,  1,  0,   0,  0,  1,    0, -1,  0],
        [-1,  0,  1,   0, -1,  0,    1,  0, -1],

        [ 0, -1,  0,   1,  0, -1,    0,  1,  0],
        [ 1,  0,  1,   0,  1,  0,   -1,  0, -1],
        [ 0,  1,  0,  -1,  0,  1,    0, 0,  0],

        [-1,  0,  1,   0, -1,  0,    1,  0, -1],
        [ 0, -1,  0,   1,  0, -1,    0,  0,  0],
        [ 0,  0, -1,   0,  1,  0,   -1,  0,  1],
    ], dtype=int)

    symmetries = game.all_symmetries_fast(b)
    boards = set()
    new_set = set()

    for board in symmetries:
        boards.add(game.canonical_board_int(board))




    for board in boards:
        b = np.array(hashing.decode_board_from_int(board)).reshape((9,9))
        print(np.array(hashing.decode_board_from_int(board)).reshape((9,9)))

        print("*"*28)
        new_set.add(game.canonical_board_int(b))
    print(boards)
    print(new_set)


    pass

if __name__ == "__main__":



    check_symmetries()
    inspect_lmdb(LMDB_PATH)
