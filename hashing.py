# hashing.py
import os
import pickle
import time

STATE_BYTES = 17  # 129 bits = 17 bytes, needed for 3^81 game states

def encode_board_to_int(board):

    value = 0
    for v in board:
        value = value * 3 + (int(v) + 1)  # force Python int, never NumPy scalar
    return value


# -----------------------------------------
# DECODING: python int → board list of 81 ints
# -----------------------------------------

def decode_board_from_int(value):
    """
    Decode the integer back to a list of 81 cells in {-1, 0, 1}.
    """
    board = [0] * 81
    for i in range(80, -1, -1):
        board[i] = (value % 3) - 1  # undo shift: 0→-1, 1→0, 2→1
        value //= 3
    return board


# -----------------------------------------
# Q-TABLE SAVE (FAST BINARY, MIN STORAGE)
# -----------------------------------------

def save_qtable(path, qtable):
    """
    Save a Q-table: dict[int → (avg, count)]
    Stored in a compact binary format using pickle.
    MUCH smaller and faster than JSON+Base64.
    """

    with open(path, "wb") as f:
        pickle.dump(qtable, f, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------
# Q-TABLE LOAD
# -----------------------------------------

def load_qtable(path):
    """
    Load Q-table from binary file.
    If missing, return empty dict.
    """

    curr_time = time.time()
    if not os.path.exists(path):
        print(f"No Q-table at {path}, starting fresh.")
        return {}

    with open(path, "rb") as f:
        qtable = pickle.load(f)

    print(f"Loaded Q-table with {len(qtable)} entries.\nThe process took {time.time() - curr_time:.1f} seconds.")
    return qtable
