import base64
import json
from math import ceil, log2

# -------------------------------------
# 1. PACK board -> 17-byte compact form
# -------------------------------------


def encode_board_to_int(board):
    value = 0
    for i in range(len(board)):
        board[i] += 1
    for x in board:
        value = value * 3 + int(x)
    return value

def decode_board_from_int(value):
    board = [0] * 81
    for i in range(80, -1, -1):
        board[i] = value % 3
        value //= 3
    for i in range(len(board)):
        board[i] -= 1
    return board


def save_qtable(path, qtable):
    """
    Saves a Q-table where keys are ints.
    Converts keys to Base64 strings for JSON.
    """
    data = {}
    for board_int, q in qtable.items():
        b = board_int.to_bytes(17, byteorder='big')  # 17 bytes for 3^81
        b64 = base64.b64encode(b).decode('ascii')
        data[b64] = [q[0],q[1]]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)



# -------------------------------------
# 4. LOAD Q-TABLE FROM JSON
# Returns: dict mapping b64state -> float
# -------------------------------------

def load_qtable(path):
    """
    Loads a Q-table from JSON, converts keys from Base64 -> int
    Returns: dict mapping int_board -> float Q-value
    """
    with open(path, "r") as f:
        data = json.load(f)

    qtable = {}
    for b64, q in data.items():
        # Decode Base64 -> bytes -> int
        b = base64.b64decode(b64)
        board_int = int.from_bytes(b, byteorder='big')
        qtable[board_int] = tuple(q)

    return qtable

