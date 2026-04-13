import os
import time


def encode_board_to_int(board):
    """encode a flat board into an integer."""
    value = 0
    for v in board:
        value = value * 3 + (int(v) + 1)
    return value


# decoding
def decode_board_from_int(value):
    """decode an integer into a flat board."""
    board = [0] * 81
    for i in range(80, -1, -1):
        board[i] = (value % 3) - 1
        value //= 3
    return board
