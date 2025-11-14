import numpy as np
import random

class tic_tac_toe_one_game():
    def __init__(self):
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)
        self.player_symbol = -1
        self.agent_symbol = 1
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None

    def get_empty_places(self):
        return np.array([[{(i, j) for i in range(3) for j in range(3)} for col in range(3)] for row in range(3)])
    def get_empty_sub_places(self):
        return {(i,j) for i in range(3) for j in range(3)}


    def check_win(self, board):
        """
        checks if there is a win in the board
        1 - agent
        0 - no win
        -1 - player
        """

        # בדיקת שורות
        for row in board:
            if np.sum(row) == 3:
                return 1
            if np.sum(row) == -3:
                return -1

        # בדיקת עמודות
        for col_idx in range(3):
            col_sum = np.sum(board[:, col_idx])
            if col_sum == 3:
                return 1
            if col_sum == -3:
                return -1

        # בדיקת אלכסונים
        trace = np.trace(board)
        flipped_trace = np.trace(np.fliplr(board))
        if trace == 3:
            return 1
        if trace == -3:
            return -1
        if flipped_trace == 3:
            return 1
        if flipped_trace == -3:
            return -1

        return 0  # אין מנצח

    def tie(self, board, empty):
        """
        בודקת אם המשחק הסתיים בתיקו.
        מחזירה True אם תיקו, אחרת False.
        """
        return len(empty) == 0 and self.check_win(board) == 0



    def is_full(self, i, j):
        return len(self.empty_places[i][j]) > 0 and self.sub_boards[i][j] == 0


    def player_random_move(self):


        # If curr_board is None, choose a random board that has empty places

        available_boards = tuple(self.empty_sub_places)
        if self.curr_board is None:

            if not available_boards:
                return None  # No moves available
            board_i, board_j = random.choice(available_boards)
        else:
            board_i, board_j = self.curr_board
            # Check if current board is still playable
            if self.is_full(board_i, board_j):
                # Current board is full or won, choose any available board
                if not available_boards:
                    return None  # No moves available
                board_i, board_j = random.choice(available_boards)

        # Choose a random empty position in the selected board
        empty_positions = list(self.empty_places[board_i][board_j])
        if not empty_positions:
            return None
        row, col = random.choice(empty_positions)

        # Make the move
        self.full_board[board_i][board_j][row][col] = self.player_symbol
        self.empty_places[board_i][board_j].remove((row, col))

        # Check if this move wins the sub-board
        win_status = self.check_win(self.full_board[board_i][board_j])
        if win_status != 0:
            self.sub_boards[board_i][board_j] = win_status
        elif self.tie(board_i, board_j):
            self.sub_boards[board_i][board_j] = 0  # Mark as tie
            self.empty_sub_places.remove((board_i, board_j))

        # Set next board based on the position played
        next_board = (row, col)
        # Check if next board is playable
        if len(self.empty_places[row][col]) > 0 and self.sub_boards[row][col] == 0:
            self.curr_board = next_board
        else:
            self.curr_board = None  # Player can choose any board

        return (board_i, board_j, row, col)
