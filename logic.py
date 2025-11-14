import numpy as np
import random

class game():
    def __init__(self):

        self.board_representation = np.zeros((9,9), dtype=int)
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)
        self.player_symbol = -1
        self.agent_symbol = 1
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None

        self.players = {0: "-", 1: "X", -1: "O"}



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


    def sub_board_is_done(self, i, j):
        #checks if there is a win or draw in the sub-board
        return len(self.empty_places[i][j]) == 0 or self.sub_boards[i][j] != 0

    def get_global_position(self, board_i, board_j, cell_row, cell_col):

        global_row = board_i * 3 + cell_row
        global_col = board_j * 3 + cell_col
        return (global_row, global_col)

    def player_random_move(self):

        #The curr_board tells us which board we need to play in, if None then anywhere

        # If curr_board is None, choose a random board that has empty places
        available_boards = tuple(self.empty_sub_places)

        if self.curr_board is None:
            board_i, board_j = random.choice(available_boards)
        else:
            board_i, board_j = self.curr_board
            # Check if current board is still playable

        # Choose a random empty position in the selected board
        empty_positions = tuple(self.empty_places[board_i][board_j])
        row, col = random.choice(empty_positions)

        # Make the move
        self.full_board[board_i][board_j][row][col] = self.player_symbol
        self.board_representation[self.get_global_position(board_i, board_j, row, col)] = self.player_symbol


        self.empty_places[board_i][board_j].remove((row, col))
        next_board = (row, col)

        # Check if this move wins the sub-board
        win_status = self.check_win(self.full_board[board_i][board_j])
        if win_status != 0 or self.tie(self.full_board[board_i][board_j], self.empty_places[board_i][board_j]):
            self.sub_boards[board_i][board_j] = win_status
            self.empty_sub_places.remove((board_i, board_j))

        if self.sub_board_is_done(*next_board):
            self.curr_board = None
        else:
            self.curr_board = next_board

    def agent_random_move(self):

        # The curr_board tells us which board we need to play in, if None then anywhere

        # If curr_board is None, choose a random board that has empty places
        available_boards = tuple(self.empty_sub_places)

        if self.curr_board is None:
            board_i, board_j = random.choice(available_boards)
        else:
            board_i, board_j = self.curr_board
            # Check if current board is still playable

        # Choose a random empty position in the selected board
        empty_positions = tuple(self.empty_places[board_i][board_j])
        row, col = random.choice(empty_positions)

        # Make the move
        self.full_board[board_i][board_j][row][col] = self.agent_symbol
        self.board_representation[self.get_global_position(board_i, board_j, row, col)] = self.agent_symbol

        self.empty_places[board_i][board_j].remove((row, col))
        next_board = (row, col)

        # Check if this move wins the sub-board
        win_status = self.check_win(self.full_board[board_i][board_j])
        if win_status != 0 or self.tie(self.full_board[board_i][board_j], self.empty_places[board_i][board_j]):
            self.sub_boards[board_i][board_j] = win_status
            self.empty_sub_places.remove((board_i, board_j))

        if self.sub_board_is_done(*next_board):
            self.curr_board = None
        else:
            self.curr_board = next_board

    def check_true_win(self):
        return self.check_win(self.sub_boards)

    def check_true_tie(self):
        return len(self.empty_sub_places) == 0 and self.check_true_win() == 0

    def init_game(self):
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None


    def is_game_running(self):
        return not self.check_true_tie() and self.check_true_win() == 0


    def get_true_board_string(self):
        s = ""
        boardNumpy = self.board_representation
        for row in range(9):
            printable_row = [self.players[x] for x in boardNumpy[row]]
            s += "".join(printable_row)
        return s

    def print_board(self):
        s = ""
        boardNumpy = self.board_representation
        for row in range(9):
            s += "|"
            for col in range(9):
                s += self.players[boardNumpy[row][col]]
                s += "|"
                if col % 3 == 2 and col != 8:
                    s += "  "
                    s += "|"
            s += '\n'
            if row % 3 == 2:
                s += "\n"
        print(s)
        print("".join(["-" for i in range(9)]))





    def play_one_game(self):
        self.init_game()
        self.print_board()
        while self.is_game_running():
            self.agent_random_move()
            self.print_board()
            if not self.is_game_running():
                break
            self.player_random_move()
            self.print_board()

        winner = self.check_true_win()
        print(f"winner is {winner}")








g = game()
g.play_one_game()


