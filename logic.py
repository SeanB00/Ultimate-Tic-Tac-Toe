import time

import numpy as np
import random
import hashing
class UltimateTicTacToeGame:
    def __init__(self):

        self.board_representation = np.zeros((9,9), dtype=int)
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)
        self.player_symbol = -1
        self.agent_symbol = 1
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None

        self.q_table = {}
        self.gamma = 0.9
        self.board_score_list = []
        self.players = {0: "-", 1: "X", -1: "O"}

    def update_q_table(self):

        for board, score in self.board_score_list:
            if board in self.q_table:
                avg, num_appeared = self.q_table[board]
                self.q_table[board] = ((avg * num_appeared + score) / (num_appeared + 1), num_appeared + 1)
            else:
                self.q_table[board] = (score, 1)

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
        self.board_representation = np.zeros((9,9), dtype=int)
        self.board_score_list = []


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




    def get_board_int(self):
        board = self.board_representation.flatten()
        #player is 0, noothing is 1 and 2 is agent
        hashed = hashing.encode_board_to_int(board)
        return hashed

    def get_board_from_int(self, value):
        board = hashing.decode_board_from_int(value)
        board = np.array(board).reshape(9,9)
        return board

    def play_one_game(self):
        self.init_game()
        boards = []
        while self.is_game_running():
            self.agent_random_move()
            boards.append(self.get_board_int())
            if not self.is_game_running():
                break
            self.player_random_move()
            boards.append(self.get_board_int())


        winner = self.check_true_win()
        state = winner
        boards.reverse()
        for i in range(len(boards)):
            self.board_score_list.append((boards[i], state))
            state *= self.gamma
        self.update_q_table()
        return np.array(boards)

class Games:

        def __init__(self, num_games):
            self.num_games = num_games
            self.agent_wins = 0
            self.player_wins = 0
            self.ties = 0
            self.game = UltimateTicTacToeGame()
            end_tables = []
            winners = []
            start_time = time.time()
            for _ in range(num_games):
                if _ % 1000 == 0:
                    speed = _ / (time.time() - start_time)
                    print(f"speed: {speed} games/s")
                    print(f"this is iteration {_}")
                boards = self.game.play_one_game()
                winner = self.game.check_true_win()
                if winner == 1:
                    self.agent_wins += 1
                elif winner == -1:
                    self.player_wins += 1
                else:
                    self.ties += 1
                winners.append(winner)
                end_tables.append(hashing.decode_board_from_int(boards[0]))
            self.X = np.array(end_tables)
            self.y = np.array(winners)
if __name__ == "__main__":

    games = Games(500_000)
    print("X (agent) won:", (games.agent_wins / games.num_games) * 100)
    print("O (player) won:", (games.player_wins / games.num_games) * 100)
    np.save("database_y", games.y)
    np.save("database_X", games.X)
    hashing.save_qtable("q.json",games.game.q_table)



