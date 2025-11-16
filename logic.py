import time
import numpy as np
import random
import hashing

class UltimateTicTacToeGame:
    def __init__(self):
        # Only one board now:
        # full_board[bi][bj][r][c]
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)

        self.player_symbol = -1
        self.agent_symbol = 1

        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None

        # Q-table stored in fast binary format
        self.q_table = hashing.load_qtable("q.pkl")

        self.gamma = 0.9
        self.board_score_list = []
        self.players = {0: "-", 1: "X", -1: "O"}

    # ---------------------------------------------------------
    # Utility: Generates the set of empty cells for each sub-board
    # ---------------------------------------------------------
    def get_empty_places(self):
        return [[{(i, j) for i in range(3) for j in range(3)} for _ in range(3)] for _ in range(3)]

    def get_empty_sub_places(self):
        return {(i, j) for i in range(3) for j in range(3)}

    # ---------------------------------------------------------
    # Win/tie checks
    # ---------------------------------------------------------
    def check_win(self, board):
        # Rows
        for row in board:
            s = np.sum(row)
            if s == 3: return 1
            if s == -3: return -1

        # Columns
        for col in range(3):
            s = np.sum(board[:, col])
            if s == 3: return 1
            if s == -3: return -1

        # Diagonals
        diag = np.trace(board)
        if diag == 3: return 1
        if diag == -3: return -1

        diag2 = np.trace(np.fliplr(board))
        if diag2 == 3: return 1
        if diag2 == -3: return -1

        return 0

    def tie(self, board, empty):
        return len(empty) == 0 and self.check_win(board) == 0

    def sub_board_is_done(self, i, j):
        return (len(self.empty_places[i][j]) == 0) or (self.sub_boards[i][j] != 0)

    # ---------------------------------------------------------
    # Board indexing utilities
    # ---------------------------------------------------------
    def global_board(self):
        # Zero-copy view of full_board as 9x9
        return self.full_board.reshape(9, 9)

    def get_global_position(self, bi, bj, r, c):
        return bi * 3 + r, bj * 3 + c

    # ---------------------------------------------------------
    # Moves
    # ---------------------------------------------------------
    def update_q_table(self):

        for board, score in self.board_score_list:
            if board in self.q_table:
                avg, num_appeared = self.q_table[board]
                self.q_table[board] = ((avg * num_appeared + score) / (num_appeared + 1), num_appeared + 1)
            else:
                self.q_table[board] = (score, 1)



    def player_random_move(self):
        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board

        r, c = random.choice(tuple(self.empty_places[bi][bj]))

        # Apply move
        self.full_board[bi][bj][r][c] = self.player_symbol

        # Remove from empty cells
        self.empty_places[bi][bj].remove((r, c))
        next_board = (r, c)

        # Check sub-board win/tie
        win_status = self.check_win(self.full_board[bi][bj])
        if win_status != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = win_status
            self.empty_sub_places.remove((bi, bj))

        # Determine next forced board
        self.curr_board = None if self.sub_board_is_done(*next_board) else next_board

    def agent_smart_move(self):
        # Decide which sub-boards are playable
        if self.curr_board is None:
            boards = tuple(self.empty_sub_places)
        else:
            if self.curr_board in self.empty_sub_places:
                boards = [self.curr_board]
            else:
                boards = tuple(self.empty_sub_places)

        best = None
        best_score = -1

        for bi, bj in boards:
            for r, c in self.empty_places[bi][bj]:

                # Simulate move
                self.full_board[bi][bj][r][c] = self.agent_symbol

                board_int = self.get_board_int()

                if board_int in self.q_table:
                    score, _ = self.q_table[board_int]
                else:
                    ws = self.check_win(self.full_board[bi][bj])
                    if ws != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
                        self.sub_boards[bi][bj] = ws
                    score = self._evaluate_board_heuristic()
                    self.sub_boards[bi][bj] = 0

                # Undo
                self.full_board[bi][bj][r][c] = 0

                if score > best_score:
                    best_score = score
                    best = (bi, bj, r, c)

        if best is None:
            self.player_random_move()
            return

        bi, bj, r, c = best

        # Apply final move
        self.full_board[bi][bj][r][c] = self.agent_symbol
        self.empty_places[bi][bj].remove((r, c))

        next_board = (r, c)

        ws = self.check_win(self.full_board[bi][bj])
        if ws != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = ws
            self.empty_sub_places.remove((bi, bj))

        self.curr_board = None if self.sub_board_is_done(*next_board) else next_board

    def agent_random_move(self):
        # Decide playable boards
        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board

        # Random empty cell (O(1))
        r, c = random.choice(tuple(self.empty_places[bi][bj]))

        # Apply move
        self.full_board[bi][bj][r][c] = self.agent_symbol
        self.empty_places[bi][bj].remove((r, c))
        next_board = (r, c)

        # Check sub-board win/tie
        ws = self.check_win(self.full_board[bi][bj])
        if ws != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = ws
            self.empty_sub_places.remove((bi, bj))

        # Next forced board
        self.curr_board = None if self.sub_board_is_done(*next_board) else next_board

    # ---------------------------------------------------------
    # Heuristic
    # ---------------------------------------------------------
    def _evaluate_board_heuristic(self):
        main_win = self.check_true_win()

        if main_win == self.agent_symbol: return 1.0
        if main_win == self.player_symbol: return -1.0

        score = 0.0
        for i in range(3):
            for j in range(3):
                if self.sub_boards[i][j] == self.agent_symbol:
                    score += 0.1
                elif self.sub_boards[i][j] == self.player_symbol:
                    score -= 0.1
        return score

    # ---------------------------------------------------------
    # Game state logic
    # ---------------------------------------------------------
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
        self.board_score_list = []

    def is_game_running(self):
        return self.check_true_win() == 0 and not self.check_true_tie()

    # ---------------------------------------------------------
    # Display utilities
    # ---------------------------------------------------------
    def print_board(self):
        board = self.global_board()
        s = ""
        for row in range(9):
            s += "|"
            for col in range(9):
                s += self.players[board[row][col]] + "|"
                if col % 3 == 2 and col != 8:
                    s += "  |"
            s += "\n"
            if row % 3 == 2:
                s += "\n"
        print(s)
        print("-" * 9)

    # ---------------------------------------------------------
    # Hashing utilities
    # ---------------------------------------------------------
    def get_board_int(self):
        return hashing.encode_board_to_int(self.global_board().ravel())

    def get_board_from_int(self, value):
        board = hashing.decode_board_from_int(value)
        return np.array(board).reshape(9, 9)

    # ---------------------------------------------------------
    # Game loop
    # ---------------------------------------------------------
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
        for b in boards:
            self.board_score_list.append((b, state))
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

        end_states = []
        winners = []

        start = time.time()

        for i in range(num_games):
            if i > 0 and i % 1000 == 0:
                speed = i / (time.time() - start)
                print(f"{i} games, speed = {speed:.1f} games/sec")

            boards = self.game.play_one_game()
            winner = self.game.check_true_win()

            if winner == 1:
                self.agent_wins += 1
            elif winner == -1:
                self.player_wins += 1
            else:
                self.ties += 1

            winners.append(winner)
            end_states.append(self.game.get_board_from_int(boards[0]))

        self.X = np.array(end_states)
        self.y = np.array(winners)


if __name__ == "__main__":
    games = Games(10_000)
    print("X (agent) won:", (games.agent_wins / games.num_games) * 100)
    print("O (player) won:", (games.player_wins / games.num_games) * 100)
    hashing.save_qtable("q.pkl", games.game.q_table)
