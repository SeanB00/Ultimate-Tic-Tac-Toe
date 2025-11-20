
import time
from copy import deepcopy

import numpy as np
import random
import hashing
from multiprocessing import Pool


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
        self.q_table = {}

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

        def find_meta_block(self):
            """
            Return (bi, bj) of the sub-board we MUST contest
            because the opponent has 2-in-a-row in the meta-board.
            Returns None if no global threat exists.
            """
            opp = self.player_symbol
            sb = self.sub_boards  # 3x3 matrix of {-1,0,1}

            # Rows
            for i in range(3):
                row = sb[i]
                if np.sum(row) == 2 * opp:
                    for j in range(3):
                        if row[j] == 0:
                            return (i, j)

            # Columns
            for j in range(3):
                col = sb[:, j]
                if np.sum(col) == 2 * opp:
                    for i in range(3):
                        if col[i] == 0:
                            return (i, j)

            # Main diagonal
            diag = np.array([sb[i][i] for i in range(3)])
            if np.sum(diag) == 2 * opp:
                for i in range(3):
                    if sb[i][i] == 0:
                        return (i, i)

            # Anti-diagonal
            diag2 = np.array([sb[i][2 - i] for i in range(3)])
            if np.sum(diag2) == 2 * opp:
                for i in range(3):
                    if sb[i][2 - i] == 0:
                        return (i, 2 - i)

            return None






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

    def find_meta_block(self):
        """
        Return (bi, bj) of the sub-board we MUST contest
        because the opponent has 2-in-a-row in the meta-board.
        Returns None if no global threat exists.
        """
        opp = self.player_symbol
        sb = self.sub_boards  # 3x3 matrix of {-1,0,1}

        # Rows
        for i in range(3):
            row = sb[i]
            if np.sum(row) == 2 * opp:
                for j in range(3):
                    if row[j] == 0:
                        return (i, j)

        # Columns
        for j in range(3):
            col = sb[:, j]
            if np.sum(col) == 2 * opp:
                for i in range(3):
                    if col[i] == 0:
                        if (i, j) in self.empty_sub_places:
                            return (i, j)

        # Main diagonal
        diag = np.array([sb[i][i] for i in range(3)])
        if np.sum(diag) == 2 * opp:
            for i in range(3):
                if sb[i][i] == 0:
                    if (i, i) in self.empty_sub_places:
                        return (i, i)

        # Anti-diagonal
        diag2 = np.array([sb[i][2 - i] for i in range(3)])
        if np.sum(diag2) == 2 * opp:
            for i in range(3):
                if sb[i][2 - i] == 0:
                    if (i, 2-i) in self.empty_sub_places:
                        return (i, 2 - i)

        return None




    def find_immidiate_danger(self, row, col):
        opp = self.player_symbol
        sb = self.full_board[row][col]  # 3x3 matrix of {-1,0,1}

        # Rows
        for i in range(3):
            row = sb[i]
            if np.sum(row) == 2 * opp:
                for j in range(3):
                    if row[j] == 0:
                        return (i, j)

        # Columns
        for j in range(3):
            col = sb[:, j]
            if np.sum(col) == 2 * opp:
                for i in range(3):
                    if col[i] == 0:
                        return (i, j)

        # Main diagonal
        diag = np.array([sb[i][i] for i in range(3)])
        if np.sum(diag) == 2 * opp:
            for i in range(3):
                if sb[i][i] == 0:
                    return (i, i)

        # Anti-diagonal
        diag2 = np.array([sb[i][2 - i] for i in range(3)])
        if np.sum(diag2) == 2 * opp:
            for i in range(3):
                if sb[i][2 - i] == 0:
                    return (i, 2 - i)



    def agent_random_move(self):

        # GLOBAL BLOCK: opponent is about to win meta-board

        # Decide playable boards

        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board
        r, c = random.choice(tuple(self.empty_places[bi][bj]))

        threat = self.find_meta_block()

        if len(self.empty_sub_places) <= 4:  # ONLY NEAR END-GAME
            if threat is not None:
                real_threat = self.find_immidiate_danger(*threat)
                if real_threat is not None:
                    if self.curr_board is None:
                        bi, bj = threat
                        r, c = real_threat
                    elif self.curr_board == threat:
                        r, c = real_threat
                    else:
                        copy = deepcopy(self.empty_places[self.curr_board])
                        copy.remove(threat)
                        if len(copy) > 0:
                            r, c = random.choice(tuple(copy))

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


def run_games_chunk(args):
    """
    Run a chunk of games in a separate process.

    args: (num_games_in_chunk, seed)
    returns:
        - local_q: dict[int -> (avg, count)]  (local Q-table updates)
        - agent_wins, player_wins, ties
        - end_states: list of final board ints (for dataset)
        - winners: list of winners (1, -1, or 0)
    """
    num_games, seed = args

    # Make randomness independent across workers
    random.seed(seed)
    np.random.seed(seed)

    game = UltimateTicTacToeGame()
    local_q = {}

    agent_wins = player_wins = ties = 0
    end_states = []
    winners = []

    for _ in range(num_games):
        game.init_game()
        boards = game.play_one_game()
        winner = game.check_true_win()

        # stats
        if winner == 1:
            agent_wins += 1
        elif winner == -1:
            player_wins += 1
        else:
            ties += 1

        winners.append(winner)
        end_states.append(boards[0])  # keep as int here

        # merge this game's board_score_list into local_q
        for board_int, reward in game.board_score_list:
            if board_int in local_q:
                avg, count = local_q[board_int]
                new_avg = (avg * count + reward) / (count + 1)
                local_q[board_int] = (new_avg, count + 1)
            else:
                local_q[board_int] = (reward, 1)

    return local_q, agent_wins, player_wins, ties, end_states, winners



class Games:

    def __init__(self, num_games, processes=None, log_every=1000, chunk_size=50):
        self.num_games = num_games

        # Stats
        self.agent_wins = 0
        self.player_wins = 0
        self.ties = 0

        # Master game just for global Q-table and helpers
        self.game = UltimateTicTacToeGame()


        print(f"Running {num_games} games with multiprocessing...")
        start_time = time.time()

        # How many chunks do we need?
        num_chunks = (num_games + chunk_size - 1) // chunk_size

        # Prepare (chunk_size, seed) for each chunk
        # Different seed per chunk to decorrelate
        args_list = []
        base_seed = int(time.time())
        for i in range(num_chunks):
            args_list.append((chunk_size, base_seed + i))

        completed_games = 0

        with Pool(processes=processes) as p:
            for local_q, a_w, p_w, t_w, end_states, winners in p.imap_unordered(
                run_games_chunk,
                args_list
            ):
                # How many games were in this chunk (last chunk may be partial)
                chunk_games = len(winners)
                completed_games += chunk_games

                # ----------------- merge stats -----------------
                self.agent_wins += a_w
                self.player_wins += p_w
                self.ties += t_w

                # ----------------- merge dataset -----------------
                # end_states are board_ints; turn into 9x9 boards


                # ----------------- merge local Q-table -----------------
                global_q = self.game.q_table
                for board_int, (avg_local, count_local) in local_q.items():
                    if board_int in global_q:
                        avg_g, count_g = global_q[board_int]
                        # merge two (avg,count) accumulators
                        total_count = count_g + count_local
                        merged_avg = (avg_g * count_g + avg_local * count_local) / total_count
                        global_q[board_int] = (merged_avg, total_count)
                    else:
                        global_q[board_int] = (avg_local, count_local)

                # ----------------- logging -----------------
                if completed_games >= log_every and completed_games % log_every < chunk_size:
                    elapsed = time.time() - start_time
                    speed = completed_games / elapsed if elapsed > 0 else 0.0
                    print(f"{completed_games}/{self.num_games} games, "
                          f"speed = {speed:.2f} games/sec")

        total_time = time.time() - start_time
        final_speed = self.num_games / total_time if total_time > 0 else 0.0

        print(f"\nFinished {self.num_games} games in {total_time:.2f}s "
              f"({final_speed:.2f} games/sec)\n")




if __name__ == "__main__":
    import multiprocessing
    cores = multiprocessing.cpu_count()
    games = Games(
        num_games=1000000,
        processes=cores,
        log_every=1000,
        chunk_size=50
    )

    print("Agent win rate:", (games.agent_wins / games.num_games) * 100)
    print("Player win rate:", (games.player_wins / games.num_games) * 100)
    print("Tie rate:", (games.ties / games.num_games) * 100)

    hashing.save_qtable("q.pkl", games.game.q_table)


