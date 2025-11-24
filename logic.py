
import time

import numpy as np
import random
import hashing
from multiprocessing import Pool



class UltimateTicTacToeGame:
    def __init__(self):
        # Only one board now:

        self.board_rep = np.array([[0 for _ in range(9)] for _ in range(9)])
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
        return self.board_rep

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


    def place_in_rep(self,bi,bj,i,j,symbol):


        self.board_rep[self.get_global_position(bi,bj,i,j)] = symbol

    def player_random_move(self):
        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board

        r, c = random.choice(tuple(self.empty_places[bi][bj]))

        # Apply move
        self.full_board[bi][bj][r][c] = self.player_symbol
        self.place_in_rep(bi,bj,r,c,self.player_symbol)

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

        # 1) Winning move?


        winning = self.find_winning_move()
        if winning is not None:
            bi, bj, r, c = winning
            return self.apply_agent_move(bi, bj, r, c)

        # 2) Meta threat?
        threat = self.find_meta_block()
        if threat is not None:
            real_threat = self.find_immidiate_danger(*threat)

            if real_threat is not None:

                # 2A) Can block directly?
                if self.curr_board is None:
                    bi, bj = threat
                    r, c = real_threat
                    return self.apply_agent_move(bi, bj, r, c)

                elif self.curr_board == threat:
                    bi, bj = threat
                    r, c = real_threat
                    return self.apply_agent_move(bi, bj, r, c)

                # 2B) Cannot block — avoid sending opponent into the threat board
                else:
                    safe_moves = []
                    bi0, bj0 = self.curr_board

                    for (r, c) in self.empty_places[bi0][bj0]:
                        if (r, c) != threat and (r,c) in self.empty_sub_places:  # avoiding sending opp into the meta-threat board
                            safe_moves.append((bi0, bj0, r, c))

                    if safe_moves:
                        # choose best safe move from Q-table
                        best = None
                        best_score = -999

                        for bi, bj, r, c in safe_moves:
                            self.full_board[bi][bj][r][c] = self.agent_symbol
                            board_int = self.get_board_int()
                            self.full_board[bi][bj][r][c] = 0

                            if board_int in self.q_table:
                                score, _ = self.q_table[board_int]

                                if score > best_score:
                                    best_score = score
                                    best = (bi, bj, r, c)

                        if best is not None:
                            return self.apply_agent_move(*best)
                        else:
                            return self.apply_agent_move(*random.choice(tuple(safe_moves)))
                    # fallback: no safe moves → choose randomly
                    bi, bj = self.curr_board
                    r, c = random.choice(tuple(self.empty_places[bi][bj]))
                    return self.apply_agent_move(bi, bj, r, c)

        # 3) No threats → evaluate all moves normally

        # Determine playable boards
        if self.curr_board is None:
            playable_boards = tuple(self.empty_sub_places)
        else:
            if self.curr_board in self.empty_sub_places:
                playable_boards = [self.curr_board]
            else:
                playable_boards = tuple(self.empty_sub_places)

        best = None
        best_score = -999

        for (bi, bj) in playable_boards:
            for (r, c) in self.empty_places[bi][bj]:

                # simulate move
                self.place_in_rep(bi,bj,r,c,self.agent_symbol)
                board_int = self.get_board_int()
                self.place_in_rep(bi,bj,r,c,0)

                if board_int in self.q_table:
                    score, _ = self.q_table[board_int]

                    if score > best_score:
                        best_score = score
                        best = (bi, bj, r, c)

        # If no best found → fallback
        if best is None:
            if self.curr_board is None:
                bi, bj = random.choice(tuple(self.empty_sub_places))
            else:
                bi, bj = self.curr_board

            r, c = random.choice(tuple(self.empty_places[bi][bj]))
            return self.apply_agent_move(bi, bj, r, c)

        # Apply best move

        return self.apply_agent_move(*best)

    def apply_agent_move(self, bi, bj, r, c):
        self.full_board[bi][bj][r][c] = self.agent_symbol
        self.place_in_rep(bi,bj,r,c,self.agent_symbol)
        self.empty_places[bi][bj].remove((r, c))
        next_board = (r, c)

        ws = self.check_win(self.full_board[bi][bj])
        if ws != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = ws
            self.empty_sub_places.remove((bi, bj))

        self.curr_board = None if self.sub_board_is_done(*next_board) else next_board
        return

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





    def find_immidiate_danger(self, r, c):
        opp = self.player_symbol
        sb = self.full_board[r][c]  # 3x3 matrix of {-1,0,1}

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

    def find_winning_move(self):

        # Determine which sub-boards we are allowed to play in
        if self.curr_board is None:
            boards_to_check = self.empty_sub_places
        else:
            if self.curr_board in self.empty_sub_places:
                boards_to_check = {self.curr_board}
            else:
                boards_to_check = self.empty_sub_places

        # Loop through playable boards
        for (bi, bj) in boards_to_check:

            sb = self.full_board[bi][bj]
            won = self.sub_boards.copy()

            # Check each empty move in this sub-board
            for (r, c) in self.empty_places[bi][bj]:

                # Check if placing here wins the sub-board
                if self._wins_subboard(sb, self.agent_symbol, r, c):

                    # Pretend this sub-board is won
                    won[bi][bj] = self.agent_symbol

                    # See if this produces a meta-win
                    if self._wins_meta(won, bi, bj, self.agent_symbol):
                        return bi, bj, r, c

                    # Undo is not needed because we used a copy

        return None

    def _wins_subboard(self, sb, player, r, c):
        # Check the row
        if (sb[r][0] == player or (r, 0) == (r, c)) and \
                (sb[r][1] == player or (r, 1) == (r, c)) and \
                (sb[r][2] == player or (r, 2) == (r, c)):
            return True

        # Check the column
        if (sb[0][c] == player or (0, c) == (r, c)) and \
                (sb[1][c] == player or (1, c) == (r, c)) and \
                (sb[2][c] == player or (2, c) == (r, c)):
            return True

        # Main diag
        if r == c:
            if (sb[0][0] == player or (0, 0) == (r, c)) and \
                    (sb[1][1] == player or (1, 1) == (r, c)) and \
                    (sb[2][2] == player or (2, 2) == (r, c)):
                return True

        # Anti diag
        if r + c == 2:
            if (sb[0][2] == player or (0, 2) == (r, c)) and \
                    (sb[1][1] == player or (1, 1) == (r, c)) and \
                    (sb[2][0] == player or (2, 0) == (r, c)):
                return True

        return False

    def _wins_meta(self, won, bi, bj, player):

        # Row in meta
        if np.all(won[bi] == player):
            return True

        # Column in meta
        if np.all(won[:, bj] == player):
            return True

        # Main diagonal
        if bi == bj:
            if won[0][0] == player and won[1][1] == player and won[2][2] == player:
                return True

        # Anti diagonal
        if bi + bj == 2:
            if won[0][2] == player and won[1][1] == player and won[2][0] == player:
                return True

        return False

    def agent_random_move(self):

        # GLOBAL BLOCK: opponent is about to win meta-board

        # Decide playable boards

        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board
        r, c = random.choice(tuple(self.empty_places[bi][bj]))


        winning = self.find_winning_move()
        if winning is not None:
            bi,bj,r,c = winning
        else:
            threat = self.find_meta_block()
            if threat is not None:
                real_threat = self.find_immidiate_danger(*threat)
                if real_threat is not None:
                    if self.curr_board is None:
                        bi, bj = threat
                        r, c = real_threat
                    elif self.curr_board == threat:
                        r, c = real_threat
                    else:
                        empty = self.empty_places[self.curr_board[0]][self.curr_board[1]]
                        copy = [val for val in empty if val != threat and val in self.empty_sub_places]
                        if len(copy) > 0:
                           r, c = random.choice(copy)

        # Apply move
        self.full_board[bi][bj][r][c] = self.agent_symbol
        self.place_in_rep(bi,bj,r,c,self.agent_symbol)
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
        self.board_rep = np.array([[0 for _ in range(9)] for _ in range(9)])

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

    def all_symmetries_fast(self, board):
        """
        Return the 8 symmetries of a square board using fast flips + transpose.
        board: numpy array (9x9)
        """
        b = board
        bt = b.T  # transpose = main diagonal reflection

        # flips
        fh = np.flipud(b)  # horizontal flip (top <-> bottom)
        fv = np.fliplr(b)  # vertical flip (left <-> right)

        # diagonal flips
        fd = np.fliplr(bt)  # reflect main diagonal
        fad = np.flipud(bt)  # reflect anti-diagonal

        # rotations
        r90 = fad  # rotate 90° clockwise
        r180 = np.flipud(fv)  # rotate 180°
        r270 = fd  # rotate 270° clockwise

        return [b, r90, r180, r270, fh, fv, fd, fad]

    def canonical_board_int(self, board):
        """
        Compute canonical integer for board using fast symmetries.
        board: 9x9 numpy array (global board)
        """
        best = None

        for sym in self.all_symmetries_fast(board):
            val = hashing.encode_board_to_int(sym.ravel())
            if best is None or val < best:
                best = val

        return best

    def get_board_int(self):
        # canonical 9x9 representation
        return self.canonical_board_int(self.global_board())

    def get_board_from_int(self, value):
        board = hashing.decode_board_from_int(value)
        return np.array(board).reshape(9, 9)

    # ---------------------------------------------------------
    # Game loop
    # ---------------------------------------------------------
    def agent_train_move(self, epsilon=1.1):
        # ε-greedy policy
        if random.random() < epsilon:
            # exploration
            self.agent_random_move()
        else:
            # exploitation
            self.agent_smart_move()

    def play_one_game(self, epsilon=0.1, training=False):
        self.init_game()
        boards = []

        while self.is_game_running():

            if training:
                self.agent_train_move()
            else:
                self.agent_smart_move()
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
        self.chunk_size = chunk_size
        self.log_every = log_every
        self.processes = processes

        # Master game just for global Q-table and helpers
        self.game = UltimateTicTacToeGame()
    def train(self):

        print(f"Running {self.num_games} games with multiprocessing...")
        start_time = time.time()

        # How many chunks do we need?
        num_chunks = (self.num_games + self.chunk_size - 1) // self.chunk_size

        # Prepare (chunk_size, seed) for each chunk
        # Different seed per chunk to decorrelate
        args_list = []
        base_seed = int(time.time())
        for i in range(num_chunks):
            args_list.append((self.chunk_size, base_seed + i))

        completed_games = 0

        with Pool(processes=self.processes) as p:
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
                if completed_games >= self.log_every and completed_games % self.log_every < self.chunk_size:
                    elapsed = time.time() - start_time
                    speed = completed_games / elapsed if elapsed > 0 else 0.0
                    print(f"{completed_games}/{self.num_games} games, "
                          f"speed = {speed:.2f} games/sec")

        total_time = time.time() - start_time
        final_speed = self.num_games / total_time if total_time > 0 else 0.0

        print(f"\nFinished {self.num_games} games in {total_time:.2f}s "
              f"({final_speed:.2f} games/sec)\n")

    def run(self):

        print(f"Running {self.num_games} games (single process)...")
        start_time = time.time()

        for i in range(1, self.num_games + 1):
            boards = self.game.play_one_game()
            winner = self.game.check_true_win()

            # Stats
            if winner == 1:
                self.agent_wins += 1
            elif winner == -1:
                self.player_wins += 1
            else:
                self.ties += 1

            # Logging
            if i % self.log_every == 0:
                elapsed = time.time() - start_time
                speed = i / elapsed if elapsed else 0
                print(f"{i}/{self.num_games} games, speed = {speed:.2f} games/sec")

        total_time = time.time() - start_time
        final_speed = self.num_games / total_time if total_time else 0

        print(f"\nFinished {self.num_games} games in {total_time:.2f}s "
              f"({final_speed:.2f} games/sec)\n")


if __name__ == "__main__":
    import multiprocessing
    cores = multiprocessing.cpu_count()
    games = Games(
        num_games=2_500,
        processes=cores,
        log_every=1000,
        chunk_size=100
    )
    games.run()
    print("Agent win rate:", (games.agent_wins / games.num_games) * 100)
    print("Player win rate:", (games.player_wins / games.num_games) * 100)
    print("Tie rate:", (games.ties / games.num_games) * 100)
    #hashing.save_qtable("q.pkl", games.game.q_table)


