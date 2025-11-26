import time
import numpy as np
import random
import hashing
from multiprocessing import Pool

# =====================================================================
# HYPERPARAMETERS
# =====================================================================
ALPHA = 0.1          # Learning rate for TD(0) updates
EPS_START = 1.0      # Starting epsilon (full exploration)
EPS_END = 0.05       # Final epsilon (mostly exploitation)

# =====================================================================
# GLOBAL Q-TABLE FOR WORKERS (set once per worker via init_worker)
# =====================================================================
GLOBAL_QTABLE = None


def init_worker(qtable):
    """Called ONCE per worker process. Sets read-only global Q-table."""
    global GLOBAL_QTABLE
    GLOBAL_QTABLE = qtable


# =====================================================================
# ULTIMATE TIC TAC TOE GAME ENGINE
# =====================================================================
class UltimateTicTacToeGame:
    def __init__(self, q_table=None, training=False, multiprocess=False):
        # Board representations
        self.board_rep = np.zeros((9, 9), dtype=int)        # global 9x9
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int) # 3x3 subboards of 3x3
        self.sub_boards = np.zeros((3, 3), dtype=int)       # meta board

        self.player_symbol = -1
        self.agent_symbol = 1

        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None

        # Q-table handling
        if q_table is None:
            self.q_table = hashing.load_qtable("q.pkl")  # MAIN ONLY
        else:
            self.q_table = q_table  # MULTIPROCESS READ-ONLY view

        self.training = training
        self.multiprocess = multiprocess

        self.gamma = 0.9
        self.board_score_list = []
        self.players = {0: "-", 1: "X", -1: "O"}

    # ------------------------------------------------------------
    # Utility: Empty cell maps
    # ------------------------------------------------------------
    def get_empty_places(self):
        return [
            [{(i, j) for i in range(3) for j in range(3)} for _ in range(3)]
            for _ in range(3)
        ]

    def get_empty_sub_places(self):
        return {(i, j) for i in range(3) for j in range(3)}

    # ------------------------------------------------------------
    # Basic board helpers (back from your original code)
    # ------------------------------------------------------------
    def global_board(self):
        return self.board_rep

    def init_game(self):
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None
        self.board_rep = np.zeros((9, 9), dtype=int)
        self.board_score_list = []

    def is_game_running(self):
        return self.check_true_win() == 0 and not self.check_true_tie()

    def get_playable_boards(self):
        if self.curr_board is None:
            return set(self.empty_sub_places)
        if self.curr_board in self.empty_sub_places:
            return {self.curr_board}
        return set(self.empty_sub_places)

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

    # ------------------------------------------------------------
    # Win logic
    # ------------------------------------------------------------
    def check_win(self, board):
        # rows
        for row in board:
            s = np.sum(row)
            if s == 3:
                return 1
            if s == -3:
                return -1

        # columns
        for c in range(3):
            s = np.sum(board[:, c])
            if s == 3:
                return 1
            if s == -3:
                return -1

        # diagonals
        d1 = np.trace(board)
        if d1 == 3:
            return 1
        if d1 == -3:
            return -1

        d2 = np.trace(np.fliplr(board))
        if d2 == 3:
            return 1
        if d2 == -3:
            return -1

        return 0

    def tie(self, board, empty):
        return len(empty) == 0 and self.check_win(board) == 0

    def sub_board_is_done(self, bi, bj):
        return len(self.empty_places[bi][bj]) == 0 or self.sub_boards[bi][bj] != 0

    # ------------------------------------------------------------
    # Place piece in global representation
    # ------------------------------------------------------------
    def get_global_position(self, bi, bj, r, c):
        return bi * 3 + r, bj * 3 + c

    def place_in_rep(self, bi, bj, r, c, symbol):
        gi, gj = self.get_global_position(bi, bj, r, c)
        self.board_rep[gi][gj] = symbol

    # ------------------------------------------------------------
    # Q-table update (single process ONLY) - TD(0) with ALPHA
    # ------------------------------------------------------------
    def update_q_table(self):
        """
        Apply TD(0)-style updates to the global Q-table using
        the targets in self.board_score_list.

        Each entry is (state_int, target_value), and we do:
        Q <- Q + ALPHA * (target - Q)
        We still keep count as metadata.
        """
        for board_int, target in self.board_score_list:
            if board_int in self.q_table:
                old_v, count = self.q_table[board_int]
            else:
                old_v, count = 0.0, 0

            new_v = old_v + ALPHA * (target - old_v)
            self.q_table[board_int] = (new_v, count + 1)

    # ------------------------------------------------------------
    # Random opponent move
    # ------------------------------------------------------------
    def player_random_move(self):
        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board

        r, c = random.choice(tuple(self.empty_places[bi][bj]))

        self.full_board[bi][bj][r][c] = self.player_symbol
        self.place_in_rep(bi, bj, r, c, self.player_symbol)
        self.empty_places[bi][bj].remove((r, c))

        w = self.check_win(self.full_board[bi][bj])
        if w != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = w
            self.empty_sub_places.remove((bi, bj))

        next_b = (r, c)
        self.curr_board = None if self.sub_board_is_done(*next_b) else next_b

    # ------------------------------------------------------------
    # Agent move selection
    # ------------------------------------------------------------
    def agent_train_move(self, epsilon=1.0):
        if random.random() < epsilon:
            self.agent_random_move()
        else:
            self.agent_smart_move()

    def agent_smart_move(self):
        win = self.find_winning_move()
        if win is not None:
            return self.apply_agent_move(*win)

        threats = self.find_meta_block()
        if threats:
            immediate = {}
            for t in threats:
                imm = self.find_immidiate_danger(*t)
                if imm:
                    immediate[t] = imm

            if immediate:
                # block move if possible
                if self.curr_board is None:
                    moves = [
                        (bi, bj, r, c)
                        for (bi, bj), cells in immediate.items()
                        for (r, c) in cells
                    ]
                elif self.curr_board in immediate:
                    bi, bj = self.curr_board
                    moves = [(bi, bj, r, c) for (r, c) in immediate[(bi, bj)]]
                else:
                    # try to avoid sending opponent into threat boards
                    safe_moves = []
                    bi, bj = self.curr_board
                    forbidden = set(immediate.keys())
                    for (r, c) in self.empty_places[bi][bj]:
                        if (r, c) not in forbidden and (r, c) in self.empty_sub_places:
                            safe_moves.append((bi, bj, r, c))
                    if safe_moves:
                        best = self._select_best_move(safe_moves)
                        if best is None:
                            best = random.choice(safe_moves)
                        return self.apply_agent_move(*best)
                    # no safe move → random legal move
                    bi, bj = self.curr_board
                    r, c = random.choice(tuple(self.empty_places[bi][bj]))
                    return self.apply_agent_move(bi, bj, r, c)

                best = self._select_best_move(moves)
                if best is None:
                    best = random.choice(moves)
                return self.apply_agent_move(*best)

        # otherwise choose best move normally
        if self.curr_board is None:
            playable = tuple(self.empty_sub_places)
        else:
            playable = (
                [self.curr_board]
                if self.curr_board in self.empty_sub_places
                else tuple(self.empty_sub_places)
            )

        all_moves = [
            (bi, bj, r, c)
            for (bi, bj) in playable
            for (r, c) in self.empty_places[bi][bj]
        ]

        best = self._select_best_move(all_moves)
        if best is None:
            best = random.choice(all_moves)
        return self.apply_agent_move(*best)

    def agent_random_move(self):
        if self.curr_board is None:
            bi, bj = random.choice(tuple(self.empty_sub_places))
        else:
            bi, bj = self.curr_board
        r, c = random.choice(tuple(self.empty_places[bi][bj]))

        self.full_board[bi][bj][r][c] = self.agent_symbol
        self.place_in_rep(bi, bj, r, c, self.agent_symbol)
        self.empty_places[bi][bj].remove((r, c))

        w = self.check_win(self.full_board[bi][bj])
        if w != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = w
            self.empty_sub_places.remove((bi, bj))

        next_b = (r, c)
        self.curr_board = None if self.sub_board_is_done(*next_b) else next_b

    # ------------------------------------------------------------
    # Apply move
    # ------------------------------------------------------------
    def apply_agent_move(self, bi, bj, r, c):
        self.full_board[bi][bj][r][c] = self.agent_symbol
        self.place_in_rep(bi, bj, r, c, self.agent_symbol)
        self.empty_places[bi][bj].remove((r, c))

        w = self.check_win(self.full_board[bi][bj])
        if w != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = w
            self.empty_sub_places.remove((bi, bj))

        next_b = (r, c)
        self.curr_board = None if self.sub_board_is_done(*next_b) else next_b

    # ------------------------------------------------------------
    # Q-score selection helper
    # ------------------------------------------------------------
    def _select_best_move(self, moves):
        best = None
        best_score = -999

        for bi, bj, r, c in moves:
            self.place_in_rep(bi, bj, r, c, self.agent_symbol)
            board_int = self.get_board_int()
            self.place_in_rep(bi, bj, r, c, 0)

            if board_int in self.q_table:
                score, _ = self.q_table[board_int]
                if score > best_score:
                    best_score = score
                    best = (bi, bj, r, c)

        return best

    # ------------------------------------------------------------
    # Threat detection
    # ------------------------------------------------------------
    def find_meta_block(self):
        opp = self.player_symbol
        sb = self.sub_boards
        threats = set()

        # rows
        for i in range(3):
            if np.sum(sb[i]) == 2 * opp:
                for j in range(3):
                    if sb[i][j] == 0 and (i, j) in self.empty_sub_places:
                        threats.add((i, j))

        # columns
        for j in range(3):
            col = sb[:, j]
            if np.sum(col) == 2 * opp:
                for i in range(3):
                    if sb[i][j] == 0 and (i, j) in self.empty_sub_places:
                        threats.add((i, j))

        # diag
        diag = [sb[i][i] for i in range(3)]
        if sum(diag) == 2 * opp:
            for i in range(3):
                if sb[i][i] == 0:
                    threats.add((i, i))

        # anti-diag
        adiag = [sb[i][2 - i] for i in range(3)]
        if sum(adiag) == 2 * opp:
            for i in range(3):
                if sb[i][2 - i] == 0:
                    threats.add((i, 2 - i))

        return list(threats)

    def find_immidiate_danger(self, bi, bj):
        sb = self.full_board[bi][bj]
        opp = self.player_symbol
        dangers = []

        # rows
        for i in range(3):
            if np.sum(sb[i]) == 2 * opp:
                for j in range(3):
                    if sb[i][j] == 0:
                        dangers.append((i, j))

        # cols
        for j in range(3):
            col = sb[:, j]
            if np.sum(col) == 2 * opp:
                for i in range(3):
                    if sb[i][j] == 0:
                        dangers.append((i, j))

        # diag
        diag = [sb[i][i] for i in range(3)]
        if sum(diag) == 2 * opp:
            for i in range(3):
                if sb[i][i] == 0:
                    dangers.append((i, i))

        # anti diag
        adiag = [sb[i][2 - i] for i in range(3)]
        if sum(adiag) == 2 * opp:
            for i in range(3):
                if sb[i][2 - i] == 0:
                    dangers.append((i, 2 - i))

        return dangers

    # ------------------------------------------------------------
    # Winning move finder
    # ------------------------------------------------------------
    def find_winning_move(self):
        if self.curr_board is None:
            boards = self.empty_sub_places
        else:
            boards = (
                {self.curr_board}
                if self.curr_board in self.empty_sub_places
                else self.empty_sub_places
            )

        for (bi, bj) in boards:
            sb = self.full_board[bi][bj]
            meta_copy = self.sub_boards.copy()

            for (r, c) in self.empty_places[bi][bj]:
                if self._wins_sub(sb, self.agent_symbol, r, c):
                    meta_copy[bi][bj] = self.agent_symbol
                    if self._wins_meta(meta_copy, bi, bj, self.agent_symbol):
                        return bi, bj, r, c

        return None

    # helper: does move (r,c) win the subboard?
    def _wins_sub(self, sb, player, r, c):
        # row
        if all((sb[r][j] == player or (r, j) == (r, c)) for j in range(3)):
            return True
        # col
        if all((sb[i][c] == player or (i, c) == (r, c)) for i in range(3)):
            return True
        # diag
        if r == c:
            if all((sb[i][i] == player or (i, i) == (r, c)) for i in range(3)):
                return True
        # anti diag
        if r + c == 2:
            if all((sb[i][2 - i] == player or (i, 2 - i) == (r, c)) for i in range(3)):
                return True
        return False

    # helper: meta win
    def _wins_meta(self, sb, bi, bj, player):
        # row
        if all(sb[bi][j] == player for j in range(3)):
            return True
        # col
        if all(sb[i][bj] == player for i in range(3)):
            return True
        # diag
        if bi == bj:
            if all(sb[i][i] == player for i in range(3)):
                return True
        # anti diag
        if bi + bj == 2:
            if all(sb[i][2 - i] == player for i in range(3)):
                return True
        return False

    # ------------------------------------------------------------
    # Symmetry hashing
    # ------------------------------------------------------------
    def all_symmetries_fast(self, b):
        bt = b.T
        fh = np.flipud(b)
        fv = np.fliplr(b)
        fd = np.fliplr(bt)
        fad = np.flipud(bt)
        r90 = fad
        r180 = np.flipud(fv)
        r270 = fd
        return [b, r90, r180, r270, fh, fv, fd, fad]

    def canonical_board_int(self, board):
        best = None
        for sym in self.all_symmetries_fast(board):
            v = hashing.encode_board_to_int(sym.ravel())
            if best is None or v < best:
                best = v
        return best

    def get_board_int(self):
        return self.canonical_board_int(self.board_rep)

    def get_board_from_int(self, value):
        board = hashing.decode_board_from_int(value)
        return np.array(board).reshape(9, 9)

    # ------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------
    def play_one_game(self, epsilon=1.0, training=None):
        if training is None:
            training = self.training

        self.init_game()
        states = []

        while True:
            if self.check_true_win() != 0 or self.check_true_tie():
                break

            if training:
                self.agent_train_move(epsilon)
            else:
                self.agent_smart_move()

            states.append(self.get_board_int())

            if self.check_true_win() != 0 or self.check_true_tie():
                break

            self.player_random_move()
            states.append(self.get_board_int())

        winner = self.check_true_win()
        score = winner

        states.reverse()
        for s in states:
            self.board_score_list.append((s, score))
            score *= self.gamma

        if not self.multiprocess:
            self.update_q_table()

        return np.array(states)

    # final win/tie checks
    def check_true_win(self):
        return self.check_win(self.sub_boards)

    def check_true_tie(self):
        return len(self.empty_sub_places) == 0 and self.check_true_win() == 0


# =====================================================================
# WORKER: RUNS A CHUNK OF GAMES AND RETURNS Q-DELTA + STATS
# =====================================================================
def run_games_chunk(args):
    num_games, seed = args

    random.seed(seed)
    np.random.seed(seed)

    game = UltimateTicTacToeGame(
        q_table=GLOBAL_QTABLE,
        training=True,
        multiprocess=True,
    )

    local_q = {}
    agent_w = player_w = ties = 0

    for i in range(num_games):
        # simple per-chunk epsilon schedule: EPS_START -> EPS_END
        if num_games > 1:
            frac = i / (num_games - 1)
        else:
            frac = 1.0
        epsilon = EPS_START + frac * (EPS_END - EPS_START)
        game.play_one_game(epsilon=epsilon, training=True)
        w = game.check_true_win()

        if w == 1:
            agent_w += 1
        elif w == -1:
            player_w += 1
        else:
            ties += 1

        # aggregate targets for each state in this chunk
        for b, target in game.board_score_list:
            if b in local_q:
                avg, c = local_q[b]
                new_avg = (avg * c + target) / (c + 1)
                local_q[b] = (new_avg, c + 1)
            else:
                local_q[b] = (target, 1)

        game.board_score_list.clear()

    return local_q, agent_w, player_w, ties


# =====================================================================
# TRAINING MANAGER
# =====================================================================
class Games:
    def __init__(self, num_games, processes=None, log_every=1000, chunk_size=50):
        self.num_games = num_games
        self.processes = processes
        self.log_every = log_every
        self.chunk_size = chunk_size

        self.agent_wins = 0
        self.player_wins = 0
        self.ties = 0

        # main game loads q.pkl ONCE
        self.game = UltimateTicTacToeGame(
            q_table=None,
            training=False,
            multiprocess=False
        )

    # ------------------------------------------------------------
    # MULTIPROCESS TRAINING
    # ------------------------------------------------------------
    def train(self):
        print(f"Training {self.num_games} games with multiprocessing...")

        start = time.time()
        num_chunks = (self.num_games + self.chunk_size - 1) // self.chunk_size

        args_list = []
        base_seed = int(time.time())
        for i in range(num_chunks):
            args_list.append((self.chunk_size, base_seed + i))

        global GLOBAL_QTABLE
        GLOBAL_QTABLE = self.game.q_table  # main reference

        completed = 0

        with Pool(
            processes=self.processes,
            initializer=init_worker,
            initargs=(self.game.q_table,)
        ) as p:
            for local_q, a_w, p_w, t_w in p.imap_unordered(run_games_chunk, args_list):

                games_in_chunk = a_w + p_w + t_w
                completed += games_in_chunk

                self.agent_wins += a_w
                self.player_wins += p_w
                self.ties += t_w

                # ------------------------------------------------
                # MERGE TD(0)-STYLE:
                #   global_v <- global_v + ALPHA * (avg_local - global_v)
                #   counts are just accumulated as metadata.
                # ------------------------------------------------
                global_q = self.game.q_table
                for b, (avg_local, count_local) in local_q.items():
                    if b in global_q:
                        old_v, old_count = global_q[b]
                        target = avg_local
                        new_v = old_v + ALPHA * (target - old_v)
                        global_q[b] = (new_v, old_count + count_local)
                    else:
                        # first time we see this state globally
                        global_q[b] = (avg_local, count_local)

                # print speed
                if completed >= self.log_every and completed % self.log_every < self.chunk_size:
                    elapsed = time.time() - start
                    speed = completed / elapsed if elapsed > 0 else 0.0
                    print(f"{completed}/{self.num_games} games, {speed:.2f} games/sec")

        total_t = time.time() - start
        print(f"\nFinished in {total_t:.2f}s ({self.num_games / total_t:.2f} games/sec)")

    # ------------------------------------------------------------
    # SINGLE-PROCESS RUN (evaluation)
    # ------------------------------------------------------------
    def run(self):
        print(f"Running {self.num_games} games single-process (evaluation)...")
        start = time.time()

        for i in range(1, self.num_games + 1):
            # evaluation: no exploration
            self.game.play_one_game(epsilon=0.0, training=False)
            w = self.game.check_true_win()

            if w == 1:
                self.agent_wins += 1
            elif w == -1:
                self.player_wins += 1
            else:
                self.ties += 1

            if i % self.log_every == 0:
                elapsed = time.time() - start
                print(f"{i}/{self.num_games} games, {i / elapsed:.2f} games/sec")

        total_t = time.time() - start
        print(f"Done in {total_t:.2f}s ({self.num_games/total_t:.2f} games/sec)")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    import multiprocessing

    cores = multiprocessing.cpu_count()

    # 1) TRAIN
    games = Games(
        num_games=250_000,       # increase this for stronger agent
        processes=cores,
        log_every=100,
        chunk_size=200
    )

    games.run()

    print("TRAINING DONE")
    print("Agent win %:", 100 * games.agent_wins / games.num_games)
    print("Player win %:", 100 * games.player_wins / games.num_games)
    print("Tie %:", 100 * games.ties / games.num_games)

    # Save resulting Q-table
    hashing.save_qtable("q.pkl", games.game.q_table)

    #2) (OPTIONAL) EVALUATION RUN AFTER TRAINING
    eval_games = Games(num_games=5_000, processes=None, log_every=1_000)
    eval_games.game.q_table = games.game.q_table  # reuse trained table
    eval_games.run()
    print("EVAL Agent win %:", 100 * eval_games.agent_wins / eval_games.num_games)
