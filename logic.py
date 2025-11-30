import time
import random
import multiprocessing
import numpy as np
import hashing
from multiprocessing import Pool
from lmdb_qtable import LMDBQTable

EPS_START = 1.0      # Starting epsilon (full exploration)
EPS_END = 0.05       # Final epsilon (mostly exploitation)

#As training grows with exploit more

GLOBAL_QTABLE = None


def init_worker(qtable):
    import lmdb_qtable        # <-- crucial
    global GLOBAL_QTABLE
    GLOBAL_QTABLE = qtable

    lmdb_qtable.GLOBAL_TXN = qtable.env.begin(write=False)




# =====================================================================
# ULTIMATE TIC TAC TOE GAME ENGINE
# =====================================================================
class UltimateTicTacToeGame:
    """
    Ultimate Tic Tac Toe environment + Q-learning integration.

    - full_board: 3x3 subboards, each 3x3
    - sub_boards: 3x3 meta board of who won each subboard
    - board_rep:  9x9 flattened representation for hashing / symmetry
    """

    def __init__(self, q_table=None, training=False, multiprocess=False):
        # Board representations
        self.board_rep = np.zeros((9, 9), dtype=int)         # global 9x9
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)  # 3x3 subboards of 3x3
        self.sub_boards = np.zeros((3, 3), dtype=int)        # meta board

        # Symbols
        self.player_symbol = -1
        self.agent_symbol = 1

        # Counters
        self.num_plays = 0       # total agent moves
        self.random_plays = 0    # times we FALL BACK to random (no Q info / forced random)

        # State tracking
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None

        # Q-table handling
        # MAIN process: load from disk if q_table is None
        # WORKER processes: q_table is passed in and treated as read-only
        if q_table is None:
            self.q_table = LMDBQTable("qtable.lmdb")
        else:
            self.q_table = q_table

        self.training = training
        self.multiprocess = multiprocess

        # Discount factor
        self.gamma = 0.9

        # (state_int, target_value) accumulated during game
        self.board_score_list = []

        # For printing
        self.players = {0: "-", 1: "X", -1: "O"}

    # ------------------------------------------------------------
    # Utility: Empty cell maps
    # ------------------------------------------------------------
    def get_empty_places(self):
        """
        Returns 3x3 grid where each cell is a set of empty (r, c)
        positions inside that subboard.
        """
        return [
            [{(i, j) for i in range(3) for j in range(3)} for _ in range(3)]
            for _ in range(3)
        ]

    def get_empty_sub_places(self):
        """
        Returns set of (bi, bj) indices of subboards that are still playable.
        """
        return {(i, j) for i in range(3) for j in range(3)}

    # ------------------------------------------------------------
    # Basic board helpers
    # ------------------------------------------------------------
    def global_board(self):
        return self.board_rep

    def init_game(self):
        """
        Reset game state to start a new game.
        """
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
        """
        Return the set of playable subboards given the current rules.
        """
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
    # Win / tie logic
    # ------------------------------------------------------------
    def check_win(self, board):
        """
        Check win on a 3x3 board: returns 1, -1 or 0.
        """
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

        # main diagonal
        d1 = np.trace(board)
        if d1 == 3:
            return 1
        if d1 == -3:
            return -1

        # anti-diagonal
        d2 = np.trace(np.fliplr(board))
        if d2 == 3:
            return 1
            # noqa
        if d2 == -3:
            return -1

        return 0

    def tie(self, board, empty):
        """
        Returns True if the given 3x3 board is a tie (no win, no moves).
        """
        return len(empty) == 0 and self.check_win(board) == 0

    def sub_board_is_done(self, bi, bj):
        """
        Returns True if the subboard (bi, bj) is no longer playable.
        """
        return len(self.empty_places[bi][bj]) == 0 or self.sub_boards[bi][bj] != 0

    # ------------------------------------------------------------
    # Place piece in global representation
    # ------------------------------------------------------------
    def get_global_position(self, bi, bj, r, c):
        """
        Map local (bi, bj, r, c) to global (gi, gj) in 9x9 board_rep.
        """
        return bi * 3 + r, bj * 3 + c

    def place_in_rep(self, bi, bj, r, c, symbol):
        """
        Place symbol in board_rep at the mapped global index.
        """
        gi, gj = self.get_global_position(bi, bj, r, c)
        self.board_rep[gi][gj] = symbol

    # ------------------------------------------------------------
    # Q-table update (single process ONLY) - AVERAGING POLICY
    # ------------------------------------------------------------
    def update_q_table(self):
        # LMDB backend
        if hasattr(self.q_table, "update_from_targets"):
            self.q_table.update_from_targets(self.board_score_list)
        else:
            # fallback to dict
            for board_int, target in self.board_score_list:
                if board_int in self.q_table:
                    old_v, count = self.q_table[board_int]
                else:
                    old_v, count = 0.0, 0

                new_v = (old_v * count + target) / (count + 1)
                self.q_table[board_int] = (new_v, count + 1)

    # ------------------------------------------------------------
    # Random opponent move
    # ------------------------------------------------------------
    def player_random_move(self):
        """
        Opponent plays a purely random legal move.
        """
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
        """
        Epsilon-greedy agent move during training.
        Exploration moves (epsilon branch) do NOT count as
        'fallback randomness' – random_plays is strictly for cases
        where Q-table couldn't guide us.
        """
        if random.random() < epsilon:
            # Pure exploration: random move, not counted as missing-Q fallback
            self.agent_random_move()
        else:
            self.agent_smart_move()
            self.num_plays += 1

    def agent_smart_move(self):
        """
        Deterministic agent move (no exploration) using:
        1) Immediate meta winning moves
        2) Blocking opponent meta threats
        3) Q-table based selection on canonical board states
        """
        # 1) Try to win the meta game if possible
        win = self.find_winning_move()
        if win is not None:
            return self.apply_agent_move(*win)

        # 2) Look for meta threats and immediate dangers in those subboards
        threats = self.find_meta_block()
        if threats:
            immediate = {}
            for t in threats:
                imm = self.find_immidiate_danger(*t)
                if imm:
                    immediate[t] = imm

            if immediate:
                # If multiple boards are threatened, try blocking in them
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
                    # Try to avoid sending the opponent into threatened boards
                    safe_moves = []
                    bi, bj = self.curr_board
                    forbidden = set(immediate.keys())
                    for (r, c) in self.empty_places[bi][bj]:
                        if (r, c) not in forbidden and (r, c) in self.empty_sub_places:
                            safe_moves.append((bi, bj, r, c))
                    if safe_moves:
                        # Use Q-best among safe moves; fallback handled inside _select_best_move
                        best = self._select_best_move(safe_moves)
                        return self.apply_agent_move(*best)

                    # No safe move → forced random legal move in current subboard
                    bi, bj = self.curr_board
                    r, c = random.choice(tuple(self.empty_places[bi][bj]))
                    # This is a REAL fallback random (no Q-guided safe option)
                    self.random_plays += 1
                    return self.apply_agent_move(bi, bj, r, c)

                # We have blocking moves, choose Q-best among them
                best = self._select_best_move(moves)
                return self.apply_agent_move(*best)

        # 3) Otherwise choose best move normally using Q-table
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
        return self.apply_agent_move(*best)

    def agent_random_move(self):
        """
        Agent plays a random legal move.
        Used ONLY for epsilon exploration; does NOT increment random_plays,
        because random_plays is reserved for missing-Q fallbacks.
        """
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
        """
        Applies the agent's move and updates state.
        """
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
        """
        Final move selection policy (after danger/threat logic in agent_smart_move):

        PRIORITY:
        1) Q-best among moves that WIN THEIR SUBBOARD immediately.
        2) Q-best among moves inside ANY 'critical meta-winning board':
               a board (bi, bj) such that IF we win that board,
               the entire META GAME is won.
        3) Q-best among all moves.
        4) Random fallback (increments random_plays).
        """

        # ----------------------------------------------------
        # Step 0: Identify all CRITICAL META-WIN boards
        # ----------------------------------------------------
        critical_winning_boards = set()

        for bi in range(3):
            for bj in range(3):

                # only consider still-playable boards
                if (bi, bj) in self.empty_sub_places:

                    # simulate: what if WE win this board?
                    self.sub_boards[bi][bj] = self.agent_symbol

                    # if winning this board wins the META game
                    if self._wins_meta(self.sub_boards, bi, bj, self.agent_symbol):
                        critical_winning_boards.add((bi, bj))

                    # revert
                    self.sub_boards[bi][bj] = 0

        # ----------------------------------------------------
        # Step 1: Collect SUBBOARD-WINNING moves
        # ----------------------------------------------------
        winning_moves = []
        for bi, bj, r, c in moves:
            sub = self.full_board[bi][bj]
            if self._wins_sub(sub, self.agent_symbol, r, c):
                winning_moves.append((bi, bj, r, c))

        # 1️⃣ PRIORITY: Subboard-winning moves
        if winning_moves:
            best = self.q_best(winning_moves)
            if best is not None:
                return best
            # No Q-values among winning moves → fallback random among them
            self.random_plays += 1
            return random.choice(winning_moves)

        # ----------------------------------------------------
        # 2️⃣ PRIORITY: Moves in CRITICAL META-WIN BOARDS
        # ----------------------------------------------------
        critical_moves = [
            (bi, bj, r, c)
            for (bi, bj, r, c) in moves
            if (bi, bj) in critical_winning_boards
        ]

        if critical_moves:
            best = self.q_best(critical_moves)
            if best is not None:
                return best
            # No Q-values among critical moves → fallback random among them
            self.random_plays += 1
            return random.choice(critical_moves)

        # ----------------------------------------------------
        # 3️⃣ Q-best among all moves
        # ----------------------------------------------------
        best = self.q_best(moves)
        if best is not None:
            return best

        # ----------------------------------------------------
        # 4️⃣ Final fallback: completely random move (no Q-info)
        # ----------------------------------------------------
        self.random_plays += 1
        return random.choice(moves)

    # ------------------------------------------------------------
    # Threat detection for meta and subboards
    # ------------------------------------------------------------
    def q_best(self, move_list):
        """
        Among the given moves, pick the one whose resulting board
        has the highest Q-value. Returns None if no successor state
        is found in Q-table.
        """
        best = None
        best_score = -99999999.0

        for bi, bj, r, c in move_list:
            # Temporarily place move
            self.place_in_rep(bi, bj, r, c, self.agent_symbol)
            b_int = self.get_board_int()
            # Revert
            self.place_in_rep(bi, bj, r, c, 0)

            if b_int in self.q_table:
                val, _ = self.q_table[b_int]
                if val > best_score:
                    best = (bi, bj, r, c)
                    best_score = val

        return best

    def find_meta_block(self):
        """
        Find meta-board positions (subboards) where opponent is
        threatening to win on the meta board (i.e., 2 in a line).
        Returns a list of (bi, bj) subboard indices.
        """
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

        # main diagonal
        diag = [sb[i][i] for i in range(3)]
        if sum(diag) == 2 * opp:
            for i in range(3):
                if sb[i][i] == 0 and (i, i) in self.empty_sub_places:
                    threats.add((i, i))

        # anti-diagonal
        adiag = [sb[i][2 - i] for i in range(3)]
        if sum(adiag) == 2 * opp:
            for i in range(3):
                if sb[i][2 - i] == 0 and (i, 2 - i) in self.empty_sub_places:
                    threats.add((i, 2 - i))

        return list(threats)

    def find_immidiate_danger(self, bi, bj):
        """
        For a given subboard (bi, bj), find cells that block an immediate
        win (2 in a row) for the opponent. Returns list of (r, c).
        """
        sb = self.full_board[bi][bj]
        opp = self.player_symbol
        dangers = []

        # rows
        for i in range(3):
            if np.sum(sb[i]) == 2 * opp:
                for j in range(3):
                    if sb[i][j] == 0:
                        dangers.append((i, j))

        # columns
        for j in range(3):
            col = sb[:, j]
            if np.sum(col) == 2 * opp:
                for i in range(3):
                    if sb[i][j] == 0:
                        dangers.append((i, j))

        # main diagonal
        diag = [sb[i][i] for i in range(3)]
        if sum(diag) == 2 * opp:
            for i in range(3):
                if sb[i][i] == 0:
                    dangers.append((i, i))

        # anti-diagonal
        adiag = [sb[i][2 - i] for i in range(3)]
        if sum(adiag) == 2 * opp:
            for i in range(3):
                if sb[i][2 - i] == 0:
                    dangers.append((i, 2 - i))

        return dangers

    # ------------------------------------------------------------
    # Winning move finder (meta win)
    # ------------------------------------------------------------
    def find_winning_move(self):
        """
        Search all legal moves; return (bi, bj, r, c) that produce
        an immediate meta-board win if it exists, else None.
        """
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

    def _wins_sub(self, sb, player, r, c):
        """
        Check if placing 'player' at (r, c) in subboard 'sb'
        wins that subboard.
        """
        # row
        if all((sb[r][j] == player or (r, j) == (r, c)) for j in range(3)):
            return True
        # column
        if all((sb[i][c] == player or (i, c) == (r, c)) for i in range(3)):
            return True
        # main diagonal
        if r == c:
            if all((sb[i][i] == player or (i, i) == (r, c)) for i in range(3)):
                return True
        # anti-diagonal
        if r + c == 2:
            if all((sb[i][2 - i] == player or (i, 2 - i) == (r, c)) for i in range(3)):
                return True
        return False

    def _wins_meta(self, sb, bi, bj, player):
        """
        Check if subboard (bi, bj) being won by 'player' yields
        a meta-board win.
        """
        # row
        if all(sb[bi][j] == player for j in range(3)):
            return True
        # column
        if all(sb[i][bj] == player for i in range(3)):
            return True
        # main diagonal
        if bi == bj:
            if all(sb[i][i] == player for i in range(3)):
                return True
        # anti-diagonal
        if bi + bj == 2:
            if all(sb[i][2 - i] == player for i in range(3)):
                return True
        return False

    # ------------------------------------------------------------
    # Symmetry hashing
    # ------------------------------------------------------------
    def all_symmetries_fast(self, b):
        """
        Return 8 symmetric variants of a 9x9 board (b).
        """
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
        """
        Returns the minimal integer hash among all symmetries of 'board'.
        """
        best = None
        for sym in self.all_symmetries_fast(board):
            v = hashing.encode_board_to_int(sym.ravel())
            if best is None or v < best:
                best = v
        return best

    def get_board_int(self):
        """
        Get canonical integer representation of current board_rep.
        """
        return self.canonical_board_int(self.board_rep)

    def get_board_from_int(self, value):
        """
        Decode an integer back to a 9x9 numpy board.
        """
        board = hashing.decode_board_from_int(value)
        return np.array(board).reshape(9, 9)

    # ------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------
    def play_one_game(self, epsilon=1.0, training=None):
        """
        Plays one game:
        - Agent plays with epsilon-greedy if training=True,
          otherwise fully greedy (smart only).
        - Opponent plays random.
        - Rewards are based on final game outcome, discounted backwards
          through the visited states.

        Returns:
            np.array of visited states (in reversed order of play).
        """
        if training is None:
            training = self.training

        self.init_game()
        states = []

        while True:
            if self.check_true_win() != 0 or self.check_true_tie():
                break

            # Agent move

            if training:
                self.agent_train_move(epsilon)
            else:
                self.agent_smart_move()
                self.num_plays += 1
            states.append(self.get_board_int())

            if self.check_true_win() != 0 or self.check_true_tie():
                break

            # Opponent random move
            self.player_random_move()
            states.append(self.get_board_int())

        winner = self.check_true_win()
        score = winner  # 1, 0, or -1

        # Backward discount
        states.reverse()
        for s in states:
            self.board_score_list.append((s, score))
            score *= self.gamma

        # If single-process, update Q-table immediately
        if not self.multiprocess and training:
            self.update_q_table()

        return np.array(states)

    # final win/tie checks on meta board
    def check_true_win(self):
        return self.check_win(self.sub_boards)

    def check_true_tie(self):
        return len(self.empty_sub_places) == 0 and self.check_true_win() == 0


# =====================================================================
# WORKER: RUNS A CHUNK OF GAMES AND RETURNS LOCAL Q-AVERAGES + STATS
# =====================================================================
def run_games_chunk(args):
    """
    Worker entrypoint for multiprocessing.

    Args:
        num_games: how many games to simulate in this chunk
        seed: RNG seed for reproducibility

    Returns:
        local_q: dict {state_int: (avg_target_over_chunk, count_in_chunk)}
        agent_w: number of agent wins
        player_w: number of opponent wins
        ties: number of ties
        random_plays: number of fallback random moves (missing Q)
        num_plays: total agent moves
        epsilon_used: epsilon for this chunk
    """
    num_games, seed, epsilon = args

    random.seed(seed)
    np.random.seed(seed)

    game = UltimateTicTacToeGame(
        q_table=GLOBAL_QTABLE,
        training=True,
        multiprocess=True,
    )

    local_q = {}
    agent_w = player_w = ties = 0

    for _ in range(num_games):

        game.play_one_game(epsilon=epsilon, training=True)
        w = game.check_true_win()

        if w == 1:
            agent_w += 1
        elif w == -1:
            player_w += 1
        else:
            ties += 1

        # Aggregate targets for each state in this chunk (averaging)
        for b, target in game.board_score_list:
            if b in local_q:
                avg, c = local_q[b]
                new_avg = (avg * c + target) / (c + 1)
                local_q[b] = (new_avg, c + 1)
            else:
                local_q[b] = (target, 1)

        game.board_score_list.clear()

    return (
        local_q,
        agent_w,
        player_w,
        ties,
        game.random_plays,
        game.num_plays,
        epsilon,
    )


# =====================================================================
# TRAINING MANAGER
# =====================================================================
class Games:
    """
    Wrapper to manage many games (training or evaluation).
    """

    def __init__(self, num_games, processes=None, log_every=1000, chunk_size=50):
        self.num_games = num_games
        self.processes = processes or multiprocessing.cpu_count()
        self.log_every = log_every
        self.chunk_size = chunk_size

        self.agent_wins = 0
        self.player_wins = 0
        self.ties = 0

        # Main game loads q.pkl ONCE and keeps it as a normal dict
        self.game = UltimateTicTacToeGame(
            q_table=None,
            training=False,
            multiprocess=False
        )

    # ------------------------------------------------------------
    # MULTIPROCESS TRAINING (EPOCH-BASED)
    # ------------------------------------------------------------
    def train(self):
        """
        Train the Q-table using multiprocessing.

        We train in epochs:
        - Each epoch creates a Pool, forking from the UPDATED Q-table.
        - Workers see the latest Q-table at the start of each epoch.
        - Within an epoch, workers use a snapshot (that's fine).
        """
        print(f"Training {self.num_games} games with multiprocessing...")

        start = time.time()
        num_chunks = (self.num_games + self.chunk_size - 1) // self.chunk_size

        args_list = []
        base_seed = int(time.time())
        completed = 0

        for i in range(num_chunks):
            # Linear epsilon schedule across chunks: EPS_START -> EPS_END
            frac = i / (num_chunks - 1) if num_chunks > 1 else 1.0
            epsilon = EPS_START + frac * (EPS_END - EPS_START)
            args_list.append((self.chunk_size, base_seed + i, epsilon))

        num_random = 0
        num_plays = 0

        # Number of chunks to process per epoch
        chunks_per_epoch = self.processes

        for epoch_start in range(0, num_chunks, chunks_per_epoch):
            epoch_end = min(epoch_start + chunks_per_epoch, num_chunks)
            epoch_args = args_list[epoch_start:epoch_end]

            # Snapshot Q-table for this epoch (workers see this at fork)
            global GLOBAL_QTABLE
            GLOBAL_QTABLE = self.game.q_table

            with Pool(
                processes=self.processes,
                initializer=init_worker,
                initargs=(GLOBAL_QTABLE,)
            ) as p:
                for (
                    local_q,
                    a_w,
                    p_w,
                    t_w,
                    r_p,
                    n_p,
                    eps
                ) in p.imap_unordered(run_games_chunk, epoch_args):

                    games_in_chunk = a_w + p_w + t_w
                    completed += games_in_chunk

                    self.agent_wins += a_w
                    self.player_wins += p_w
                    self.ties += t_w

                    num_random += r_p
                    num_plays += n_p


                    # Merge Q-tables via weighted average of averages:
                    # combined = (old_v * old_count + avg_local * count_local)
                    #            / (old_count + count_local)
                    if hasattr(self.game.q_table, "batch_merge_local_q"):
                        self.game.q_table.batch_merge_local_q(local_q)
                    else:
                        # fallback dict merge
                        for b, (avg_local, count_local) in local_q.items():
                            if b in self.game.q_table:
                                old_v, old_count = self.game.q_table[b]
                                combined = (old_v * old_count + avg_local * count_local) / (old_count + count_local)
                                self.game.q_table[b] = (combined, old_count + count_local)
                            else:
                                self.game.q_table[b] = (avg_local, count_local)

                    # Logging: approximate every log_every games
                    if (
                        completed >= self.log_every
                        and completed % self.log_every < self.chunk_size
                    ):
                        elapsed = time.time() - start
                        speed = completed / elapsed if elapsed > 0 else 0.0
                        randomness_pct = (
                            100.0 * num_random / num_plays if num_plays > 0 else 0.0
                        )
                        print(f"{completed}/{self.num_games} games, {speed:.2f} games/sec")
                        print(f"Current randomness (fallback %): {randomness_pct:.2f}%")
                        print(f"Current Epsilon (last chunk): {eps:.4f}")

        total_t = time.time() - start
        speed = self.num_games / total_t if total_t > 0 else 0.0
        final_randomness = (
            100.0 * num_random / num_plays if num_plays > 0 else 0.0
        )
        print(f"\nFinished training in {total_t:.2f}s ({speed:.2f} games/sec)")
        print(f"Overall fallback randomness: {final_randomness:.2f}%")
        print(f"Total agent moves: {num_plays}, fallback random moves: {num_random}")

    # ------------------------------------------------------------
    # SINGLE-PROCESS RUN (evaluation)
    # ------------------------------------------------------------
    def run(self):
        """
        Run games without exploration (epsilon=0) for evaluation,
        using current Q-table.
        """
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
                speed = i / elapsed if elapsed > 0 else 0.0
                print(f"{i}/{self.num_games} games, {speed:.2f} games/sec")

        total_t = time.time() - start
        speed = self.num_games / total_t if total_t > 0 else 0.0
        print(f"Done evaluation in {total_t:.2f}s ({speed:.2f} games/sec)")


# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    cores = multiprocessing.cpu_count()

    # 1) TRAIN
    games = Games(
        num_games=500_000,     # increase this for stronger agent
        processes=cores,
        log_every=10_000,
        chunk_size=100
    )

    games.train()

    print("TRAINING DONE")
    print("Agent win %:", 100 * games.agent_wins / games.num_games)
    print("Player win %:", 100 * games.player_wins / games.num_games)
    print("Tie %:", 100 * games.ties / games.num_games)

    # Save resulting Q-table (already a normal dict)
    # hashing.save_qtable("q.pkl", games.game.q_table)

    # 2) (OPTIONAL) EVALUATION RUN AFTER TRAINING
    print("Linux training done")
    eval_games = Games(num_games=1_000, processes=None, log_every=1_000)
    eval_games.run()

    print("EVAL Agent win %:", 100 * eval_games.agent_wins / eval_games.num_games)
    print("EVAL Player win %:", 100 * eval_games.player_wins / eval_games.num_games)
    print("EVAL Tie %:", 100 * eval_games.ties / eval_games.num_games)
    print(
        "Random % (fallback during eval):",
        100 * eval_games.game.random_plays / eval_games.game.num_plays
        if eval_games.game.num_plays > 0 else 0.0
    )
