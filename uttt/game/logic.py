import time
import random
import multiprocessing
import sys
import numpy as np
from multiprocessing import Pool
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from uttt.game import hashing, lmdb_qtable, shrinker
from uttt.game.lmdb_qtable import LMDBQTable
EPS_START = 1.0      # (full exploration)
EPS_END = 0.2       # (mostly exploitation)

#as training grows with exploit more linearly

GLOBAL_QTABLE = None


def init_worker(qtable):
    global GLOBAL_QTABLE
    GLOBAL_QTABLE = qtable

    lmdb_qtable.GLOBAL_TXN = qtable.env.begin(write=False)





# The game logic
class UltimateTicTacToeGame:

    #Training tells if we epsilon-greedy or run regularly, multiprocess if we update the table inside or outside the class
    def __init__(self, q_table=None, training=False, multiprocess=False, randomPlayer=True):
        # Board representations
        self.board_rep = np.zeros((9, 9), dtype=int)         # global 9x9
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)  # 3x3 subboards of 3x3
        self.sub_boards = np.zeros((3, 3), dtype=int)        # meta board

        # Symbols
        self.player_symbol = -1
        self.agent_symbol = 1
        self.randomPlayer = randomPlayer
        # Counters
        self.num_plays = 0       # total agent moves
        self.random_plays = 0    # times we FALL BACK to random (no Q info / forced random)

        # State tracking
        self.empty_places = self.get_empty_places()
        self.empty_sub_places = self.get_empty_sub_places()
        self.curr_board = None


        if q_table is None:
            self.q_table = LMDBQTable()
        else:
            self.q_table = q_table
        if lmdb_qtable.GLOBAL_TXN is None and isinstance(self.q_table, LMDBQTable):
            lmdb_qtable.GLOBAL_TXN = self.q_table.env.begin(write=False)

        self.training = training
        self.multiprocess = multiprocess

        # discount factor
        self.gamma = 0.97

        # (state_int, target_value) accumulated during game
        self.board_score_list = []

        # For printing
        self.players = {0: "-", 1: "X", -1: "O"}

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


    def global_board(self):
        return self.board_rep

    def init_game(self):

        #reseting the game
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

    def get_available_moves(self):
        return self.legal_moves()

    def print_board(self, board):

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
        return (bi, bj) not in self.empty_sub_places

    def legal_moves(self, playable_boards=None):
        boards = self.get_playable_boards() if playable_boards is None else playable_boards
        return [
            (bi, bj, r, c)
            for (bi, bj) in boards
            for (r, c) in self.empty_places[bi][bj]
        ]

    def cell_wins(self, board, cell, symbol):
        r, c = cell
        if board[r][c] != 0:
            return False
        board[r][c] = symbol
        won = self.check_win(board) == symbol
        board[r][c] = 0
        return won

    def winning_cells(self, board, empty_cells, symbol):
        return [cell for cell in empty_cells if self.cell_wins(board, cell, symbol)]

    def move_wins_sub(self, move, symbol):
        bi, bj, r, c = move
        return self.cell_wins(self.full_board[bi][bj], (r, c), symbol)

    def move_wins_meta(self, move, symbol):
        bi, bj, _, _ = move
        return self.move_wins_sub(move, symbol) and self.cell_wins(
            self.sub_boards, (bi, bj), symbol
        )

    def find_meta_winning_boards(self, symbol):
        return self.winning_cells(self.sub_boards, self.empty_sub_places, symbol)

    def find_winning_moves(self, symbol, legal_moves=None, target="meta"):
        if legal_moves is None:
            legal_moves = self.get_available_moves()
        if target == "meta":
            return [move for move in legal_moves if self.move_wins_meta(move, symbol)]
        if target == "sub":
            return [move for move in legal_moves if self.move_wins_sub(move, symbol)]
        raise ValueError(f"Unsupported target: {target}")

    def apply_move(self, bi, bj, r, c, symbol):
        self.full_board[bi][bj][r][c] = symbol
        self.place_in_rep(bi, bj, r, c, symbol)
        self.empty_places[bi][bj].remove((r, c))

        winner = self.check_win(self.full_board[bi][bj])
        if winner != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = winner
            self.empty_sub_places.discard((bi, bj))

        next_board = (r, c)
        self.curr_board = None if self.sub_board_is_done(*next_board) else next_board

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
        bi, bj, r, c = random.choice(self.get_available_moves())
        self.apply_player_move(bi, bj, r, c)


    def player_smart_move(self):
        """
        Smart opponent move logic.

        PRIORITY:
        1) Immediate META win
        2) Block agent META win
        3) Immediate LOCAL subboard win
        4) Random fallback
        """
        all_moves = self.get_available_moves()

        meta_wins = self.find_winning_moves(self.player_symbol, all_moves, target="meta")
        if meta_wins:
            self.apply_player_move(*random.choice(meta_wins))
            return

        threatened = self.find_meta_block(self.agent_symbol)
        if threatened:
            immediate = {}
            for bi, bj in threatened:
                danger_cells = self.find_immediate_danger(bi, bj, self.agent_symbol)
                if danger_cells:
                    immediate[(bi, bj)] = danger_cells

            if immediate:
                if self.curr_board is None:
                    blocking_moves = [
                        (bi, bj, r, c)
                        for (bi, bj), cells in immediate.items()
                        for (r, c) in cells
                    ]
                    safe_blocks = []
                    for move in blocking_moves:
                        bi, bj, r, c = move
                        closes_and_sends_to_self = (r, c) == (bi, bj) and (
                            self.move_wins_sub(move, self.player_symbol)
                            or len(self.empty_places[bi][bj]) == 1
                        )
                        sends_to_done = (r, c) not in self.empty_sub_places or closes_and_sends_to_self
                        if len(immediate) > 1 and (sends_to_done or ((r, c) in immediate and (r, c) != (bi, bj))):
                            continue
                        safe_blocks.append(move)

                    moves = safe_blocks if safe_blocks else blocking_moves
                    self.apply_player_move(*random.choice(moves))
                    return

                if self.curr_board in immediate:
                    bi, bj = self.curr_board
                    blocking_moves = [(bi, bj, r, c) for (r, c) in immediate[(bi, bj)]]
                    other_threats = set(immediate) - {self.curr_board}
                    safe_blocks = []
                    for move in blocking_moves:
                        _, _, r, c = move
                        closes_and_sends_to_self = (r, c) == (bi, bj) and (
                            self.move_wins_sub(move, self.player_symbol)
                            or len(self.empty_places[bi][bj]) == 1
                        )
                        sends_to_done = (r, c) not in self.empty_sub_places or closes_and_sends_to_self
                        if other_threats and (sends_to_done or (r, c) in other_threats):
                            continue
                        safe_blocks.append(move)

                    moves = safe_blocks if safe_blocks else blocking_moves
                    self.apply_player_move(*random.choice(moves))
                    return

                bi, bj = self.curr_board
                safe_moves = []
                forbidden = set(immediate)
                for (r, c) in self.empty_places[bi][bj]:
                    move = (bi, bj, r, c)
                    closes_and_sends_to_self = (r, c) == (bi, bj) and (
                        self.move_wins_sub(move, self.player_symbol)
                        or len(self.empty_places[bi][bj]) == 1
                    )
                    sends_to_done = (r, c) not in self.empty_sub_places or closes_and_sends_to_self
                    if (r, c) not in forbidden and not sends_to_done:
                        safe_moves.append(move)

                if safe_moves:
                    self.apply_player_move(*random.choice(safe_moves))
                    return

                r, c = random.choice(tuple(self.empty_places[bi][bj]))
                self.apply_player_move(bi, bj, r, c)
                return

        local_wins = self.find_winning_moves(self.player_symbol, all_moves, target="sub")
        if local_wins:
            self.apply_player_move(*random.choice(local_wins))
            return

        self.apply_player_move(*random.choice(all_moves))

    def apply_player_move(self, bi, bj, r, c):
        self.apply_move(bi, bj, r, c, self.player_symbol)

    # ------------------------------------------------------------
    # Agent move selection
    # ------------------------------------------------------------
    def agent_train_move(self, epsilon=1.0):

        if random.random() < epsilon:
            # random move, not counted as missing-Q fallback
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
        all_moves = self.get_available_moves()
        meta_wins = self.find_winning_moves(self.agent_symbol, all_moves, target="meta")
        if meta_wins:
            best, used = self.select_best_move(meta_wins)
            self.apply_agent_move(*best)
            return used

        threatened = self.find_meta_block(self.player_symbol)
        if threatened:
            immediate = {}
            for bi, bj in threatened:
                danger_cells = self.find_immediate_danger(bi, bj, self.player_symbol)
                if danger_cells:
                    immediate[(bi, bj)] = danger_cells

            if immediate:
                if self.curr_board is None:
                    blocking_moves = [
                        (bi, bj, r, c)
                        for (bi, bj), cells in immediate.items()
                        for (r, c) in cells
                    ]
                    safe_blocks = []
                    for move in blocking_moves:
                        bi, bj, r, c = move
                        closes_and_sends_to_self = (r, c) == (bi, bj) and (
                            self.move_wins_sub(move, self.agent_symbol)
                            or len(self.empty_places[bi][bj]) == 1
                        )
                        sends_to_done = (r, c) not in self.empty_sub_places or closes_and_sends_to_self
                        if len(immediate) > 1 and (sends_to_done or ((r, c) in immediate and (r, c) != (bi, bj))):
                            continue
                        safe_blocks.append(move)

                    best, used = self.select_best_move(safe_blocks if safe_blocks else blocking_moves)
                    self.apply_agent_move(*best)
                    return used

                if self.curr_board in immediate:
                    bi, bj = self.curr_board
                    blocking_moves = [(bi, bj, r, c) for (r, c) in immediate[(bi, bj)]]
                    other_threats = set(immediate) - {self.curr_board}
                    safe_blocks = []
                    for move in blocking_moves:
                        _, _, r, c = move
                        closes_and_sends_to_self = (r, c) == (bi, bj) and (
                            self.move_wins_sub(move, self.agent_symbol)
                            or len(self.empty_places[bi][bj]) == 1
                        )
                        sends_to_done = (r, c) not in self.empty_sub_places or closes_and_sends_to_self
                        if other_threats and (sends_to_done or (r, c) in other_threats):
                            continue
                        safe_blocks.append(move)

                    best, used = self.select_best_move(safe_blocks if safe_blocks else blocking_moves)
                    self.apply_agent_move(*best)
                    return used

                bi, bj = self.curr_board
                safe_moves = []
                forbidden = set(immediate)
                for (r, c) in self.empty_places[bi][bj]:
                    move = (bi, bj, r, c)
                    closes_and_sends_to_self = (r, c) == (bi, bj) and (
                        self.move_wins_sub(move, self.agent_symbol)
                        or len(self.empty_places[bi][bj]) == 1
                    )
                    sends_to_done = (r, c) not in self.empty_sub_places or closes_and_sends_to_self
                    if (r, c) not in forbidden and not sends_to_done:
                        safe_moves.append(move)

                if safe_moves:
                    best, used = self.select_best_move(safe_moves)
                    self.apply_agent_move(*best)
                    return used

                r, c = random.choice(tuple(self.empty_places[bi][bj]))
                self.random_plays += 1
                self.apply_agent_move(bi, bj, r, c)
                return False

        local_wins = self.find_winning_moves(self.agent_symbol, all_moves, target="sub")
        if local_wins:
            best, used = self.select_best_move(local_wins)
            self.apply_agent_move(*best)
            return used

        best, used = self.select_best_move(all_moves)
        self.apply_agent_move(*best)
        return used

    def agent_random_move(self):
        """
        Agent plays a random legal move.
        Used ONLY for epsilon exploration; does NOT increment random_plays,
        because random_plays is reserved for missing-Q fallbacks.
        """
        bi, bj, r, c = random.choice(self.get_available_moves())
        self.apply_agent_move(bi, bj, r, c)

    # ------------------------------------------------------------
    # Apply move
    # ------------------------------------------------------------
    def apply_agent_move(self, bi, bj, r, c):
        """
        Applies the agent's move and updates state.
        """
        self.apply_move(bi, bj, r, c, self.agent_symbol)

    # ------------------------------------------------------------
    # Q-score selection helper
    # ------------------------------------------------------------
    def select_best_move(self, moves):
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

        critical_winning_boards = set(self.find_meta_winning_boards(self.agent_symbol))
        winning_moves = self.find_winning_moves(self.agent_symbol, moves, target="sub")

        # 1ï¸âƒ£ PRIORITY: Subboard-winning moves
        if winning_moves:
            best = self.best_from_moves(winning_moves)
            if best is not None:
                return best, True
            self.random_plays += 1
            return random.choice(winning_moves), False


        critical_moves = [move for move in moves if (move[0], move[1]) in critical_winning_boards]

        if critical_moves:
            best = self.best_from_moves(critical_moves)
            if best is not None:
                return best, True
            # No Q-values among critical moves â†’ fallback random among them
            self.random_plays += 1
            return random.choice(critical_moves), False


        best = self.best_from_moves(moves)
        if best is not None:
            return best, True


        self.random_plays += 1
        return random.choice(moves), False

    # ------------------------------------------------------------
    # Threat detection for meta and subboards
    # ------------------------------------------------------------
    def best_from_moves(self, move_list):
        """
        Among the given moves, pick the one whose resulting board
        has the highest Q-value/CNN-value. Returns None if no successor state
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

    def find_meta_block(self, opp = None):
        """
        Find meta-board positions (subboards) where opponent is
        threatening to win on the meta board (i.e., 2 in a line).
        Returns a list of (bi, bj) subboard indices.
        """
        if opp is None:
            opp = self.player_symbol
        return self.find_meta_winning_boards(opp)

    def find_immediate_danger(self, bi, bj, opp = None):
        """
        For a given subboard (bi, bj), find cells that block an immediate
        win (2 in a row) for the opponent. Returns list of (r, c).
        """
        if opp is None:
            opp = self.player_symbol
        return self.winning_cells(self.full_board[bi][bj], self.empty_places[bi][bj], opp)

    # ------------------------------------------------------------
    # Winning move finder (meta win)
    # ------------------------------------------------------------
    def find_winning_move(self, symbol=None):
        """
        Search all legal moves; return (bi, bj, r, c) that produce
        an immediate meta-board win if it exists, else None.
        """
        if symbol is None:
            symbol = self.agent_symbol
        winning_moves = self.find_winning_moves(symbol, target="meta")
        return winning_moves[0] if winning_moves else None

    # ------------------------------------------------------------
    # Symmetry hashing
    # ------------------------------------------------------------
    # def all_symmetries_fast(self, b):
    #     """
    #     Return 8 symmetric variants of a 9x9 board (b).
    #     """
    #     bt = b.T
    #     fh = np.flipud(b)
    #     fv = np.fliplr(b)
    #     fd = np.fliplr(bt)
    #     fad = np.flipud(bt)
    #     r90 = fad
    #     r180 = np.flipud(fv)
    #     r270 = fd
    #     return [b, r90, r180, r270, fh, fv, fd, fad]

    def all_symmetries_fast(self, b):
        """
        Return the 8 TRUE UTTT symmetries of a 9x9 board.

        Correctly applies D4 on UTTT structure:
        - rearrange the 3x3 grid of subboards (treat each subboard as an atom)
        - apply the same transform inside each 3x3 subboard
        """


        # ---- transforms that work for BOTH 2D (3x3) and 3D (3x3x9) ----
        def I(x):
            return x

        def R90(x):
            return np.rot90(x, 1, axes=(0, 1))

        def R180(x):
            return np.rot90(x, 2, axes=(0, 1))

        def R270(x):
            return np.rot90(x, 3, axes=(0, 1))

        def FLIP_LR(x):
            return np.flip(x, axis=1)  # mirror left<->right on first two axes

        def FLIP_UD(x):
            return np.flip(x, axis=0)  # mirror up<->down on first two axes

        def DIAG(x):
            return np.swapaxes(x, 0, 1)  # reflect main diagonal

        def ADIAG(x):
            return np.swapaxes(R180(x), 0, 1)  # reflect anti-diagonal

        transforms = [I, R90, R180, R270, FLIP_LR, FLIP_UD, DIAG, ADIAG]

        # reshape 9x9 -> (3,3,3,3)
        blk = b.reshape(3, 3, 3, 3)

        out = []
        for f in transforms:
            # 1) rearrange BIG grid of subboards (treat each subboard as a length-9 atom)
            big = blk.reshape(3, 3, 9)  # (big_r,big_c,flat_small)
            big2 = f(big)  # transform only first two axes
            blk2 = big2.reshape(3, 3, 3, 3)

            # 2) transform INSIDE each small 3x3 board
            res = np.empty_like(blk2)
            for i in range(3):
                for j in range(3):
                    res[i, j] = f(blk2[i, j])

            out.append(res.reshape(9, 9))

        return out

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
    def play_one_game(self, epsilon=1.0, training=None, randomPlayer=None):
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

        if randomPlayer is None:
            randomPlayer = self.randomPlayer




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
            if randomPlayer:
                self.player_random_move()
            else:
                self.player_smart_move()
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
    num_games, seed, epsilon, randomPlayer = args

    random.seed(seed)
    np.random.seed(seed)

    game = UltimateTicTacToeGame(
        q_table=GLOBAL_QTABLE,
        training=True,
        multiprocess=True,
        randomPlayer=randomPlayer
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



class Games:


    def __init__(self, num_games, processes=None, log_every=1000, chunk_size=50, randomPlayer=True):
        self.num_games = num_games
        self.processes = processes or multiprocessing.cpu_count()
        self.log_every = log_every
        self.chunk_size = chunk_size

        self.agent_wins = 0
        self.player_wins = 0
        self.ties = 0
        self.randomPlayer = randomPlayer

        # Main game loads ONCE and keeps it as a normal dict
        self.game = UltimateTicTacToeGame(
            q_table=None,
            training=False,
            multiprocess=False,
            randomPlayer=randomPlayer
        )

    # ------------------------------------------------------------
    # MULTIPROCESS TRAINING (EPOCH-BASED)
    # ------------------------------------------------------------
    def multi_process_train(self):
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
            args_list.append((self.chunk_size, base_seed + i, epsilon, self.randomPlayer))

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
    def single_process_train(self, training=False, epsilon=0.0):
        """
        Run games without exploration (epsilon=0) for evaluation/training,
        using current Q-table.
        """
        print(f"Running {self.num_games} games single-process (evaluation)...")
        start = time.time()

        for i in range(1, self.num_games + 1):
            # evaluation: no exploration


            self.game.play_one_game(epsilon=epsilon, training=training, randomPlayer=self.randomPlayer)

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





if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    start_time = time.time()
    # 1) TRAIN

    random_player = random.choice([False, True, True])

    print(f"Training against random: {random_player}")

    games = Games(
    num_games=350_000,     # increase this for stronger agent
    processes=cores,
    log_every=10_000,
    chunk_size=100,
    randomPlayer=random_player
    )

    games.multi_process_train()

    print("TRAINING DONE")
    print("Agent win %:", 100 * games.agent_wins / games.num_games)
    print("Player win %:", 100 * games.player_wins / games.num_games)
    print("Tie %:", 100 * games.ties / games.num_games)

    #Save resulting Q-table (already a normal dict)
    #hashing.save_qtable("q.pkl", games.game.q_table)

    # 2) (OPTIONAL) EVALUATION RUN AFTER TRAINING
    print("Linux training done")
    shrinker.refresh()



    eval_games = Games(num_games=2_000, processes=None, log_every=10, randomPlayer=True)
    eval_games.single_process_train(training=True, epsilon=0.0)

    print("EVAL agent win %:", 100 * eval_games.agent_wins / eval_games.num_games)
    print("EVAL player win %:", 100 * eval_games.player_wins / eval_games.num_games)
    print("EVAL tie %:", 100 * eval_games.ties / eval_games.num_games)
    print(
        "Random % (fallback during eval):",
        100 * eval_games.game.random_plays / eval_games.game.num_plays
        if eval_games.game.num_plays > 0 else 0.0
    )
    end_time = time.time()
    delta_time = end_time - start_time
    hours = delta_time // 3600
    mins = (delta_time - 3600 * hours) // 60
    print(f"Overall run time is {hours}:{mins}h")

