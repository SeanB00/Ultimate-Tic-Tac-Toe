import time
import random
import multiprocessing
import sys
import numpy as np
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from uttt.game import hashing, lmdb_qtable, shrinker
from uttt.game.lmdb_qtable import LMDBQTable


# game logic
class UltimateTicTacToeGame:

    PLAYERS = {0: "-", 1: "X", -1: "O"}

    def __init__(self, q_table=None, training=False, multiprocess=False, random_player=True):
        """set up one game state."""
        self.board_rep = np.zeros((9, 9), dtype=int)
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)

        self.player_symbol = -1
        self.agent_symbol = 1
        self.random_player = random_player
        self.num_plays = 0
        self.random_plays = 0

        self.empty_places = self.new_empty_places()
        self.empty_sub_places = self.new_empty_sub_places()
        self.curr_board = None

        if q_table is None:
            self.q_table = LMDBQTable()
        else:
            self.q_table = q_table
        if lmdb_qtable.GLOBAL_TXN is None and isinstance(self.q_table, LMDBQTable):
            lmdb_qtable.GLOBAL_TXN = self.q_table.begin_read()

        self.training = training
        self.multiprocess = multiprocess

        self.gamma = 0.97
        self.board_score_list = []

    @staticmethod
    def new_empty_places():
        """build the empty-cell sets for all subboards."""
        return [
            [{(i, j) for i in range(3) for j in range(3)} for _ in range(3)]
            for _ in range(3)
        ]

    @staticmethod
    def new_empty_sub_places():
        """build the set of playable subboards."""
        return {(i, j) for i in range(3) for j in range(3)}

    def init_game(self):
        """reset the board state."""
        self.full_board = np.zeros((3, 3, 3, 3), dtype=int)
        self.sub_boards = np.zeros((3, 3), dtype=int)
        self.empty_places = self.new_empty_places()
        self.empty_sub_places = self.new_empty_sub_places()
        self.curr_board = None
        self.board_rep = np.zeros((9, 9), dtype=int)
        self.board_score_list = []

    def is_game_running(self):
        """return whether the game is still active."""
        return self.check_true_win() == 0 and not self.check_true_tie()

    def get_playable_boards(self):
        """return the currently playable subboards."""
        if self.curr_board is None:
            return set(self.empty_sub_places)
        if self.curr_board in self.empty_sub_places:
            return {self.curr_board}
        return set(self.empty_sub_places)

    @staticmethod
    def print_board(board):
        """print a 9x9 board."""

        s = ""
        for row in range(9):
            s += "|"
            for col in range(9):
                s += UltimateTicTacToeGame.PLAYERS[board[row][col]] + "|"
                if col % 3 == 2 and col != 8:
                    s += "  |"
            s += "\n"
            if row % 3 == 2:
                s += "\n"
        print(s)
        print("-" * 9)


    @staticmethod
    def check_win(board):
        """check a 3x3 board for a winner."""
        for row in board:
            s = np.sum(row)
            if s == 3:
                return 1
            if s == -3:
                return -1

        for c in range(3):
            s = np.sum(board[:, c])
            if s == 3:
                return 1
            if s == -3:
                return -1

        d1 = np.trace(board)
        if d1 == 3:
            return 1
        if d1 == -3:
            return -1

        d2 = np.trace(np.fliplr(board))
        if d2 == 3:
            return 1
            # noqa
        if d2 == -3:
            return -1

        return 0

    @staticmethod
    def tie(board, empty):
        """check whether a 3x3 board is tied."""
        return len(empty) == 0 and UltimateTicTacToeGame.check_win(board) == 0

    def sub_board_is_done(self, bi, bj):
        """check whether a subboard is finished."""
        return (bi, bj) not in self.empty_sub_places

    def legal_moves(self, playable_boards=None):
        """return legal moves for the current state."""
        boards = self.get_playable_boards() if playable_boards is None else playable_boards
        return [
            (bi, bj, r, c)
            for (bi, bj) in boards
            for (r, c) in self.empty_places[bi][bj]
        ]

    @staticmethod
    def cell_wins(board, cell, symbol):
        """check whether one cell move wins a 3x3 board."""
        r, c = cell
        if board[r][c] != 0:
            return False
        board[r][c] = symbol
        won = UltimateTicTacToeGame.check_win(board) == symbol
        board[r][c] = 0
        return won

    @staticmethod
    def winning_cells(board, empty_cells, symbol):
        """return empty cells that win a 3x3 board."""
        return [
            cell for cell in empty_cells
            if UltimateTicTacToeGame.cell_wins(board, cell, symbol)
        ]

    def move_wins_sub(self, move, symbol):
        """check whether a move wins its local board."""
        bi, bj, r, c = move
        return self.cell_wins(self.full_board[bi][bj], (r, c), symbol)

    def move_wins_meta(self, move, symbol):
        """check whether a move wins the meta board."""
        bi, bj, _, _ = move
        return self.move_wins_sub(move, symbol) and self.cell_wins(
            self.sub_boards, (bi, bj), symbol
        )

    def find_meta_winning_boards(self, symbol):
        """return meta-board cells that would win immediately."""
        return self.winning_cells(self.sub_boards, self.empty_sub_places, symbol)

    def find_sub_winning_moves(self, symbol, legal_moves=None):
        """return legal moves that win a subboard."""
        if legal_moves is None:
            legal_moves = self.legal_moves()
        return [move for move in legal_moves if self.move_wins_sub(move, symbol)]

    def find_meta_winning_moves(self, symbol, legal_moves=None):
        """return legal moves that win the meta board."""
        if legal_moves is None:
            legal_moves = self.legal_moves()
        return [move for move in legal_moves if self.move_wins_meta(move, symbol)]

    def apply_move(self, bi, bj, r, c, symbol):
        """apply one move and update cached state."""
        self.full_board[bi][bj][r][c] = symbol
        self.place_in_rep(bi, bj, r, c, symbol)
        self.empty_places[bi][bj].remove((r, c))

        winner = self.check_win(self.full_board[bi][bj])
        if winner != 0 or self.tie(self.full_board[bi][bj], self.empty_places[bi][bj]):
            self.sub_boards[bi][bj] = winner
            self.empty_sub_places.discard((bi, bj))

        next_board = (r, c)
        self.curr_board = None if self.sub_board_is_done(*next_board) else next_board

    # board mapping
    @staticmethod
    def to_global_position(bi, bj, r, c):
        """map local coordinates to the 9x9 board."""
        return bi * 3 + r, bj * 3 + c

    def place_in_rep(self, bi, bj, r, c, symbol):
        """write one symbol into board_rep."""
        gi, gj = self.to_global_position(bi, bj, r, c)
        self.board_rep[gi][gj] = symbol

    # q-table updates
    def update_q_table(self):
        """merge the collected game targets into the q-table."""
        if hasattr(self.q_table, "update_from_targets"):
            self.q_table.update_from_targets(self.board_score_list)
        else:
            for board_int, target in self.board_score_list:
                if board_int in self.q_table:
                    old_v, count = self.q_table[board_int]
                else:
                    old_v, count = 0.0, 0

                new_v = (old_v * count + target) / (count + 1)
                self.q_table[board_int] = (new_v, count + 1)

    def player_random_move(self):
        """play a random move for the player."""
        bi, bj, r, c = random.choice(self.legal_moves())
        self.apply_move(bi, bj, r, c, self.player_symbol)

    def immediate_threats(self, opponent_symbol):
        """map threatened boards to the cells that block them."""
        immediate = {}
        for bi, bj in self.find_meta_winning_boards(opponent_symbol):
            danger_cells = self.winning_cells(
                self.full_board[bi][bj],
                self.empty_places[bi][bj],
                opponent_symbol,
            )
            if danger_cells:
                immediate[(bi, bj)] = danger_cells
        return immediate

    def move_sends_to_done_board(self, move, symbol):
        """check whether a move sends play to a finished board."""
        bi, bj, r, c = move
        closes_and_sends_to_self = (r, c) == (bi, bj) and (
            self.move_wins_sub(move, symbol) or len(self.empty_places[bi][bj]) == 1
        )
        return (r, c) not in self.empty_sub_places or closes_and_sends_to_self

    def find_smart_move_candidates(self, symbol, opponent_symbol, legal_moves=None):
        """return the highest-priority move candidates for a symbol."""
        if legal_moves is None:
            legal_moves = self.legal_moves()

        meta_wins = self.find_meta_winning_moves(symbol, legal_moves)
        if meta_wins:
            return meta_wins

        immediate = self.immediate_threats(opponent_symbol)
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
                    sends_to_done = self.move_sends_to_done_board(move, symbol)
                    if len(immediate) > 1 and (sends_to_done or ((r, c) in immediate and (r, c) != (bi, bj))):
                        continue
                    safe_blocks.append(move)

                return safe_blocks if safe_blocks else blocking_moves

            if self.curr_board in immediate:
                bi, bj = self.curr_board
                blocking_moves = [(bi, bj, r, c) for (r, c) in immediate[(bi, bj)]]
                other_threats = set(immediate) - {self.curr_board}
                safe_blocks = []
                for move in blocking_moves:
                    _, _, r, c = move
                    sends_to_done = self.move_sends_to_done_board(move, symbol)
                    if other_threats and (sends_to_done or (r, c) in other_threats):
                        continue
                    safe_blocks.append(move)

                return safe_blocks if safe_blocks else blocking_moves

            bi, bj = self.curr_board
            safe_moves = []
            forbidden = set(immediate)
            for (r, c) in self.empty_places[bi][bj]:
                move = (bi, bj, r, c)
                sends_to_done = self.move_sends_to_done_board(move, symbol)
                if (r, c) not in forbidden and not sends_to_done:
                    safe_moves.append(move)

            if safe_moves:
                return safe_moves

            return [(bi, bj, r, c) for (r, c) in self.empty_places[bi][bj]]

        critical_boards = set(self.find_meta_winning_boards(symbol))
        if critical_boards:
            critical_moves = [move for move in legal_moves if (move[0], move[1]) in critical_boards]
            if critical_moves:
                critical_wins = self.find_sub_winning_moves(symbol, critical_moves)
                return critical_wins if critical_wins else critical_moves

        local_wins = self.find_sub_winning_moves(symbol, legal_moves)
        if local_wins:
            return local_wins

        return legal_moves

    def choose_random_move(self, moves):
        """pick a random move from the candidate list."""
        return random.choice(moves), False

    def choose_agent_move(self, moves):
        """pick the best scored move or fall back to random."""
        best = self.best_from_moves(moves)
        if best is not None:
            return best, True

        self.random_plays += 1
        return random.choice(moves), False

    def smart_move(self, symbol, opponent_symbol, chooser):
        """build smart candidates, choose one move, and apply it."""
        candidates = self.find_smart_move_candidates(symbol, opponent_symbol)
        move, used = chooser(candidates)
        self.apply_move(*move, symbol)
        return used

    def player_smart_move(self):
        """play a heuristic move for the player."""
        self.smart_move(
            self.player_symbol,
            self.agent_symbol,
            self.choose_random_move,
        )

    def agent_train_move(self, epsilon=1.0):
        """play one training move for the agent."""

        if random.random() < epsilon:
            self.agent_random_move()
        else:
            self.agent_smart_move()
            self.num_plays += 1

    def agent_smart_move(self):
        """play a heuristic q-table move for the agent."""
        return self.smart_move(
            self.agent_symbol,
            self.player_symbol,
            self.choose_agent_move,
        )

    def agent_random_move(self):
        """play a random move for the agent."""
        bi, bj, r, c = random.choice(self.legal_moves())
        self.apply_move(bi, bj, r, c, self.agent_symbol)

    def best_from_moves(self, move_list):
        """pick the highest-scoring move from a list."""
        best = None
        best_score = -99999999.0

        for bi, bj, r, c in move_list:
            self.place_in_rep(bi, bj, r, c, self.agent_symbol)
            b_int = self.canonical_board_int(self.board_rep)
            self.place_in_rep(bi, bj, r, c, 0)

            if b_int in self.q_table:
                val, _ = self.q_table[b_int]
                if val > best_score:
                    best = (bi, bj, r, c)
                    best_score = val

        return best

    @staticmethod
    def all_symmetries_fast(b):
        """return the 8 uttt symmetries of a board."""

        # symmetry transforms
        def I(x):
            return x

        def R90(x):
            return np.rot90(x, 1, axes=(0, 1))

        def R180(x):
            return np.rot90(x, 2, axes=(0, 1))

        def R270(x):
            return np.rot90(x, 3, axes=(0, 1))

        def FLIP_LR(x):
            return np.flip(x, axis=1)

        def FLIP_UD(x):
            return np.flip(x, axis=0)

        def DIAG(x):
            return np.swapaxes(x, 0, 1)

        def ADIAG(x):
            return np.swapaxes(R180(x), 0, 1)

        transforms = [I, R90, R180, R270, FLIP_LR, FLIP_UD, DIAG, ADIAG]

        blk = b.reshape(3, 3, 3, 3)

        out = []
        for f in transforms:
            big = blk.reshape(3, 3, 9)
            big2 = f(big)
            blk2 = big2.reshape(3, 3, 3, 3)

            res = np.empty_like(blk2)
            for i in range(3):
                for j in range(3):
                    res[i, j] = f(blk2[i, j])

            out.append(res.reshape(9, 9))

        return out

    @staticmethod
    def canonical_board_int(board):
        """return the smallest hash across all symmetries."""
        best = None
        for sym in UltimateTicTacToeGame.all_symmetries_fast(board):
            v = hashing.encode_board_to_int(sym.ravel())
            if best is None or v < best:
                best = v
        return best

    @staticmethod
    def get_board_from_int(value):
        """decode a hash back into a 9x9 board."""
        board = hashing.decode_board_from_int(value)
        return np.array(board).reshape(9, 9)

    def play_one_game(self, epsilon=1.0, training=None, random_player=None):
        """play one full game and collect state targets."""
        if training is None:
            training = self.training

        if random_player is None:
            random_player = self.random_player
        self.init_game()
        states = []

        while True:
            if self.check_true_win() != 0 or self.check_true_tie():
                break

            if training:
                self.agent_train_move(epsilon)
            else:
                self.agent_smart_move()
                self.num_plays += 1
            states.append(self.canonical_board_int(self.board_rep))

            if self.check_true_win() != 0 or self.check_true_tie():
                break

            if random_player:
                self.player_random_move()
            else:
                self.player_smart_move()
            states.append(self.canonical_board_int(self.board_rep))

        winner = self.check_true_win()
        score = winner

        states.reverse()
        for s in states:
            self.board_score_list.append((s, score))
            score *= self.gamma

        if not self.multiprocess and training:
            self.update_q_table()

        return np.array(states)

    def check_true_win(self):
        """check the meta board for a winner."""
        return self.check_win(self.sub_boards)

    def check_true_tie(self):
        """check whether the full game is tied."""
        return len(self.empty_sub_places) == 0 and self.check_true_win() == 0

if __name__ == "__main__":
    from uttt.game.q_training import Games

    cores = multiprocessing.cpu_count()
    print(cores)
    start_time = time.time()
    # 1) TRAIN

    # random_player = random.choice([False, True, True])
    #
    # print(f"Training against random: {random_player}")
    #
    # games = Games(
    # num_games=350_000,     # increase this for stronger agent
    # processes=cores,
    # log_every=10_000,
    # chunk_size=100,
    # random_player=random_player
    # )
    #
    # games.multi_process_train()
    #
    # print("TRAINING DONE")
    # print("Agent win %:", 100 * games.agent_wins / games.num_games)
    # print("Player win %:", 100 * games.player_wins / games.num_games)
    # print("Tie %:", 100 * games.ties / games.num_games)
    #
    # #Save resulting Q-table (already a normal dict)
    # #hashing.save_qtable("q.pkl", games.game.q_table)
    #
    # # 2) (OPTIONAL) EVALUATION RUN AFTER TRAINING
    # print("Linux training done")
    # shrinker.refresh()



    eval_games = Games(num_games=1_000, processes=None, log_every=10, random_player=False)
    eval_games.single_process_train(training=True, epsilon=0.0)

    print("eval agent win %:", 100 * eval_games.agent_wins / eval_games.num_games)
    print("eval player win %:", 100 * eval_games.player_wins / eval_games.num_games)
    print("eval tie %:", 100 * eval_games.ties / eval_games.num_games)
    print(
        "random % (fallback during eval):",
        100 * eval_games.game.random_plays / eval_games.game.num_plays
        if eval_games.game.num_plays > 0 else 0.0
    )
    end_time = time.time()
    delta_time = end_time - start_time
    hours = delta_time // 3600
    mins = (delta_time - 3600 * hours) // 60
    print(f"overall run time is {hours}:{mins}h")
