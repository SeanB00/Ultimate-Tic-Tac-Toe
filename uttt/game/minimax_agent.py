"""Depth-limited alpha-beta search using the CNN as the leaf evaluator."""

import math
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uttt.ml.cnn_agent import UltimateTicTacToeCNN


class UltimateTicTacToeMinimaxCNN(UltimateTicTacToeCNN):
    """Search a few plies ahead and score cutoff states with the CNN."""

    WIN_SCORE = 1_000_000.0

    def __init__(
        self,
        model,
        device,
        search_depth=3,
        branch_limit=None,
        use_smart_ordering=True,
        **kwargs,
    ):
        """Store search settings and reuse the CNN utilities from the base class."""
        super().__init__(model=model, device=device, mode="minimax_cnn", **kwargs)
        self.search_depth = max(1, int(search_depth))
        self.branch_limit = None if branch_limit is None else max(1, int(branch_limit))
        self.use_smart_ordering = use_smart_ordering

    def _copy_for_search(self):
        """Clone the mutable game state while sharing the loaded model."""
        clone = object.__new__(type(self))
        clone.model = self.model
        clone.device = self.device
        clone.mode = self.mode
        clone.search_depth = self.search_depth
        clone.branch_limit = self.branch_limit
        clone.use_smart_ordering = self.use_smart_ordering

        clone.board_rep = self.board_rep.copy()
        clone.full_board = self.full_board.copy()
        clone.sub_boards = self.sub_boards.copy()
        clone.empty_places = [[cells.copy() for cells in row] for row in self.empty_places]
        clone.empty_sub_places = self.empty_sub_places.copy()
        clone.curr_board = self.curr_board

        clone.player_symbol = self.player_symbol
        clone.agent_symbol = self.agent_symbol
        clone.random_player = self.random_player
        clone.num_plays = self.num_plays
        clone.random_plays = self.random_plays

        clone.q_table = self.q_table
        clone.training = self.training
        clone.multiprocess = self.multiprocess
        clone.gamma = self.gamma
        clone.board_score_list = list(self.board_score_list)
        return clone

    @staticmethod
    def _position_rank(index):
        """Prefer center, then corners, then edges for deterministic ordering."""
        if index == 1:
            return 0
        if index in {0, 2}:
            return 1
        return 2

    def _ordered_moves(self, symbol, legal_moves=None):
        """Put tactical candidates first and keep the rest in a stable order."""
        if legal_moves is None:
            legal_moves = self.legal_moves()

        moves = sorted(
            legal_moves,
            key=lambda move: (
                self._position_rank(move[0]),
                self._position_rank(move[1]),
                self._position_rank(move[2]),
                self._position_rank(move[3]),
                move,
            ),
        )
        if not self.use_smart_ordering or len(moves) <= 1:
            return moves if self.branch_limit is None else moves[: self.branch_limit]

        opponent_symbol = self.player_symbol if symbol == self.agent_symbol else self.agent_symbol
        smart_moves = self.find_smart_move_candidates(symbol, opponent_symbol, moves)
        smart_set = set(smart_moves)
        ordered = list(smart_moves) + [move for move in moves if move not in smart_set]
        if self.branch_limit is not None:
            ordered = ordered[: self.branch_limit]
        return ordered

    def _terminal_value(self, depth_remaining):
        """Return a finished-game score from the agent's point of view."""
        winner = self.check_true_win()
        if winner == self.agent_symbol:
            return self.WIN_SCORE + depth_remaining
        if winner == self.player_symbol:
            return -self.WIN_SCORE - depth_remaining
        if self.check_true_tie():
            return 0.0
        return None

    def _cache_key(self, symbol, depth_remaining):
        """Build a cache key for one search node."""
        return self.board_rep.tobytes(), self.curr_board, symbol, depth_remaining

    def _minimax_value(self, depth_remaining, alpha, beta, symbol, cache):
        """Evaluate one node with alpha-beta pruning."""
        terminal = self._terminal_value(depth_remaining)
        if terminal is not None:
            return terminal
        if depth_remaining == 0:
            return self.value_of_board(self.board_rep)

        key = self._cache_key(symbol, depth_remaining)
        if key in cache:
            return cache[key]

        moves = self._ordered_moves(symbol)
        if not moves:
            return self.value_of_board(self.board_rep)

        opponent_symbol = self.player_symbol if symbol == self.agent_symbol else self.agent_symbol
        if symbol == self.agent_symbol:
            best_value = -math.inf
            for move in moves:
                child = self._copy_for_search()
                child.apply_move(*move, symbol)
                value = child._minimax_value(depth_remaining - 1, alpha, beta, opponent_symbol, cache)
                if value > best_value:
                    best_value = value
                if best_value > alpha:
                    alpha = best_value
                if alpha >= beta:
                    break
        else:
            best_value = math.inf
            for move in moves:
                child = self._copy_for_search()
                child.apply_move(*move, symbol)
                value = child._minimax_value(depth_remaining - 1, alpha, beta, opponent_symbol, cache)
                if value < best_value:
                    best_value = value
                if best_value < beta:
                    beta = best_value
                if alpha >= beta:
                    break

        cache[key] = best_value
        return best_value

    def search_best_move(self, symbol, legal_moves=None):
        """Search for the best move for the given symbol."""
        moves = self._ordered_moves(symbol, legal_moves)
        if not moves:
            return None

        maximizing = symbol == self.agent_symbol
        best_score = -math.inf if maximizing else math.inf
        best_move = None
        best_move = None
        alpha = -math.inf
        beta = math.inf
        cache = {}
        opponent_symbol = self.player_symbol if symbol == self.agent_symbol else self.agent_symbol

        for move in moves:
            child = self._copy_for_search()
            child.apply_move(*move, symbol)
            score = child._minimax_value(self.search_depth - 1, alpha, beta, opponent_symbol, cache)

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                if best_score > alpha:
                    alpha = best_score
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                if best_score < beta:
                    beta = best_score

            if alpha >= beta:
                break

        return best_move

    def best_from_moves(self, moves):
        """Pick the agent move chosen by the search."""
        return self.search_best_move(self.agent_symbol, moves)

    def agent_smart_move(self):
        """Pick and apply the minimax-CNN move."""
        all_moves = self.legal_moves()
        best = self.best_from_moves(all_moves)
        if best is None:
            self.random_plays += 1
            best = random.choice(all_moves)
        self.apply_move(*best, self.agent_symbol)
        return


