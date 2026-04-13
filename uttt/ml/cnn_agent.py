# cnn play and eval utilities

# openmp runtime workaround
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
import numpy as np
import torch
import torch.nn as nn

from uttt.game.logic import UltimateTicTacToeGame
import uttt.ml.cnn_core as cnn_core


class UltimateTicTacToeCNN(UltimateTicTacToeGame):
    """pick agent moves with a cnn value model."""

    def __init__(self, model: nn .Module, device: torch.device, mode: str, **kwargs):
        """store the cnn model and mode."""
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.mode = mode
        self.model.eval()

    def board_to_tensor(self, board9x9: np.ndarray) -> torch.Tensor:
        """convert a board into a model tensor."""
        x = torch.from_numpy(board9x9.astype(np.float32).reshape((1, 1, 9, 9)))
        return x.to(self.device)

    @torch.no_grad()
    def value_of_board(self, board9x9: np.ndarray) -> float:
        """score one board with the cnn."""
        x = self.board_to_tensor(board9x9)
        v = self.model(x).item()
        return float(v)

    def current_board_value(self):
        """return the cnn value for the current board."""
        return self.value_of_board(self.board_rep)

    def best_from_moves(self, moves):
        """pick the highest-value move."""
        best = None
        best_score = -1e30

        for bi, bj, r, c in moves:
            self.place_in_rep(bi, bj, r, c, self.agent_symbol)
            score = self.value_of_board(self.board_rep)
            self.place_in_rep(bi, bj, r, c, 0)

            if score > best_score:
                best_score = score
                best = (bi, bj, r, c)
        return best

    def agent_smart_move(self):
        """pick and apply the cnn move."""
        if self.mode == "heuristic":
            super().agent_smart_move()
            return

        all_moves = self.legal_moves()

        if self.mode == "random":
            self.apply_move(*random.choice(all_moves), self.agent_symbol)
            return

        if self.mode != "pure_cnn":
            raise ValueError(f"Unknown CNN mode: {self.mode}")

        best = self.best_from_moves(all_moves)
        if best is None:
            self.random_plays += 1
            best = random.choice(all_moves)
        self.apply_move(*best, self.agent_symbol)
        return

def load_model(model_option: str):
    """load one trained cnn model."""
    return cnn_core.load_trained_model(model_option=model_option)

def play_games(model: nn.Module, device: torch.device, mode: str, n_games: int = 2000, random_player: bool = True):
    """play repeated games for quick evaluation."""

    agent_w = 0
    player_w = 0
    ties = 0

    game = UltimateTicTacToeCNN(
        model=model,
        device=device,
        mode=mode,
        q_table={},
        training=False,
        multiprocess=False,
        random_player=random_player
    )
    for _ in range(n_games):
        if _ % 10 == 0:
            print(f"game {_}")
        game.play_one_game(epsilon=0, training=False)
        w = game.check_true_win()
        if w == 1:
            agent_w += 1
        elif w == -1:
            player_w += 1
        else:
            ties += 1

    total = n_games
    print(f"mode: {mode}")
    print(f"agent win %: {100.0 * agent_w / total:.2f}")
    print(f"player win %: {100.0 * player_w / total:.2f}")
    print(f"tie %: {100.0 * ties / total:.2f}")
    if getattr(game, "num_plays", 0) > 0:
        print(f"fallback random %: {100.0 * game.random_plays / game.num_plays:.3f}")
    else:
        print("no moves tracked")

    return agent_w, player_w, ties
def test_evaluations(model_option: str):
    """print cnn values for one sampled finished game."""

    model, device = load_model(model_option)
    game = UltimateTicTacToeCNN(
        model=model,
        device=device,
        mode="random",
        q_table={},
        training=False,
        multiprocess=False,
        random_player=True
    )
    boards = game.play_one_game(training=False)
    win = game.check_true_win()
    print(f"win: {win}")
    for board in boards:
        print(game.value_of_board(UltimateTicTacToeGame.get_board_from_int(board)))

if __name__ == "__main__":
    test_evaluations("C")
    for model_option in ["A", "B", "C", "D", "E"]:
        model, device = load_model(model_option)
        play_games(model, device, mode="random", n_games=1000)
