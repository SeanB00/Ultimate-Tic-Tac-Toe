# CNN_utils.py
# (CNN play/eval utilities + Kivy integration)

# --- OpenMP duplicate runtime workaround (Windows + torch/numpy/mkl)
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
from uttt.ml.cnn_core import load_trained_model


# ============================================================
# =============== CNN-BASED AGENT SUBCLASS ===================
# ============================================================

class UltimateTicTacToeCNN(UltimateTicTacToeGame):
    """Subclass that chooses agent moves based on a trained CNN value function."""

    def __init__(self, model: nn .Module, device: torch.device, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.mode = mode  # "pure_cnn", "heuristic", "random"
        self.model.eval()

    # -----------------------------
    # Board -> tensor (1 channel)
    # -----------------------------
    def board_to_tensor(self, board9x9: np.ndarray) -> torch.Tensor:
        # single-channel tensor with values in {-1,0,1}
        x = torch.from_numpy(board9x9.astype(np.float32).reshape((1,1,9,9)))
        return x.to(self.device)

    @torch.no_grad()
    def value_of_board(self, board9x9: np.ndarray) -> float:
        x = self.board_to_tensor(board9x9)
        v = self.model(x).item()


        return float(v)

    # -----------------------------
    # CNN-based best move
    # -----------------------------
    def best_from_moves(self, moves):

        best = None
        best_score = -1e30

        for bi, bj, r, c in moves:
            # apply move on board_rep temporarily
            self.place_in_rep(bi, bj, r, c, self.agent_symbol)
            score = self.value_of_board(self.board_rep)
            self.place_in_rep(bi, bj, r, c, 0)

            if score > best_score:
                best_score = score
                best = (bi, bj, r, c)



        return best

    # -----------------------------
    # Agent move selection
    # -----------------------------

    def agent_smart_move(self):
        """Pick and apply the CNN agent move."""
        if self.mode == "heuristic":
            super().agent_smart_move()
            return

        all_moves = self.get_available_moves()

        if self.mode == "random":
            self.apply_agent_move(*random.choice(all_moves))
            return

        if self.mode != "pure_cnn":
            raise ValueError(f"Unknown CNN mode: {self.mode}")

        best = self.best_from_moves(all_moves)
        if best is None:
            self.random_plays += 1
            best = random.choice(all_moves)
        self.apply_agent_move(*best)
        return

    # def player_smart_move(self):
    #     moves = self.get_available_moves()
    #     self.board_rep *= -1
    #     best_move = self.best_from_moves(moves)
    #     self.board_rep *= -1
    #     self.apply_player_move(*best_move)

def load_model(model_option: str):
    return load_trained_model(model_option=model_option)

# ============================================================
# OPTIONAL: QUICK EVAL HARNESS (kept from your file)
# ============================================================

def play_games(model: nn.Module, device: torch.device, mode: str, n_games: int = 2000):


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
        randomPlayer=True
    )


    for _ in range(n_games):
        if _%10==0:
            print(f"Game {_}")
        game.play_one_game(epsilon=0, training=False)
        w = game.check_true_win()
        if w == 1:
            agent_w += 1
        elif w == -1:
            player_w += 1
        else:
            ties += 1

    total = n_games
    print(f"\n=== MODE: {mode} ===")
    print(f"Agent win % : {100.0 * agent_w / total:.2f}")
    print(f"Player win %: {100.0 * player_w / total:.2f}")
    print(f"Tie %      : {100.0 * ties / total:.2f}")
    if getattr(game, "num_plays", 0) > 0:
        print(f"Fallback random % (if any): {100.0 * game.random_plays / game.num_plays:.3f}")
    else:
        print("No moves tracked.")

    return agent_w, player_w, ties


def test_evaluations(model_option: str):
    """This function takes an end game board and tests if model outputs 1/-1 for it"""

    model, device = load_model(model_option)
    game = UltimateTicTacToeCNN(
        model=model,
        device=device,
        mode="random",
        q_table={},
        training=False,
        multiprocess=False,
        randomPlayer=True
    )
    boards = game.play_one_game(training=False)
    win = game.check_true_win()
    print(f"Win: {win}")
    for board in boards:

        print(game.value_of_board(game.get_board_from_int(board)))







if __name__ == "__main__":
    test_evaluations("E")
    for model_option in ["A","B","C","D","E"]:
        model, device = load_model(model_option)
        play_games(model, device, mode="pure_cnn", n_games=2000)
