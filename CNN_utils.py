# CNN_utils.py
# (CNN play/eval utilities + Kivy integration)

# --- OpenMP duplicate runtime workaround (Windows + torch/numpy/mkl)
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
import numpy as np
import torch
import torch.nn as nn

from logic import UltimateTicTacToeGame
from CNN import build_model, pick_device, TrainConfig


# ============================================================
# =============== CNN-BASED AGENT SUBCLASS ===================
# ============================================================

class UltimateTicTacToeCNN(UltimateTicTacToeGame):
    """Subclass that chooses agent moves based on a trained CNN value function."""

    def __init__(self, model: nn.Module, device: torch.device, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.mode = mode  # "meta_only", "pure_cnn", "local_priority"
        self.model.eval()

    # -----------------------------
    # Board -> tensor (3 channels)
    # -----------------------------
    def _board_to_tensor(self, board9x9: np.ndarray) -> torch.Tensor:
        # board9x9 is your {-1,0,1} representation.
        board = board9x9.astype(np.int8)
        x_plane = (board == 1).astype(np.float32)
        o_plane = (board == -1).astype(np.float32)
        e_plane = (board == 0).astype(np.float32)

        x = np.stack([x_plane, o_plane, e_plane], axis=0)  # (3,9,9)
        x = torch.from_numpy(x)[None, :, :, :]             # (1,3,9,9)
        return x.to(self.device)

    @torch.no_grad()
    def _value_of_board(self, board9x9: np.ndarray) -> float:
        x = self._board_to_tensor(board9x9)
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
            score = self._value_of_board(self.board_rep)
            self.place_in_rep(bi, bj, r, c, 0)

            if score > best_score:
                best_score = score
                best = (bi, bj, r, c)

        return best

    # -----------------------------
    # Agent move selection (3 modes)
    # -----------------------------

    def agent_smart_move(self):
        """Same as your current file, just uses 3ch value net now."""

        if self.mode == "heuristic":
            super().agent_smart_move()

        # playable boards
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

        # ------------------ PURE CNN ------------------
        if self.mode == "pure_cnn":
            best = self.cnn_best_move(all_moves)
            if best is None:
                self.random_plays += 1
                best = random.choice(all_moves)
            self.apply_agent_move(*best)
            return True

        # ------------------ META WIN ------------------
        win = self.find_winning_move()
        if win is not None:
            self.apply_agent_move(*win)
            return True

        # ------------------ META BLOCK ----------------
        threats = self.find_meta_block()
        if threats:
            blocking = []
            for (bi, bj) in threats:
                dangers = self.find_immidiate_danger(bi, bj)
                for (r, c) in dangers:
                    if (r, c) in self.empty_places[bi][bj]:
                        blocking.append((bi, bj, r, c))
            if blocking:
                best = self.best_from_moves(blocking)
                if best is None:
                    self.random_plays += 1
                    best = random.choice(blocking)
                self.apply_agent_move(*best)
                return True

        # ------------------ META ONLY MODE -------------
        if self.mode == "meta_only":
            best = self.best_from_moves(all_moves)
            if best is None:
                self.random_plays += 1
                best = random.choice(all_moves)
            self.apply_agent_move(*best)
            return True

        # ------------------ LOCAL PRIORITY MODE --------
        if self.mode == "local_priority":
            # 1) Local immediate win
            winning_moves = []
            for bi, bj, r, c in all_moves:
                sb = self.full_board[bi][bj]
                if self._wins_sub(sb, self.agent_symbol, r, c):
                    winning_moves.append((bi, bj, r, c))
            if winning_moves:
                best = self.best_from_moves(winning_moves)
                if best is None:
                    self.random_plays += 1
                    best = random.choice(winning_moves)
                self.apply_agent_move(*best)
                return True


            # 3) Otherwise CNN on all moves
            best = self.best_from_moves(all_moves)
            if best is None:
                self.random_plays += 1
                best = random.choice(all_moves)
            self.apply_agent_move(*best)
            return True

        # fallback
        best = self.best_from_moves(all_moves)
        if best is None:
            self.random_plays += 1
            best = random.choice(all_moves)
        self.apply_agent_move(*best)
        return True


# ============================================================
# MODEL LOADER (NEW CHECKPOINT FORMAT)
# ============================================================

def load_model(run_dir: str, model_option: str):
    device = pick_device(True)
    print(">>> DEVICE:", device)

    model = build_model(model_option)
    path = f"{run_dir}/model_{model_option}.pt"

    print(">>> Loading model from:", path)
    obj = torch.load(path, map_location=device)

    # Backwards/forwards compatible:
    # - new format: {model_state, optimizer_state, step, history}
    # - older format (your old script): {model, optim, step}
    # - oldest format: raw state_dict
    if isinstance(obj, dict) and "model_state" in obj:
        model.load_state_dict(obj["model_state"])
        step = obj.get("step", None)
        if step is not None:
            print(">>> Loaded checkpoint step:", step)
    elif isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        step = obj.get("step", None)
        if step is not None:
            print(">>> Loaded checkpoint step:", step)
    else:
        model.load_state_dict(obj)

    model.to(device)
    model.eval()

    return model, device


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
        random=False
    )



    for _ in range(n_games):
        if _%50==0:
            print(_)
        game.play_one_game(training=False, epsilon=0.0)
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


if __name__ == "__main__":
    run_dir = TrainConfig.out_dir


    #play_games(model, device, mode="meta_only", n_games=1000)


    for model_option in ["A","B","C","D","E"]:

        model, device = load_model(run_dir, model_option)

        play_games(model, device, mode="heuristic", n_games=400)

    #play_games(model, device, mode="local_priority", n_games=1000)
