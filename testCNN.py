# cnn_play_test.py
import os
import random
import numpy as np
import torch
import torch.nn as nn

from logic import UltimateTicTacToeGame
from CNN import build_model, pick_device, cfg


# ============================================================
# =============== CNN-BASED AGENT SUBCLASS ===================
# ============================================================

class UltimateTicTacToeCNN(UltimateTicTacToeGame):
    """
    Subclass that chooses agent moves based on a trained CNN value function.
    We override agent move selection only; the rest stays identical.
    """

    def __init__(self, model: nn.Module, device: torch.device, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.device = device
        self.mode = mode  # "meta_only", "pure_cnn", "local_priority"
        self.model.eval()

    # -----------------------------
    # Board -> tensor (1 channel)
    # -----------------------------
    def _board_to_tensor(self, board9x9: np.ndarray) -> torch.Tensor:
        # shape (1,1,9,9)
        x = torch.from_numpy(board9x9.astype(np.float32))[None, None, :, :]
        return x.to(self.device)

    @torch.no_grad()
    def _value_of_board(self, board9x9: np.ndarray) -> float:
        x = self._board_to_tensor(board9x9)
        v = self.model(x).item()
        return float(v)

    # -----------------------------
    # CNN-based best move
    # -----------------------------
    def cnn_best_move(self, moves):
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
        """
        Three play styles:

        1) mode="meta_only":
            - meta win move if exists
            - meta block if threatened
            - then CNN choose among legal moves
            - IMPORTANT: does NOT do local mini-board win/block logic

        2) mode="pure_cnn":
            - no heuristics at all
            - CNN chooses among all legal moves

        3) mode="local_priority":
            - meta win if exists
            - meta block if threatened
            - local immediate win (subboard win) priority
            - local block priority
            - then CNN among remaining
        """

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
                # fallback random
                self.random_plays += 1
                best = random.choice(all_moves)
            self.apply_agent_move(*best)
            return True

        # ------------------ META WIN ------------------
        win = self.find_winning_move()  # your meta win finder (agent_symbol)
        if win is not None:
            self.apply_agent_move(*win)
            return True

        # ------------------ META BLOCK ----------------
        # Use your find_meta_block (blocks player_symbol threats)
        threats = self.find_meta_block()
        if threats:
            blocking = []
            # For each threatened board, block immediate local win there (same as your logic)
            for (bi, bj) in threats:
                dangers = self.find_immidiate_danger(bi, bj)
                for (r, c) in dangers:
                    # only if still legal
                    if (r, c) in self.empty_places[bi][bj]:
                        blocking.append((bi, bj, r, c))
            if blocking:
                best = self.cnn_best_move(blocking)
                if best is None:
                    self.random_plays += 1
                    best = random.choice(blocking)
                self.apply_agent_move(*best)
                return True

        # ------------------ META ONLY MODE -------------
        if self.mode == "meta_only":
            best = self.cnn_best_move(all_moves)
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
                best = self.cnn_best_move(winning_moves)
                if best is None:
                    self.random_plays += 1
                    best = random.choice(winning_moves)
                self.apply_agent_move(*best)
                return True

            # 2) Local block (block player immediate win)
            blocking_moves = []
            for bi, bj, r, c in all_moves:
                sb = self.full_board[bi][bj]
                if self._wins_sub(sb, self.player_symbol, r, c):
                    blocking_moves.append((bi, bj, r, c))
            if blocking_moves:
                best = self.cnn_best_move(blocking_moves)
                if best is None:
                    self.random_plays += 1
                    best = random.choice(blocking_moves)
                self.apply_agent_move(*best)
                return True

            # 3) Otherwise CNN on all moves
            best = self.cnn_best_move(all_moves)
            if best is None:
                self.random_plays += 1
                best = random.choice(all_moves)
            self.apply_agent_move(*best)
            return True

        # fallback
        best = self.cnn_best_move(all_moves)
        if best is None:
            self.random_plays += 1
            best = random.choice(all_moves)
        self.apply_agent_move(*best)
        return True


# ============================================================
# =================== EVALUATION HARNESS =====================
# ============================================================

def play_games(model: nn.Module, device: torch.device, mode: str, n_games: int = 2000, seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)

    agent_w = 0
    player_w = 0
    ties = 0

    game = UltimateTicTacToeCNN(
        model=model,
        device=device,
        mode=mode,
        q_table={},          # not using Q-table in this player
        training=False,
        multiprocess=False,
    )

    for _ in range(n_games):
        game.play_one_game(training=False, epsilon=0.0)  # uses our overridden agent_smart_move
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
    if game.num_plays > 0:
        print(f"Fallback random % (if any): {100.0 * game.random_plays / game.num_plays:.3f}")
    else:
        print("No moves tracked.")
    return agent_w, player_w, ties


def load_model(run_dir, model_option):
    device = pick_device(True)
    print("DEVICE:", device)

    model = build_model(model_option)
    path = f"{run_dir}/model_{model_option}.pt"

    print("Loading model from:", path)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    return model, device



if __name__ == "__main__":
    # Example:
    # After training, run:
    #   python cnn_play_test.py
    #
    # Make sure run_dir + model_option matches what you trained.

    run_dir = cfg.out_dir  # folder created by cnn_train.py
    model_option = "A"                     # A/B/C/D/E

    model, device = load_model(run_dir, model_option)

    # 3 play styles:
    play_games(model, device, mode="meta_only", n_games=2000, seed=1)
    play_games(model, device, mode="pure_cnn", n_games=2000, seed=2)
    play_games(model, device, mode="local_priority", n_games=2000, seed=3)
