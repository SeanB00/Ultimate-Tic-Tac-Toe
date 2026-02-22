# q_only_eval.py

import random
import time

from logic import UltimateTicTacToeGame
from lmdb_qtable import LMDBQTable


class QOnlyUltimateTicTacToe(UltimateTicTacToeGame):
    """
    Pure Q-table agent.
    NO heuristics, NO meta logic, NO blocking.
    Chooses argmax Q(s') over all legal moves.
    """

    def agent_smart_move(self):
        # determine playable subboards
        if self.curr_board is None:
            playable = tuple(self.empty_sub_places)
        else:
            playable = (
                [self.curr_board]
                if self.curr_board in self.empty_sub_places
                else tuple(self.empty_sub_places)
            )

        # collect all legal moves

        moves = [
            (bi, bj, r, c)
            for (bi, bj) in playable
            for (r, c) in self.empty_places[bi][bj]
        ]

        # best_move = None
        # best_score = -1e18
        #
        # for bi, bj, r, c in moves:
        #     # simulate move
        #     self.place_in_rep(bi, bj, r, c, self.agent_symbol)
        #     b_int = self.get_board_int()
        #     self.place_in_rep(bi, bj, r, c, 0)
        #
        #     if b_int in self.q_table:
        #         val, _ = self.q_table[b_int]
        #         if val > best_score:
        #             best_score = val
        #             best_move = (bi, bj, r, c)
        #
        # # apply best Q move if exists
        # if best_move is not None:
        #     self.apply_agent_move(*best_move)
        #     self.num_plays += 1
        #  return True

        # fallback: random legal move
        self.random_plays += 1
        bi, bj, r, c = random.choice(moves)
        self.apply_agent_move(bi, bj, r, c)
        return False


def run_q_only_eval(num_games=2000):
    qtable = LMDBQTable("fixed_qtable.lmdb")

    game = QOnlyUltimateTicTacToe(
        q_table=qtable,
        training=False,
        multiprocess=False,
        random=True
    )

    agent_wins = 0
    player_wins = 0
    ties = 0

    start = time.time()

    for i in range(1, num_games + 1):

        game.play_one_game(training=False, epsilon=0.0)

        w = game.check_true_win()
        if w == 1:
            agent_wins += 1
        elif w == -1:
            player_wins += 1
        else:
            ties += 1

        if i % 200 == 0:
            print(f"{i}/{num_games} games played")

    elapsed = time.time() - start

    print("\n===== Q-ONLY EVALUATION =====")
    print(f"Games: {num_games}")
    print(f"Agent wins:  {agent_wins} ({100 * agent_wins / num_games:.2f}%)")
    print(f"Player wins:{player_wins} ({100 * player_wins / num_games:.2f}%)")
    print(f"Ties:       {ties} ({100 * ties / num_games:.2f}%)")

    if game.num_plays > 0:
        print(
            f"Fallback randomness: "
            f"{100 * game.random_plays / game.num_plays:.2f}%"
        )
    else:
        print("Fallback randomness: N/A")

    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    run_q_only_eval(num_games=100)
