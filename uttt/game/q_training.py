import multiprocessing
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uttt.game import lmdb_qtable
from uttt.game.logic import UltimateTicTacToeGame


# multiprocessing training
EPS_START = 1.0
EPS_END = 0.2
GLOBAL_QTABLE = None


def init_worker(qtable):
    """attach the shared q-table to one worker."""
    global GLOBAL_QTABLE
    GLOBAL_QTABLE = qtable
    lmdb_qtable.GLOBAL_TXN = qtable.begin_read()


def run_games_chunk(args):
    """run one chunk of games in a worker."""
    num_games, seed, epsilon, random_player = args

    random.seed(seed)
    np.random.seed(seed)

    game = UltimateTicTacToeGame(
        q_table=GLOBAL_QTABLE,
        training=True,
        multiprocess=True,
        random_player=random_player,
    )

    local_q = {}
    agent_w = player_w = ties = 0

    for _ in range(num_games):
        game.play_one_game(epsilon=epsilon, training=True)
        winner = game.check_true_win()

        if winner == 1:
            agent_w += 1
        elif winner == -1:
            player_w += 1
        else:
            ties += 1

        for board_int, target in game.board_score_list:
            if board_int in local_q:
                avg, count = local_q[board_int]
                local_q[board_int] = ((avg * count + target) / (count + 1), count + 1)
            else:
                local_q[board_int] = (target, 1)

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
    """run repeated games for training or evaluation."""

    def __init__(self, num_games, processes=None, log_every=1000, chunk_size=50, random_player=True):
        """store run settings and counters."""
        self.num_games = num_games
        self.processes = processes or multiprocessing.cpu_count()
        self.log_every = log_every
        self.chunk_size = chunk_size

        self.agent_wins = 0
        self.player_wins = 0
        self.ties = 0
        self.random_player = random_player

        self.game = UltimateTicTacToeGame(
            q_table=None,
            training=False,
            multiprocess=False,
            random_player=random_player,
        )

    def multi_process_train(self):
        """train with multiple worker processes."""
        print(f"training {self.num_games} games with multiprocessing")

        start = time.time()
        num_chunks = (self.num_games + self.chunk_size - 1) // self.chunk_size

        args_list = []
        base_seed = int(time.time())
        completed = 0

        for i in range(num_chunks):
            frac = i / (num_chunks - 1) if num_chunks > 1 else 1.0
            epsilon = EPS_START + frac * (EPS_END - EPS_START)
            args_list.append((self.chunk_size, base_seed + i, epsilon, self.random_player))

        num_random = 0
        num_plays = 0

        for epoch_start in range(0, num_chunks, self.processes):
            epoch_end = min(epoch_start + self.processes, num_chunks)
            epoch_args = args_list[epoch_start:epoch_end]

            global GLOBAL_QTABLE
            GLOBAL_QTABLE = self.game.q_table

            with Pool(
                processes=self.processes,
                initializer=init_worker,
                initargs=(GLOBAL_QTABLE,),
            ) as pool:
                for local_q, a_w, p_w, t_w, r_p, n_p, eps in pool.imap_unordered(
                    run_games_chunk,
                    epoch_args,
                ):
                    games_in_chunk = a_w + p_w + t_w
                    completed += games_in_chunk

                    self.agent_wins += a_w
                    self.player_wins += p_w
                    self.ties += t_w

                    num_random += r_p
                    num_plays += n_p

                    if hasattr(self.game.q_table, "batch_merge_local_q"):
                        self.game.q_table.batch_merge_local_q(local_q)
                    else:
                        for board_int, (avg_local, count_local) in local_q.items():
                            if board_int in self.game.q_table:
                                old_v, old_count = self.game.q_table[board_int]
                                combined = (
                                    old_v * old_count + avg_local * count_local
                                ) / (old_count + count_local)
                                self.game.q_table[board_int] = (combined, old_count + count_local)
                            else:
                                self.game.q_table[board_int] = (avg_local, count_local)

                    if completed >= self.log_every and completed % self.log_every < self.chunk_size:
                        elapsed = time.time() - start
                        speed = completed / elapsed if elapsed > 0 else 0.0
                        randomness_pct = 100.0 * num_random / num_plays if num_plays > 0 else 0.0
                        print(f"{completed}/{self.num_games} games, {speed:.2f} games/sec")
                        print(f"fallback randomness: {randomness_pct:.2f}%")
                        print(f"epsilon: {eps:.4f}")

        total_t = time.time() - start
        speed = self.num_games / total_t if total_t > 0 else 0.0
        final_randomness = 100.0 * num_random / num_plays if num_plays > 0 else 0.0
        print(f"finished training in {total_t:.2f}s ({speed:.2f} games/sec)")
        print(f"fallback randomness: {final_randomness:.2f}%")
        print(f"agent moves: {num_plays}, fallback random moves: {num_random}")

    def single_process_train(self, training=False, epsilon=0.0):
        """run games in one process."""
        print(f"running {self.num_games} games single-process")
        start = time.time()

        for i in range(1, self.num_games + 1):
            self.game.play_one_game(epsilon=epsilon, training=training, random_player=self.random_player)

            winner = self.game.check_true_win()
            if winner == 1:
                self.agent_wins += 1
            elif winner == -1:
                self.player_wins += 1
            else:
                self.ties += 1

            if i % self.log_every == 0:
                elapsed = time.time() - start
                speed = i / elapsed if elapsed > 0 else 0.0
                print(f"{i}/{self.num_games} games, {speed:.2f} games/sec")

        total_t = time.time() - start
        speed = self.num_games / total_t if total_t > 0 else 0.0
        print(f"finished in {total_t:.2f}s ({speed:.2f} games/sec)")
