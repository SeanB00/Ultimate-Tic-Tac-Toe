# lmdb_qtable.py
import lmdb
import os
import struct
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uttt.paths import FIXED_QTABLE_PATH

KEY_BYTES = 32  # Size of keys



GLOBAL_TXN = None


class LMDBQTable:
    """
    LMDB-backed dict-like object for Q-table:
        - q[state_int] -> (value, count)
        - q[state_int] = (value, count)
        - state_int in q
        - q.get(state_int, default)
    """

    def __init__(self, path=os.fspath(FIXED_QTABLE_PATH), map_size=1 * 1 << 30):
        self.env = lmdb.open(
            os.fspath(path),
            map_size=map_size,
            subdir=False,
            max_readers=4096,
            lock=True,
            readahead=False,
            writemap=False,
        )

    # ---------- helpers ----------
    def key_bytes(self, state_int: int) -> bytes:
        return state_int.to_bytes(KEY_BYTES, "big", signed=False)

    def decode_value(self, data: bytes):
        return struct.unpack("dI", data)

    def encode_value(self, val: float, count: int) -> bytes:
        return struct.pack("dI", val, count)

    # ---------- dict-like API ----------
    def __contains__(self, state_int: int) -> bool:
        key = self.key_bytes(state_int)


        txn = GLOBAL_TXN
        return txn.get(key) is not None

    def __getitem__(self, state_int: int):
        key = self.key_bytes(state_int)


        txn = GLOBAL_TXN
        data = txn.get(key)
        if data is None:
            raise KeyError(state_int)
        return self.decode_value(data)

    def get(self, state_int: int, default=None):
        key = self.key_bytes(state_int)
        txn = GLOBAL_TXN
        data = txn.get(key)
        if data is None:
            return default
        return self.decode_value(data)

    def __setitem__(self, state_int: int, value_tuple):
        """Writes stay the same — must open a write txn."""
        val, count = value_tuple
        key = self.key_bytes(state_int)
        with self.env.begin(write=True) as txn:
            txn.put(key, self.encode_value(val, count))

    # ---------- your training APIs ----------
    def update_from_targets(self, board_score_list):
        """Single-process write. Safe."""
        with self.env.begin(write=True) as txn:
            for state_int, target in board_score_list:
                key = self.key_bytes(state_int)
                data = txn.get(key)

                if data is None:
                    old_v, count = 0.0, 0
                else:
                    old_v, count = self.decode_value(data)

                new_count = count + 1
                new_v = (old_v * count + target) / new_count
                txn.put(key, self.encode_value(new_v, new_count))

    def batch_merge_local_q(self, local_q):
        """Write-only, safe."""
        with self.env.begin(write=True) as txn:
            for state_int, (avg_local, count_local) in local_q.items():
                key = self.key_bytes(state_int)
                data = txn.get(key)
                if data is None:
                    txn.put(key, self.encode_value(avg_local, count_local))
                else:
                    old_v, old_count = self.decode_value(data)
                    total = old_count + count_local
                    new_v = (old_v * old_count + avg_local * count_local) / total
                    txn.put(key, self.encode_value(new_v, total))
