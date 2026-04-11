import os
import struct
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import lmdb

from uttt.game import hashing
from uttt.paths import FIXED_QTABLE_PATH

KEY_BYTES = 32
VALUE_FMT = "dI"
VALUE_SIZE = struct.calcsize(VALUE_FMT)

GLOBAL_TXN = None


class LMDBQTable:
    """
    LMDB-backed dict-like object for Q-table:
        - q[state_int] -> (value, count)
        - q[state_int] = (value, count)
        - state_int in q
        - q.get(state_int, default)
    """

    def __init__(
        self,
        path=os.fspath(FIXED_QTABLE_PATH),
        map_size=22 * 1 << 30,
        readonly=False,
        lock=True,
        readahead=False,
        max_readers=4096,
    ):
        self.path = os.fspath(path)
        self.env = lmdb.open(
            self.path,
            map_size=map_size,
            subdir=False,
            max_readers=max_readers,
            readonly=readonly,
            lock=lock,
            readahead=readahead,
            writemap=False,
        )

    # ---------- helpers ----------
    def begin_read(self):
        return self.env.begin(write=False)

    def close(self):
        self.env.close()

    def key_bytes(self, state_int: int) -> bytes:
        return state_int.to_bytes(KEY_BYTES, "big", signed=False)

    def state_int_from_key(self, key: bytes) -> int:
        return int.from_bytes(key, "big", signed=False)

    def key_is_valid(self, key: bytes) -> bool:
        return len(key) == KEY_BYTES

    def value_is_valid(self, data: bytes) -> bool:
        return len(data) == VALUE_SIZE

    def decode_value(self, data: bytes):
        return struct.unpack(VALUE_FMT, data)

    def encode_value(self, val: float, count: int) -> bytes:
        return struct.pack(VALUE_FMT, val, count)

    def decode_entry(self, key: bytes, data: bytes):
        if not self.key_is_valid(key):
            raise ValueError(f"Expected {KEY_BYTES} key bytes, got {len(key)}")
        if not self.value_is_valid(data):
            raise ValueError(f"Expected {VALUE_SIZE} value bytes, got {len(data)}")
        state_int = self.state_int_from_key(key)
        q_value, visits = self.decode_value(data)
        return state_int, q_value, visits

    def decode_board_from_state_int(self, state_int: int):
        return hashing.decode_board_from_int(state_int)

    def decode_board_from_key(self, key: bytes):
        return self.decode_board_from_state_int(self.state_int_from_key(key))

    def iter_raw_items(self, max_entries=None, txn=None):
        if txn is None:
            with self.begin_read() as read_txn:
                yield from self.iter_raw_items(max_entries=max_entries, txn=read_txn)
            return

        cursor = txn.cursor()
        for idx, (key, data) in enumerate(cursor):
            if max_entries is not None and idx >= max_entries:
                break
            yield key, data

    def iter_entries(self, max_entries=None, txn=None):
        for key, data in self.iter_raw_items(max_entries=max_entries, txn=txn):
            if not self.key_is_valid(key) or not self.value_is_valid(data):
                continue
            yield self.decode_entry(key, data)

    def info(self):
        return self.env.info()

    def stat(self):
        with self.begin_read() as txn:
            return txn.stat()

    def page_size(self):
        return self.env.stat()["psize"]

    def copy(self, dst, compact=False):
        self.env.copy(os.fspath(dst), compact=compact)

    def read_bytes(self, state_int: int, txn=None):
        key = self.key_bytes(state_int)

        if txn is not None:
            return txn.get(key)
        if GLOBAL_TXN is not None:
            return GLOBAL_TXN.get(key)
        with self.begin_read() as read_txn:
            return read_txn.get(key)

    # ---------- dict-like API ----------
    def __contains__(self, state_int: int) -> bool:
        return self.read_bytes(state_int) is not None

    def __getitem__(self, state_int: int):
        data = self.read_bytes(state_int)
        if data is None:
            raise KeyError(state_int)
        return self.decode_value(data)

    def get(self, state_int: int, default=None):
        data = self.read_bytes(state_int)
        if data is None:
            return default
        return self.decode_value(data)

    def __setitem__(self, state_int: int, value_tuple):
        """Writes stay the same; this opens a write txn."""
        val, count = value_tuple
        key = self.key_bytes(state_int)
        with self.env.begin(write=True) as txn:
            txn.put(key, self.encode_value(val, count))

    # ---------- training APIs ----------
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
