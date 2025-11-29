# lmdb_qtable.py
import lmdb
import struct

KEY_BYTES = 32


class LMDBQTable:
    """
    LMDB-backed dict-like object for Q-table:
        - q[state_int] -> (value, count)
        - q[state_int] = (value, count)
        - state_int in q
        - q.get(state_int, default)
    """

    def __init__(self, path="qtable.lmdb", map_size=1 << 40):
        self.env = lmdb.open(
            path,
            map_size=map_size,
            subdir=True,
            max_readers=4096,
            lock=True,
            readahead=True,
            writemap=False,
        )

    # ---------- helpers ----------
    def _k(self, state_int: int) -> bytes:
        return state_int.to_bytes(KEY_BYTES, "big", signed=False)

    def _decode(self, data: bytes):
        return struct.unpack("dI", data)

    def _encode(self, val: float, count: int) -> bytes:
        return struct.pack("dI", val, count)

    # ---------- dict-like API ----------
    def __contains__(self, state_int: int) -> bool:
        key = self._k(state_int)
        with self.env.begin(write=False) as txn:
            return txn.get(key) is not None

    def __getitem__(self, state_int: int):
        key = self._k(state_int)
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                raise KeyError(state_int)
            return self._decode(data)

    def __setitem__(self, state_int: int, value_tuple):
        val, count = value_tuple
        key = self._k(state_int)
        with self.env.begin(write=True) as txn:
            txn.put(key, self._encode(val, count))

    def get(self, state_int: int, default=None):
        key = self._k(state_int)
        with self.env.begin(write=False) as txn:
            data = txn.get(key)
            if data is None:
                return default
            return self._decode(data)

    # ---------- your training APIs ----------
    def update_from_targets(self, board_score_list):
        """Single-process update: same averaging rule as your dict version."""
        with self.env.begin(write=True) as txn:
            for state_int, target in board_score_list:
                key = self._k(state_int)
                data = txn.get(key)

                if data is None:
                    old_v, count = 0.0, 0
                else:
                    old_v, count = self._decode(data)

                new_count = count + 1
                new_v = (old_v * count + target) / new_count
                txn.put(key, self._encode(new_v, new_count))

    def batch_merge_local_q(self, local_q):
        """
        local_q: { state_int : (avg_local, count_local) }
        Merging rule identical to your dictionary training.
        """
        with self.env.begin(write=True) as txn:
            for state_int, (avg_local, count_local) in local_q.items():
                key = self._k(state_int)
                data = txn.get(key)
                if data is None:
                    txn.put(key, self._encode(avg_local, count_local))
                else:
                    old_v, old_count = self._decode(data)
                    total = old_count + count_local
                    new_v = (old_v * old_count + avg_local * count_local) / total
                    txn.put(key, self._encode(new_v, total))
