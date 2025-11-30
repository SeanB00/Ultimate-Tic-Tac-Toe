# convert_q_pickle_to_lmdb.py
import pickle, lmdb, struct

KEY_BYTES = 32

def k(x): return x.to_bytes(KEY_BYTES, "big", signed=False)
def v(val, cnt): return struct.pack("dI", val, cnt)

env = lmdb.open(
    "qtable.lmdb",
    map_size=1 << 40,
    subdir=False,       # IMPORTANT: store DB in a single file
    lock=True,
    writemap=False,
    max_readers=4096,
)


with open("q.pkl", "rb") as f:
    q = pickle.load(f)

with env.begin(write=True) as txn:
    for s, (val, cnt) in q.items():
        txn.put(k(s), v(val, cnt))

print("Done.")
