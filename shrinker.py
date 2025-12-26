import lmdb
import os
def shrink(dst="qtable_shrink.lmdb", src="qtable.lmdb"):
    env = lmdb.open(src, readonly=True, lock=False, subdir=False)
    env.copy(dst, compact=True)   # compact=True = shrink pages
    env.close()

def refresh():
    shrink()
    try:
        if os.path.exists("qtable.lmdb"):
            os.remove("qtable.lmdb")
        os.rename("qtable_shrink.lmdb", "qtable.lmdb")
    except FileNotFoundError:
        print("File not found.")



