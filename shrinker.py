import lmdb
import os
def shrink(dst="qtable_shrink.lmdb", src="fixed_qtable.lmdb"):
    env = lmdb.open(src, readonly=True, lock=False, subdir=False)
    env.copy(dst, compact=True)   # compact=True = shrink pages
    env.close()

def refresh():
    shrink()
    try:
        if os.path.exists("qtable_shrink.lmdb"):
            os.remove("fixed_qtable.lmdb")
        os.rename("qtable_shrink.lmdb", "fixed_qtable.lmdb")
    except FileNotFoundError:
        print("File not found.")
if __name__ == "__main__":
    refresh()


