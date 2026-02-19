import lmdb
import os
def shrinkTo(dst="qtable_shrink.lmdb", src="fixed_qtable.lmdb"):
    env = lmdb.open(src, readonly=True, lock=False, subdir=False)
    env.copy(dst, compact=True)   # compact=True = shrink pages
    env.close()

def refresh(src="fixed_qtable.lmdb", dst="qtable_shrink.lmdb"):
    shrinkTo(dst, src)
    try:
        if os.path.exists(dst):
            os.remove(src)
        os.rename(dst, src)
    except FileNotFoundError:
        print("File not found.")
if __name__ == "__main__":
    refresh("backup_qtable.lmdb")


