import lmdb

def shrink(src="qtable.lmdb", dst="qtable_shrink.lmdb"):
    env = lmdb.open(src, readonly=True, lock=False, subdir=False)
    env.copy(dst, compact=True)   # compact=True = shrink pages
    env.close()

shrink()
