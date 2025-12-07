import lmdb
env = lmdb.open("qtable.lmdb", readonly=True, lock=False)
env.copy("qtable_compacted.lmdb")
