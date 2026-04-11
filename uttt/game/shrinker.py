import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from uttt.game.lmdb_qtable import LMDBQTable
from uttt.paths import FIXED_QTABLE_PATH, SHRUNK_QTABLE_PATH


def shrinkTo(dst=os.fspath(SHRUNK_QTABLE_PATH), src=os.fspath(FIXED_QTABLE_PATH)):
    qtable = LMDBQTable(path=src, readonly=True, lock=False, max_readers=1)
    try:
        qtable.copy(dst, compact=True)
    finally:
        qtable.close()

def refresh(src=os.fspath(FIXED_QTABLE_PATH), dst=os.fspath(SHRUNK_QTABLE_PATH)):
    shrinkTo(dst, src)
    try:
        if os.path.exists(dst):
            os.remove(src)
        os.rename(dst, src)
    except FileNotFoundError:
        print("File not found.")
if __name__ == "__main__":
    refresh()


