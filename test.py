import hashing
import numpy as np


b = hashing.decode_board_from_int(270982853926300859412819218711702712268)
b = np.array(b).reshape(9,9)
print(b)


q_table = hashing.load_qtable("q.json")

print("Loaded q_table:")
input("Continue:")


print(len(q_table))

items = q_table.items()

max = 0
max_board = None
for board,stats in items:
    if stats[1] > max:
        max = stats[1]
        max_board = board

print(max_board)
print(q_table[max_board])
