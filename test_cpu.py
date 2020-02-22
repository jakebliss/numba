'''
import numba.cuda
import numpy as np

@numba.cuda.jit()
def test(x, y):
    for i in range(x.shape[0]):
        y[i] = x[i]


x = np.ones(1000000, dtype=np.float32)
#x = np.broadcast_to(np.ones(1, dtype=np.float32), 1000000)
y = np.ones(len(x), dtype=np.float32)
test(x, y)
'''


import numba
import numpy as np

@numba.jit()
def test(x, y):
    for i in range(x.shape[0]):
        y[i] = x[i] * x[i]


x = np.ones(1000000, dtype=np.float32)
x = np.broadcast_to(np.ones(1, dtype=np.float32), 1000000)
y = np.ones(len(x), dtype=np.float32)
test(x, y)
