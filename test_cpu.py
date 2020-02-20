import numba
import numpy as np

@numba.jit(nopython=True)
def test(x):
    g = 0
    for i in range(x.shape[0]):
        g += x[i] * x[i]
    return g


a = np.broadcast_to(1, 1000000)
b = np.ones(1000000)
test(a)
test(b)
