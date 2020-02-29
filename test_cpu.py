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
import time
from numba import cuda


@numba.jit()
def test(x, y):
    for i in range(x.shape[0]):
        y[i] = x[i] * x[i]

# Tests how long it take to access all of the elements in a 3 dimentional array
# by calculating the mean of the array
@cuda.jit
@numba.jit(nopython=True)
def arraySumTest(arr):
    arr_sum = 0
    start = cuda.event()
    end = cuda.event()
    timing_nb = 0
    timing_nb_wall = 0
    start.record()
    _s = time.time()

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                arr_sum += arr[i,j,k]

    mean = arr_sum / (arr.shape[0] * arr.shape[1] * arr.shape[2])

    end.record()
    end.synchronize()
    _e = time.time()
    timing_nb += cuda.get_elapsed_time(start, end)
    timing_nb_wall += (_e - _s)

    print('numba events:', timing_nb / 3, 'ms')
    print('numba wall  :', timing_nb_wall / 3 * 1000, 'ms')
    return mean

# Test how long it takes to do a convolution on a 10x10x10 array with a
# 2x2x2 kernel
@cuda.jit
@numba.jit(nopython=True)
def convolutionTest(array):
    kernel = np.ones((2,2,2), dtype=np.float32)

    start = cuda.event()
    end = cuda.event()
    timing_nb = 0
    timing_nb_wall = 0
    return_array = np.zeros((9,9,9))

    start.record()
    _s = time.time()
    for i in range(9):
        for j in range(9):
            for k in range(9):
                return_array[i,j,k] = dot(array[i:i + 2, j:j + 2, k:k + 2],
                                          kernel)

    end.record()
    end.synchronize()
    _e = time.time()
    timing_nb += cuda.get_elapsed_time(start, end)
    timing_nb_wall += (_e - _s)

    print('numba events:', timing_nb / 3, 'ms')
    print('numba wall  :', timing_nb_wall / 3 * 1000, 'ms')


@numba.jit(nopython=True)
def dot(arr_a, arr_b):
    sum = 0
    for i in range(arr_a.shape[0]):
        for j in range(arr_a.shape[1]):
            for k in range(arr_a.shape[2]):
                sum += arr_a[i, j, k] * arr_b[i,j,k]


x = np.ones(1000000, dtype=np.float32)
x = np.broadcast_to(np.ones(1, dtype=np.float32), 1000000)
y = np.ones(len(x), dtype=np.float32)
#test(x, y)
array = np.ones((10,10,10), dtype=np.float32)
transposed_array = array.transpose(0,2,1)
convolutionTest(array)
convolutionTest(transposed_array)
