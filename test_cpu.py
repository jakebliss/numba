import os

import numba
import numpy as np
import time
from numba import cuda


def time_function(f, *args):
    f(*args)
    start = cuda.event()
    end = cuda.event()
    timing_nb = 0
    timing_nb_wall = 0
    start.record()
    _s = time.time()

    for i in range(3):
        f(*args)

    end.record()
    end.synchronize()
    _e = time.time()
    if not os.environ.get('NUMBA_ENABLE_CUDASIM'):
        timing_nb += cuda.event_elapsed_time(start, end)
    timing_nb_wall += _e - _s

    print("numba events:", timing_nb / 3, "ms")
    print("numba wall  :", timing_nb_wall / 3 * 1000, "ms")


@cuda.jit
def copy_vector(x, y):
    for i in range(x.shape[0]):
        y[i] = x[i]


@cuda.jit
def mean(arr, result):
    result[0] = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                result[0] += arr[i, j, k] / (arr.shape[0] * arr.shape[1] * arr.shape[2])


@cuda.jit
def conv3D(array, kernel, result):
    for i in range(array.shape[0] - kernel.shape[0]):
        for j in range(array.shape[1] - kernel.shape[1]):
            for k in range(array.shape[2] - kernel.shape[2]):

                dot = 0
                for m in range(kernel.shape[0]):
                    for n in range(kernel.shape[1]):
                        for o in range(kernel.shape[2]):
                            dot += array[i + m, j + n, k + o] * kernel[m, n, o]
                result[i, j, k] = dot


array = np.ones((10, 10, 10), dtype=np.float32)
result = np.zeros_like(array)
kernel = np.ones((3, 3, 3))

x = np.ones(100000, dtype=np.float32)
y = np.zeros(100000, dtype=np.float32)

time_function(conv3D, array, kernel, result)
time_function(mean, array, np.ones(1))
time_function(copy_vector, x, y)
