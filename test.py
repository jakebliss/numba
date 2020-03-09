import os
import time

import numba.cuda as cuda
import numpy as np
import pandas as pd
import tqdm
import terminaltables


def time_function(f, kernel_launch_args, *args):
    f[kernel_launch_args](*args)
    start = cuda.event()
    end = cuda.event()
    cuda_event_time = 0
    wall_time = 0
    start.record()
    _s = time.time()

    n = 3
    for i in range(n):
        f[kernel_launch_args](*args)

    end.record()
    end.synchronize()
    _e = time.time()
    cuda_event_time += cuda.event_elapsed_time(start, end)
    wall_time += _e - _s

    return cuda_event_time / n, wall_time / n * 1000


@cuda.jit
def column_products(x, y):
    j = cuda.grid(1)
    for i in range(x.shape[0]):
        y[j] *= x[i, j]

@cuda.jit
def upper_triangular_column_products(x, y):
    j = cuda.grid(1)
    for i in range(j, x.shape[0]):
        y[j] *= x[i, j]

@cuda.jit
def conv2D(x, y):
    g = cuda.grid(1)
    jmin = g * (x.shape[1] // 32)
    jmax = (g + 1) * (x.shape[1] // 32)
    for i in range(x.shape[0]):
        for j in range(jmin, jmax):

            dot = 0
            for k in range(3):
                for l in range(3):
                    dot += x[i, j] * y[k * 3 + l]
            y[-1] += dot


m = 32
n = 2_000_000
tables = []
benchmarks = [column_products, upper_triangular_column_products, conv2D]
opts = ["", "1"]

pbar = tqdm.tqdm(total=len(opts) * len(benchmarks))
for benchmark in benchmarks:
    results = []
    for numba_hardcode_strides in opts:
        pbar.update()

        os.environ["NUMBA_HARDCODE_STRIDES"] = numba_hardcode_strides
        benchmark.definitions = {}

        x_cont = cuda.to_device(np.ones((n, m), dtype=np.float32))
        x_bcast = cuda.to_device(np.broadcast_to(np.ones((1, 1), dtype=np.float32), (n, m)))

        y_cont = cuda.to_device(np.ones(n, dtype=np.float32))
        y_bcast = cuda.to_device(np.ones(n, dtype=np.float32))

        t1_events, t1_wall = time_function(benchmark, (1, m), x_cont, y_cont)
        t2_events, t2_wall = time_function(benchmark, (1, m), x_bcast, y_bcast)

        results.append((t1_events, t2_events))

    speedup_contig = (results[0][0] - results[1][0]) / results[0][0]
    speedup_bcast = (results[0][1] - results[1][1]) / results[0][1]
    tables.append(
        terminaltables.AsciiTable(
            [
                [
                    benchmark.py_func.__name__,
                    "Opt. off (ms)",
                    "Opt. on (ms)",
                    "Relative speedup",
                ],
                [
                    "Contiguous",
                    "{:.1f}".format(results[0][0]),
                    "{:.1f}".format(results[1][0]),
                    "{:.1%}".format(speedup_contig),
                ],
                [
                    "Broadcast",
                    "{:.1f}".format(results[0][1]),
                    "{:.1f}".format(results[1][1]),
                    "{:.1%}".format(speedup_bcast),
                ],
            ]
        )
    )
pbar.close()

for table in tables:
    print(table.table)
