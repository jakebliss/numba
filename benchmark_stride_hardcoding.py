import io
import os

import matplotlib.pyplot as plt
import numba.cuda as cuda
import numpy as np
import tqdm


def time_function(f, kernel_launch_args, n, *args):
    f[kernel_launch_args](*args)
    start = cuda.event()
    end = cuda.event()
    cuda_event_time = 0
    wall_time = 0
    start.record()
    _s = time.time()

    for i in range(n):
        f[kernel_launch_args](*args)

    end.record()
    end.synchronize()
    _e = time.time()
    cuda_event_time += cuda.event_elapsed_time(start, end)
    wall_time += _e - _s

    return cuda_event_time / n, wall_time / n * 1000


@cuda.jit
def increment(x, y, z):
    j = cuda.grid(1)
    if j < x.shape[1]:
        for i in range(x.shape[0]):
            x[i, j] += 1


@cuda.jit
def cumulative_column_sum(x, y, z):
    j = cuda.grid(1)
    if j < x.shape[1]:
        for i in range(1, x.shape[0]):
            y[i, j] += x[i - 1, j]


@cuda.jit
def column_sum(x, y, z):
    j = cuda.grid(1)
    if j < x.shape[1]:
        for i in range(x.shape[0]):
            y[i, j] += x[i - 1, j]


@cuda.jit
def elementwise_product(x, y, z):
    j = cuda.grid(1)
    if j < x.shape[1]:
        for i in range(x.shape[0]):
            z[i, j] = x[i, j] * y[i, j]


@cuda.jit
def set_zero(x, y, z):
    j = cuda.grid(1)
    if j < x.shape[1]:
        for i in range(x.shape[0]):
            x[i, j] = 0


@cuda.jit
def conv2D(x, y, z):
    j = cuda.grid(1)
    if j < x.shape[1]:
        for i in range(x.shape[0]):
            dot = 0
            for k in range(3):
                for l in range(3):
                    dot += x[i, j] * y[k, l]
            z[i, j] = dot


m = 1
ns = [1000, 10_000, 100_000, 1_000_000, 10_000_000]
tables = []
benchmarks = [increment, cumulative_column_sum, column_sum, set_zero, conv2D, elementwise_product]
opts = ["", "1"]

pbar = tqdm.tqdm(total=len(opts) * len(benchmarks) * len(ns))
results = {}
for benchmark in benchmarks:
    results[benchmark.py_func.__name__] = {
        opt: {"contig": [], "bcast": []} for opt in opts
    }
    for numba_hardcode_strides in opts:
        for n in ns:
            os.environ["NUMBA_HARDCODE_STRIDES"] = numba_hardcode_strides
            benchmark.definitions = {}

            x_cont = cuda.to_device(np.ones((n, m), dtype=np.float32))
            x_bcast = cuda.to_device(
                np.broadcast_to(np.ones((1, 1), dtype=np.float32), (n, m))
            )

            y_cont = cuda.to_device(np.ones((n, m), dtype=np.float32))
            y_bcast = cuda.to_device(
                np.broadcast_to(np.ones((1, 1), dtype=np.float32), (n, m))
            )

            z_cont = cuda.to_device(np.ones((n, m), dtype=np.float32))
            z_bcast = cuda.to_device(
                np.broadcast_to(np.ones((1, 1), dtype=np.float32), (n, m))
            )

            trials = int(1e7 / n)

            t1_events, t1_wall = time_function(
                benchmark, (1, m), trials, x_cont, y_cont, z_cont
            )
            t2_events, t2_wall = time_function(
                benchmark, (1, m), trials, x_bcast, y_bcast, z_bcast
            )

            contig_ptx = io.StringIO()
            contig_ptx.write(list(benchmark.definitions.values())[0].ptx)
            contig_ptx.seek(0)

            bcast_ptx = io.StringIO()
            bcast_ptx.write(list(benchmark.definitions.values())[1].ptx)
            bcast_ptx.seek(0)

            with open(
                "{}-{}-contig.txt".format(
                    benchmark.py_func.__name__, numba_hardcode_strides
                ),
                "w",
            ) as f:
                f.write(contig_ptx.read())

            with open(
                "{}-{}-bcast.txt".format(
                    benchmark.py_func.__name__, numba_hardcode_strides
                ),
                "w",
            ) as f:
                f.write(bcast_ptx.read())

            results[benchmark.py_func.__name__][numba_hardcode_strides][
                "contig"
            ].append(t1_wall)
            results[benchmark.py_func.__name__][numba_hardcode_strides]["bcast"].append(
                t2_wall
            )

            pbar.update()


plt.figure()
for i, benchmark in enumerate(benchmarks):
    t_opt_contig = np.array(results[benchmark.py_func.__name__]["1"]["contig"])
    t_contig = np.array(results[benchmark.py_func.__name__][""]["contig"])

    t_opt_bcast = np.array(results[benchmark.py_func.__name__]["1"]["bcast"])
    t_bcast = np.array(results[benchmark.py_func.__name__][""]["bcast"])

    speedup_contig = t_contig / t_opt_contig
    speedup_bcast = t_bcast / t_opt_bcast

    plt.subplot(2, 3, i + 1)
    plt.title(benchmark.py_func.__name__)
    plt.plot(speedup_bcast, ".--")
    plt.xticks(range(len(ns)), ns)
    plt.ylabel("Speedup x")
    plt.xlabel("Array size")
    plt.savefig("results-{}.png".format(benchmark.py_func.__name__), dpi=300)
    plt.close()


plt.figure()
plt.title("Benchmark timing: elementwise_product")
plt.plot(results["elementwise_product"][""]["contig"], ".--")
plt.plot(results["elementwise_product"][""]["bcast"], ".--")
plt.xticks(range(len(ns)), ns)
plt.ylabel("Time (ms)")
plt.xlabel("Array size")
plt.legend(("Contiguous array", "Broadcast array"))
plt.savefig("motivation.png", dpi=300)
plt.close()
