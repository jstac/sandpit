import numpy as np
import numba as nb
import time

@nb.njit
def bench_baseline(n):
    z = np.random.randn(n)
    z_sq = z * z
    mean = np.sum(z_sq) / n
    std = np.sqrt(np.sum((z_sq - mean)**2) / n)
    return std

@nb.njit(fastmath=True)
def bench_fastmath(n):
    z = np.random.randn(n)
    z_sq = z * z
    mean = np.sum(z_sq) / n
    std = np.sqrt(np.sum((z_sq - mean)**2) / n)
    return std

n = 100_000_000

# Warm up
bench_baseline(10)
bench_fastmath(10)

for name, f in [("baseline", bench_baseline),
                ("fastmath", bench_fastmath)]:
    t_start = time.perf_counter()
    std = f(n)
    t_end = time.perf_counter()
    print(f"{name}  std={std:.6f}  time={t_end - t_start:.4f}s")
