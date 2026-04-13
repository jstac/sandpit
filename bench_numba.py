import numpy as np
import numba as nb
import time

@nb.njit
def bench(n):
    z = np.random.randn(n)
    z_sq = z * z
    mean = np.sum(z_sq) / n
    std = np.sqrt(np.sum((z_sq - mean)**2) / n)
    return std

n = 100_000_000

# Warm-up JIT compilation
bench(10)

t_start = time.perf_counter()
std = bench(n)
t_end = time.perf_counter()

print(f"Standard deviation: {std:12.6f}")
print(f"Time:               {t_end - t_start:12.6f} seconds")
