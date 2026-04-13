import numpy as np
from numba import jit
import time


@jit(fastmath=True)
def std_normals_numba(n):
    x = np.empty(n)
    for i in range(n):
        x[i] = np.random.randn()
    x = x * x
    total = 0.0
    for i in range(n):
        total += x[i]
    mean = total / n
    var_sum = 0.0
    for i in range(n):
        var_sum += (x[i] - mean) ** 2
    std_dev = np.sqrt(var_sum / (n - 1))
    return mean, std_dev


n = 100_000_000

# Warm up JIT
std_normals_numba(10)

start = time.time()
mean, std_dev = std_normals_numba(n)
elapsed = time.time() - start

print(f"Mean of squared normals:    {mean:12.6f}")
print(f"Std dev of squared normals: {std_dev:12.6f}")
print(f"Elapsed time:  {elapsed:10.4f} seconds")
