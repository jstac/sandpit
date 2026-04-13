import numpy as np
from numba import njit
import time


@njit
def std_of_squared_normals(n):
    x = np.random.randn(n)
    for i in range(n):
        x[i] = x[i] ** 2
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n
    var = 0.0
    for i in range(n):
        var += (x[i] - mean) ** 2
    var /= (n - 1)
    return np.sqrt(var)


n = 100_000_000

# Warm up JIT
std_of_squared_normals(10)

t = time.time()
sd = std_of_squared_normals(n)
elapsed = time.time() - t

print(f"Standard deviation: {sd:.6f}")
print(f"Time: {elapsed:.6f} seconds")
