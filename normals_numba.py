import numpy as np
import numba as nb
import time

@nb.njit(fastmath=True)
def normals_std(n):
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        z = np.random.randn()
        z_sq = z * z
        sum1 += z_sq
        sum2 += z_sq * z_sq

    mean = sum1 / n
    std_dev = np.sqrt(sum2 / n - mean * mean)
    return mean, std_dev

n = 100_000_000

# Warm up JIT
normals_std(10)

start = time.time()
mean, std_dev = normals_std(n)
elapsed = time.time() - start

print(f"Mean of squared normals:    {mean:.6f}")
print(f"Std dev of squared normals: {std_dev:.6f}")
print(f"Time: {elapsed:.4f} seconds")
