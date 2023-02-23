import jax.numpy as jnp
import numpy as np
from jax import jit, random
import time

N = 1000

# Generate random matrices A and B
key = random.PRNGKey(0)
subkeys = random.split(key, num=6)
A = random.normal(subkeys[0], (N, N))
B = random.normal(subkeys[1], (N, N))

A_blank = random.normal(subkeys[2], (N, N))
B_blank = random.normal(subkeys[3], (N, N))

A_blank_ = random.normal(subkeys[4], (N, N))
B_blank_ = random.normal(subkeys[5], (N, N))

assert (not jnp.array_equal(A_blank_, A_blank))
# Define matrix multiplication function
@jit
def matmul(A, B):
    return jnp.dot(A, B)

matmul(A_blank, B_blank) # blank run to compile
matmul(A_blank_, B_blank_) # blank run to compile

# Multiply matrices A and B
N_ITER = 1000
times = np.zeros((N_ITER,))

for i in jnp.arange(N_ITER):
    key, *subkeys = random.split(key, num=3)
    A = random.normal(subkeys[0], (N, N))
    B = random.normal(subkeys[1], (N, N))
    start = time.time()
    C = matmul(A, B).block_until_ready() # ~ 170 mus (microseconds)
    end_t = time.time()
    times[i] = end_t - start

# Print the output matrix and the execution time
# print("Output matrix C:\n", C)
print("Elapsed time:", (np.mean(times))*1000000, "mus")
