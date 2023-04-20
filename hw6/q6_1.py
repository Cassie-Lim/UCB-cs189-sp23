import numpy as np

np.random.seed(13)
a = np.random.random((5, 5))
b = np.random.random((5,))
# 1. calculate trace
trace = np.einsum('ii', a)
print("Trace of a:", trace)

# 2. do matrix multiplication
mul_res = np.einsum('ij,j', a, b)
print("Multiplication of a and b:", mul_res)

# 3. do vector outer product
a = np.random.random((5,))
outer_product = np.einsum('i,j->ij', a, b)
print("Outer product of a and b:", outer_product)
