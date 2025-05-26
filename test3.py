import numpy as np
from numpy.linalg import svd

from func import sketched_svd, generate_matrix_with_singular_values, remove_small_values

A, Sigma, U, V = generate_matrix_with_singular_values(5, 5, 3)

U, sigma0, V0_T = svd(A)

sigma1, V1 = sketched_svd(A, 3)

# print(remove_small_values(V))
# print(remove_small_values(V0_T.T))

# print(Sigma, remove_small_values(sigma0))

# print(A)

# print()

# Sigma0 = np.zeros((5,5))
# for i in range(len(Sigma)):
#     Sigma0[i, i] = Sigma[i]
# print(U @ Sigma0 @ V.T)

# print()

# Sigma0 = np.zeros((5,5))
# for i in range(len(sigma0)):
#     Sigma0[i, i] = sigma0[i]
# print(U0 @ Sigma0 @ V0_T)


print()
[print(V[:, i]) for i in range(5)]

print()
[print(V0_T.T[:, i]) for i in range(5)]

print()
[print(V1[:, i]) for i in range(5)]

# error_avg(A, 5, 3, Sigma[:3], V)

print(Sigma, "\n", 
      remove_small_values(sigma0), "\n", 
      remove_small_values(sigma1))

print(np.linalg.norm(V[:, 0] - V1[:, 0]))
print(np.linalg.norm(V[:, 0] + V1[:, 0]))
