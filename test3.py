from numpy.linalg import svd

from func import generate_matrix_with_singular_values

N = 5000        # Matrix size
n = 4800         # Matrix size
k = 500          # Rank

A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)


