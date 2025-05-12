from numpy.linalg import svd

from func import generate_matrix_with_singular_values

A, Sigma, U, V = generate_matrix_with_singular_values(5, 5, 5, (1, 10))

print(Sigma)
print(U)
print(V)

U0, Sigma0, V0T = svd(A, full_matrices=True)
print(Sigma0)
print(U0)
print(V0T.T)
