from func import generate_matrix_with_singular_values, svd

A, S, U, V = generate_matrix_with_singular_values(10, 5, sigma=[1, 2, 3, 0])

# print(A, S, U, V)
# print(svd(A))

