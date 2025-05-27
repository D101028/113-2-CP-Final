import numpy as np
from numpy.linalg import svd

from func import remove_small_values, sketched_svd, generate_matrix_with_singular_values, timer, title_print

def direct_svd(A):
    """
    Perform SVD on matrix A and return U, Sigma, V_T.
    :param A: Input matrix of shape (N, n).
    :return: Sigma, V_T from the SVD of A.
    """
    U, Sigma, V_T = svd(A, full_matrices=True)
    U, Sigma, V_T = map(remove_small_values, (U, Sigma, V_T))
    # print("U:\n", U)
    print(Sigma[:100])
    # print("V_T:\n", V_T)
    return Sigma, V_T.T

def random_svd(A, m):
    """
    Perform SVD on matrix A and return U, Sigma, V_T.
    :param A: Input matrix of shape (N, n).
    :return: Sigma, V_T from the SVD of A.
    """
    Sigma, V_T = sketched_svd(A, m)
    Sigma, V_T = map(remove_small_values, (Sigma, V_T))
    print(Sigma[:100])
    return Sigma, V_T.T

if __name__ == "__main__":
    # Parameters
    N = 1000        # Matrix size
    n = 800         # Matrix size
    k = 10          # Rank
    m = 100         # Compression dimension

    if m > N:
        print("m must be less than N")
        exit(1)

    # Generate random matrix
    A, sigma, U, V = generate_matrix_with_singular_values(N, n, k, (500, 1000))

    title_print("Origin Singular Values:")
    print(sigma[:100])

    # Direct SVD
    title_print("Direct SVD:")
    t1, result1 = timer(direct_svd, A)
    print("\n--------------------------------\n", 
          "Direct SVD time:", 
          t1)

    # Randomized SVD
    title_print("Randomized SVD:")
    t2, result2 = timer(random_svd, A, m)
    # print("\n--------------------------------\n", 
    #       "Randomized SVD time:", 
    #       t2)

    # S1 = result1[0][:100]
    S2 = result2[0][:100]
    sigma = sigma[:100]

    # Compare the first 100 singular values
    title_print("First 100 singular values error:")
    print((sigma - S2) / sigma)