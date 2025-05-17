import numpy as np
from numpy.linalg import svd
import time

from func import generate_matrix_with_singular_values, sketched_svd

def test_compression_ratio(N = 1000, n = 800, k = 10):
    """
    Test the compression ratio of the randomized SVD.
    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param k: Rank of the matrix.
    :return: None
    """

    A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)
    S1 = Sigma[:k]
    t0 = time.time()
    for m in range(k, n, 10):
        sigma, V = sketched_svd(A, m)
        S2 = sigma[:k]
        t = time.time()
        print(m, t - t0, np.average(np.abs((S1 - S2)/S2)))
        t0 = t
    for m in range(n, N, 50):
        sigma, V = sketched_svd(A, m)
        S2 = sigma[:k]
        t = time.time()
        print(m, t - t0, np.average(np.abs((S1 - S2)/S2)))
        t0 = t
        
