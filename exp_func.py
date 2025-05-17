import time

import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt

from func import generate_matrix_with_singular_values, sketched_svd

def error_avg(S1, A, m, k):
    """
    :return: m, dt, avg
    """
    t = time.time()
    sigma, V = sketched_svd(A, m)
    S2 = sigma[:k]
    dt = time.time() - t
    avg = np.average(np.abs((S1 - S2)/S2))
    print(f"{m} {dt:.4f} {avg:.4f}")
    return m, dt, avg

def test_compression_ratio(N = 1000, n = 800, k_arr = [10, 20, 30, 50, 100]):
    """
    Test the compression ratio of the randomized SVD.
    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param k: Rank of the matrix.
    :return: None
    """
    for k in k_arr:
        A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)
        S1 = Sigma[:k]
        X, Y = [], []
        for m in range(k, n // 2, 10):
            m, dt, avg = error_avg(S1, A, m, k)
            X.append(m)
            Y.append(avg)

        plt.plot(X, Y, label=f"k = {k}")
    plt.title("Compression Ratio by Average Error (N = 1000, n = 800)")
    plt.xlabel("m")
    plt.ylabel("Average Error")
    plt.legend()
    plt.show()
