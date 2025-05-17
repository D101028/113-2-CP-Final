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
    dt = time.time() - t
    S2 = sigma[:k]
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
    fig, (ax1, ax2) = plt.subplots(1, 2)  # 一行兩張子圖
    for k in k_arr:
        A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)
        S1 = Sigma[:k]
        X, Y, T = [], [], []
        for m in range(k, n // 2, 10):
            m, dt, avg = error_avg(S1, A, m, k)
            X.append(m)
            T.append(dt)
            Y.append(avg)
        ax1.plot(X, Y, label=f"k = {k}")
        ax2.plot(X, T, label=f"k = {k}")
    ax1.set_title("Error by Compression Ratio (N = 1000, n = 800)")
    ax1.set_xlabel("m")
    ax1.set_ylabel("Average of Absolute Error")
    ax1.legend()

    ax2.set_title("Time by Compression Ratio (N = 1000, n = 800)")
    ax2.set_xlabel("m")
    ax2.set_ylabel("Time (s)")
    ax2.legend()

    plt.tight_layout()
    plt.show()
