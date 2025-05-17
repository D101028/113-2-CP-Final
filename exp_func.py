import time

import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

from func import generate_matrix_with_singular_values, sketched_svd

def error_avg(A, m, k, Sigma, V1):
    """
    :return: m, dt, avg, V_diff
    """
    t = time.time()
    sigma, V = sketched_svd(A, m)
    dt = time.time() - t
    S2 = sigma[:k]
    avg = np.average(np.abs((Sigma[:k] - S2) / S2))
    # Calculus the difference between V and V1
    # Vectors in V 可能會差一個負號
    V_diff = np.average([
        min((
                np.linalg.norm(V1[:, i] - V[:, i]), 
                np.linalg.norm(V1[:, i] + V[:, i])
            )) for i in range(k)
    ])
    print(f"{m} {dt:.4f} {avg:.4f} {V_diff:.4f}")
    return m, dt, avg, V_diff

def test_compression_ratio(N = 1000, n = 800, k_arr = [5, 10, 20, 40, 100]):
    """
    Test the compression ratio of the randomized SVD.

    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param k: Rank of the matrix.
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)

    import concurrent.futures # enhance the speed

    def process_k(k):
        A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)
        X, Y, T, Vs = [], [], [], []
        for m in range(k, n // 2, 10):
            m_val, dt, avg, V_diff = error_avg(A, m, k, Sigma, V)
            X.append(m_val)
            T.append(dt)
            Y.append(avg)
            Vs.append(V_diff)
        return k, X, Y, T, Vs

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_k, k) for k in k_arr]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Keep the original k_arr order
    results.sort(key=lambda x: k_arr.index(x[0]))

    for k, X, Y, T, Vs in results:
        ax1.plot(X, Y, label=f"k = {k}")
        ax2.plot(X, Vs, label=f"k = {k}")
        # ax3.plot(X, T, label=f"k = {k}")

    ax1.set_title("Error by Compressed Row Number (N = 1000, n = 800)")
    ax1.set_xlabel("m")
    ax1.set_ylabel("Average of Absolute Error")
    ax1.legend()
    
    ax2.set_title("V difference by Compressed Row Number (N = 1000, n = 800)")
    ax2.set_xlabel("m")
    ax2.set_ylabel("V difference (norm(V-V'))")
    ax2.legend()

    # ax3.set_title("Time by Compressed Row Number (N = 1000, n = 800)")
    # ax3.set_xlabel("m")
    # ax3.set_ylabel("Time (s)")
    # ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_compression_ratio()

