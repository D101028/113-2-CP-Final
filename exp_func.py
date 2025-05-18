import time

import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

from func import generate_matrix_with_singular_values, sketched_svd

def error_avg(A, m, k, Sigma, V1):
    """
    :param A: Input matrix
    :param m: Compressed rank
    :param Sigma: Original sigular values
    :param V1: Original right sigular vectors
    :return: dt, avg, V_diff
    """
    t = time.time()
    sigma, V = sketched_svd(A, m)
    dt = time.time() - t
    S2 = sigma[:k]
    avg = np.average(np.abs((Sigma[:k] - S2) / S2))
    # Calculate the difference between V and V1
    # Vectors in V 可能會差一個負號
    V_diff = np.average([
        min((
                np.linalg.norm(V1[:, i] - V[:, i]), 
                np.linalg.norm(V1[:, i] + V[:, i])
            )) for i in range(k)
    ])
    print(f"{m} {dt:.4f} {avg:.4f} {V_diff:.4f}")
    return dt, avg, V_diff

def plot_graphs(results, graphs):
    for item in results:
        k = item[0]
        data = item[1:]
        for idx, (i, j, title, xlabel, ylabel) in enumerate(graphs):
            plt.figure(idx)
            plt.plot(data[i], data[j], label=f"k = {k}")

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()

def plot_graphs_together(results, graphs):
    num = len(graphs)
    fig, axs = plt.subplots(1, num)
    for item in results:
        k = item[0]
        data = item[1:]
        for idx, (i, j, title, xlabel, ylabel) in enumerate(graphs):
            ax = axs[idx]
            ax.plot(data[i], data[j], label=f"k = {k}")

            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()

def test_compression_ratio(N = 1000, n = 100, k_arr = [5, 10, 50, 100], max_m = 500, m_interval = 5):
    """
    Test the compression ratio of the randomized SVD.

    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param k: Rank of the matrix.
    :param max_m: Max of testing value of m
    :param m_interval: Interval of each testing m
    :return: None
    """

    import concurrent.futures # enhance the speed

    def process_k(k):
        A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)
        X, Y, T, Vs = [], [], [], []
        for m in range(k, max_m + 1, m_interval):
            dt, avg, V_diff = error_avg(A, m, k, Sigma, V)
            X.append(m)
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

    # graphs: (i, j, title, xlabel, ylabel)
    plot_graphs(results, graphs = (
        (0, 1, 
         f"Error v.s. Compressed Row Number (N = {N}, n = {n})", 
         "m", 
         "Average of Absolute Error"), 
        (0, 3, 
         f"V difference v.s. Compressed Row Number (N = {N}, n = {n})", 
         "m", 
         "V difference (norm(V-V'))")
    ))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_compression_ratio()

