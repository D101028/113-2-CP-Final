"""
測試不同的 m (壓縮後的 rank 大小) 
對於 sketched svd 的 Singular values 
與 Right sigular vectors 的影響。
"""

import concurrent.futures # enhance the speed
import time

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

from func import generate_matrix_with_singular_values, sketched_svd

def error_avg(A, m, k, sigma, V1):
    """
    :param A: Input matrix
    :param m: Compressed rank
    :param sigma: Original sigular values
    :param V1: Original right sigular vectors
    :return: dt, avg, V_diff
    """
    t = time.time()
    sketched_sigma, sketched_V = sketched_svd(A, m)
    dt = time.time() - t
    s = sigma[:k] ** 2
    s1 = sketched_sigma[:k] ** 2
    avg = np.average(np.abs((s1 - s) / s))
    # Calculate the difference between V and V1
    # Vectors in sketched_V may be different by a sign
    V_diff = np.average([
        min((
                norm(V1[:, i] - sketched_V[:, i]), 
                norm(V1[:, i] + sketched_V[:, i])
            )) for i in range(k)
    ])
    print(f"{m} {dt:.4f} {avg:.4f} {V_diff:.4f}")
    return dt, avg, V_diff

def plot_graphs(results, graphs):
    """
    Plot the graphs by the given datas. 
    Graphs will be plotted in different windows. 

    :param results: datas, format: `( (k_value, data_x, data_y), (index, ...), ... )`
    For example 
        >>> ( 
            (20, [1, 2, 3], [4, 5, 6]), 
            (40, [5, 4, 3], [2, 1, 0]) 
        )
        
    :param graphs: graphs msg, format: `( (index, code_x, code_y, title, xlabel, ylabel) )`
    For example
        >>> (
            (0, 1, 0, "title", "xlabel", "ylabel"), 
            (1, 0, 2, "title", "xlabel", "ylabel")
        )

        The example above means to plot graph with code 0 by 1 and 2 by 0. 
    """
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
    plt.tight_layout()
    plt.show()

def plot_graphs_together(results, graphs):
    """
    Plot the graphs by the given datas. 
    Graphs will be plotted in the same window. 

    :param results: datas, format: `( (k_value, data_x, data_y), (index, ...), ... )`
    For example 
        >>> ( 
            (20, [1, 2, 3], [4, 5, 6]), 
            (40, [5, 4, 3], [2, 1, 0]) 
        )
        
    :param graphs: graphs msg, format: `( (index, code_x, code_y, title, xlabel, ylabel) )`
    For example
        >>> (
            (0, 1, 0, "title", "xlabel", "ylabel"), 
            (1, 0, 2, "title", "xlabel", "ylabel")
        )

        The example above means to plot graph with code 0 by 1 and 2 by 0. 
    """
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
    plt.tight_layout()
    plt.show()

def test_compression_ratio(N = 1000, n = 400, k_arr = [5, 10, 50, 100], max_m = 500, m_interval = 5):
    """
    Test the compression ratio of the randomized SVD.

    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param k: Rank of the matrix.
    :param max_m: Max of testing value of m
    :param m_interval: Interval of each testing m
    :return: None
    """

    def process_k(k):
        A, sigma, U, V = generate_matrix_with_singular_values(N, n, k)
        X, Y, T, Vs = [], [], [], []
        for m in range(k, min(max_m, N) + 1, m_interval):
            dt, avg, V_diff = error_avg(A, m, k, sigma, V)
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

    # graphs: (i, j, title, xlabel, ylabel), the graph of j-th variable by i-th variable
    plot_graphs(results, graphs = (
        (0, 1, 
         f"Error by Compressed Row Number (N = {N}, n = {n})", 
         "m", 
         "Average of Absolute Error"), 
        (0, 2, 
         f"Time by Compressed Row Number (N = {N}, n = {n})", 
         "m", 
         "Time Comsume (s)"), 
        (0, 3, 
         f"V difference by Compressed Row Number (N = {N}, n = {n})", 
         "m", 
         "V difference (norm(V-V'))")
    ))

if __name__ == "__main__":
    test_compression_ratio()

