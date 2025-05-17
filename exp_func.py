import time

import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

from func import generate_matrix_with_singular_values, sketched_svd

def error_avg(A, m, k, Sigma, V1):
    """
    :return: dt, avg, V_diff
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
    return dt, avg, V_diff

# def plot_graphs(results, N, n):
#     num = len(results)
#     for i in range(num):
#         k, X, Y, T, Vs = results[i]
#         plt.figure(i)
#         plt.plot(X, Y, label=f"k = {k}")

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

def test_compression_ratio(N = 1000, n = 800, k_arr = [5, 10, 50, 100]):
    """
    Test the compression ratio of the randomized SVD.

    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param k: Rank of the matrix.
    :return: None
    """

    import concurrent.futures # enhance the speed

    def process_k(k):
        A, Sigma, U, V = generate_matrix_with_singular_values(N, n, k)
        X, Y, T, Vs = [], [], [], []
        for m in range(k, n // 2, 5):
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

    plot_graphs_together(results, graphs = (
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

