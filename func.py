import numpy as np
from numpy.linalg import svd
import time

def remove_small_values(matrix, threshold=1e-8):
    return np.where(np.abs(matrix) < threshold, 0, matrix)

def generate_JL_matrix(m, n):
    """
    Generate a Johnson-Lindenstrauss (JL) matrix.
    :param m: Number of rows in the JL matrix.
    :param n: Number of columns in the JL matrix.
    :return: A random JL matrix of shape (m, n).
    """
    Phi = np.random.randn(m, n) / np.sqrt(m)
    return Phi

def sketched_svd(X, m):
    """
    Perform a randomized SVD using a JL matrix.
    :param X: Input matrix of shape (N, n).
    :param m: Number of rows in the JL matrix.
    :return: Singular values and right singular vectors of the compressed matrix.
    """
    N, n = X.shape
    Phi = generate_JL_matrix(m, N)  # Generate JL matrix
    Y = Phi @ X
    U_Y, Sigma_Y, V_Y_T = svd(Y, full_matrices=True)
    return Sigma_Y, V_Y_T.T  # Return Σ_Y and V_Y

def generate_matrix_with_singular_values(m, n, rank, ranging = None, sigma = None):
    """
    Generate a random matrix with specified singular values.
    :param m: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param rank: Rank of the matrix.
    :param ranging: Range for generating singular values. If None, random singular values are generated.
    :param sigma: List of singular values. If None, random singular values are generated.
    :return: A, Sigma, U, V.
    """

    # 隨機生成奇異值 sigma
    if sigma is None:
        # Define the range for singular values
        if ranging is None:
            low, upper = 1, 500
        else:
            low, upper = ranging
        a = list(np.random.uniform(low, upper, rank))
        a += [0]*(min(m, n) - rank)
        sigma = np.sort(a)[::-1]
    else:
        sigma = np.sort(sigma)[::-1]

    # 構造對角矩陣 Sigma
    Sigma = np.zeros((m, n))
    for i in range(len(sigma)):
        Sigma[i, i] = sigma[i]

    # 隨機生成正交矩陣 U 和 V
    U, _ = np.linalg.qr(np.random.randn(m, m))
    V, _ = np.linalg.qr(np.random.randn(n, n))

    # 計算 A = U Σ V^T
    A = U @ Sigma @ V.T

    return A, sigma, U, V

def timer(func, *args, **kwargs):
    """
    Timer decorator to measure the execution time of a function.
    :param func: Function to be timed.
    :param args: Positional arguments for the function.
    :param kwargs: Keyword arguments for the function.
    :return: Execution time in seconds.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time, result

def title_print(title):
    """
    Print a title with a separator line.
    :param title: Title to be printed.
    """
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40 + "\n")
