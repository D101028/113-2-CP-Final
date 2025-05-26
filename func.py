import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd

def remove_small_values(matrix, threshold=1e-8):
    return np.where(np.abs(matrix) < threshold, 0, matrix)

def generate_JL_matrix(m, n):
    """
    Generate a distributionsal Johnson-Lindenstrauss (JL) matrix.
    (We consider the Gaussian Distribution Matrix)
    
    :param m: Number of rows in the JL matrix.
    :param n: Number of columns in the JL matrix.
    :return: A random JL matrix of shape (m, n).
    """
    Phi = np.random.randn(m, n) / np.sqrt(m)
    return Phi

def sketched_svd(X, m, compute_uv = True):
    """
    Perform a randomized SVD using a JL matrix.

    :param X: Input matrix of shape (N, n).
    :param m: Number of rows in the JL matrix.
    :return: sigma_Y, V_Y: 
    Singular values and right singular vectors of the compressed matrix.
    """
    N, n = X.shape
    Phi = generate_JL_matrix(m, N)  # Generate JL matrix
    Y = Phi @ X
    if compute_uv:
        U_Y, sigma_Y, V_Y_T = svd(Y, compute_uv=True)
        return sigma_Y, V_Y_T.T  # Return Σ_Y and V_Y
    else:
        sigma_Y = svd(Y, compute_uv=False)
        return sigma_Y

def generate_matrix_with_singular_values(N, n, rank = None, ranging = None, sigma = None):
    """
    Generate a random matrix with specified singular values.

    :param N: Number of rows in the matrix.
    :param n: Number of columns in the matrix.
    :param rank: Rank of the matrix. If None, set as full rank (max(m, n)). 
    :param ranging: Range for generating singular values. If None, random singular values are generated.
    :param sigma: List of singular values. 
        If None, random singular values are generated. 
        If not None, the `rank` and `ranging` parameters will be ignored.
    :return: A, sigma, U, V.
    """

    # Set sigular values sigma 
    if sigma is None:
        if rank is None:
            # Set the rank to be the full rank
            rank = min(N, n)
        # Define the range for singular values
        if ranging is None:
            lower, upper = 1, 500
        else:
            lower, upper = ranging
        a = np.random.uniform(lower, upper, rank)
        # a += [0]*(min(N, n) - rank)
        sigma = np.sort(a)[::-1]
    else:
        sigma = np.sort(sigma)[::-1]
        # Count the rank and cast 0's in sigma
        rank = len(sigma)
        for i in reversed(sigma):
            if i == 0:
                rank -= 1
                continue
            if i != 0:
                break
        sigma = sigma[:rank]

    # Construct the diagonal matrix Sigma
    Sigma = np.diag(sigma)

    # Randomly generate orthogonal matrix U, V
    U, _ = np.linalg.qr(np.random.randn(N, rank))
    V, _ = np.linalg.qr(np.random.randn(n, rank))

    # Compute A = U Σ V^T
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

def m_func(k, epsilon, delta):
    return int(np.ceil((6 / (epsilon ** 2)) * (k * np.log(42 / epsilon) + np.log(2 / delta))))

def draw_diagram2(x, y1_info, y2_info, xlabel = "", ylabel = "", title = "", figsize = (9, 5)):
    """
    Plot a graph with two sets of data: y1, y2.
    
    :param x: data set x
    :param y1_info: a tuple of (data set, style, label) of y1
    :param y2_info: a tuple of (data set, style, label) of y2
    :param xlabel: graph xlabel, default: ""
    :param ylabel: graph ylabel, default: ""
    :param title: graph title, default: (9, 5)
    :return: None
    """
    y1_data, y1_style, y1_label = y1_info
    y2_data, y2_style, y2_label = y2_info

    plt.figure(figsize=figsize)
    plt.plot(x, y1_data, y1_style, label=y1_label)
    plt.plot(x, y2_data, y2_style, label=y2_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def draw_diagram3(x, y_info, l1_info, l2_info, xlabel = "", ylabel = "", title = "", figsize = (8, 5)):
    """
    Plot a graph with data: y, l1 (lower bound), l2 (upper bound).
    
    :param x: data set x
    :param y_info: a tuple of (data set, label) of y
    :param l1_info: a tuple of (data set, label) of l1
    :param l2_info: a tuple of (data set, label) of l2
    :param xlabel: graph xlabel, default: ""
    :param ylabel: graph ylabel, default: ""
    :param title: graph title, default: (9, 5)
    :return: None
    """
    y_data, y_label = y_info
    l1_data, l1_label = l1_info
    l2_data, l2_label = l2_info

    plt.figure(figsize=figsize)
    plt.plot(x, y_data, marker='o', label=y_label)
    plt.axhline(y=l1_data, color='r', linestyle='--', label=l1_label)
    plt.axhline(y=l2_data, color='g', linestyle='--', label=l2_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
