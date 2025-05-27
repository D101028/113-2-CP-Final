"""
分別測試：
1. 給定 ε (理論誤差), δ (失敗機會)，計算所需的 m (壓縮後維度)，並繪製 sigular values 誤差與理論上下界。
2. 給定 ε (理論誤差), m (壓縮後維度)，繪製 sigular values 誤差與理論上下界。
"""

import numpy as np

from func import generate_matrix_with_singular_values, sketched_svd, m_func, draw_diagram3

def compute_ratio_and_bounds(N, n, k, m, epsilon):
    # Generate the experiment data
    X, true_singular_values, _, _ = generate_matrix_with_singular_values(N, n, sigma=np.linspace(10, 1, k))
    sketch_singular_values = sketched_svd(X, m, False)

    # Take the first k terms
    sigma_X = true_singular_values[:k]
    sigma_Y = sketch_singular_values[:k]

    # Compute the ratio with the corresponding bound
    ratios = sigma_Y / sigma_X
    lower_bound = np.sqrt(1 - epsilon)
    upper_bound = np.sqrt(1 + epsilon)

    return ratios, lower_bound, upper_bound

def test_value_ratio_JL(N = 500, n = 20, k = 10, epsilon = 0.1, delta = 0.01):
    """
    Compute results from given epsilon and delta. 
    m is computed from epsilon and delta. 
    >>> N, n = 500, 20     # Dimension of X
    >>> k = 10             # Rank 
    >>> epsilon = 0.1      # Error
    >>> delta = 0.01       # Failure probability
    """
    # Compute m
    m = m_func(k, epsilon, delta)

    ratios, lower_bound, upper_bound = compute_ratio_and_bounds(N, n, k, m, epsilon)

    # Draw the diagram
    draw_diagram3(range(1, k+1), 
        (ratios, 'σ_Y / σ_X'), 
        (lower_bound, 'Theorem 1 lower bound'), 
        (upper_bound, 'Theorem 1 upper bound'), 
        xlabel  = "Singular Value Index (j)", 
        ylabel  = "σ_j(Y) / σ_j(X)", 
        title   = "Ratio of Sketched vs True Singular Values"
    )

def test_value_ratio_n_JL(N = 500, n = 20, k = 10, epsilon = 0.1, m = 40):
    """
    Compute results from given epsilon and m
    >>> N, n = 500, 20      # Dimension of X 
    >>> k = 10              # Rank
    >>> epsilon = 0.1       # Error
    >>> m = 40              # Compressed rows number
    """
    ratios, lower_bound, upper_bound = compute_ratio_and_bounds(N, n, k, m, epsilon)

    # Draw the diagram
    draw_diagram3(
        range(1, k+1), 
        (ratios, 'σ_Y / σ_X'), 
        (lower_bound, 'Theorem 1 lower bound'), 
        (upper_bound, 'Theorem 1 upper bound'), 
        xlabel  = "Singular Value Index (j)", 
        ylabel  = "σ_j(Y) / σ_j(X)", 
        title   = "Ratio of Sketched vs True Singular Values"
    )

if __name__ == "__main__":
    test_value_ratio_JL()
