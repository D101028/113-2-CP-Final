"""
測試不同 gap (singular value 之間的間隔)
對於第 j 項 right singular vector 的影響。
"""

import numpy as np
from numpy.linalg import norm

from func import generate_matrix_with_singular_values, sketched_svd, draw_diagram2

def compute_theoretical_bound(sigma_vals, j, epsilon):
    """Compute the theoretical bound"""
    sigma_j = sigma_vals[j]
    numerator = epsilon * np.sqrt(1 + epsilon) / np.sqrt(1 - epsilon)
    max_term = 0

    for i in range(len(sigma_vals)):
        if i == j:
            continue
        sigma_i = sigma_vals[i]
        denom_candidates = [
            abs(sigma_i**2 - sigma_j**2 * (1 + c * epsilon)) for c in [-1, 0, 1]
        ]
        denom = min(denom_candidates)
        if denom != 0:
            term = np.sqrt(2) * sigma_i * sigma_j / denom
            max_term = max(max_term, term)

    bound = numerator * max_term
    return min(np.sqrt(2), bound)

def test_value_gap_j(N = 500, n = 20, k = 10, m = 40, j = 0, 
                      epsilon = 0.1, gaps = np.linspace(0.01, 3.0, 30)):
    """
    N, n, k = 500, 20, 10
    m = 40          # Set m
    j = 0           # We only consider the first right singular vector
    epsilon = 0.1   # Error
    gaps = np.linspace(0.01, 3.0, 30) # Gap
    """

    vector_errors = []
    theoretical_bounds = []

    # Select error and the theoretical bound
    for gap in gaps:
        # Generate singular values
        s1 = 10
        singular_vals = [s1 - i * gap for i in range(k)]
        singular_vals = np.maximum(singular_vals, 0.1)  # Here we require singular to be nonnegative

        # Generate the experiment data
        X, _, _, V_X = generate_matrix_with_singular_values(N, n, sigma=singular_vals)
        _, V_Y = sketched_svd(X, m)

        # The error of the j-th vector 
        v = V_X[:, j]
        v_p = V_Y[:, j]

        # Ensure the nonnegative inner product
        if np.dot(v, v_p) < 0:
            v_p = -v_p
        err = norm(v - v_p)
        vector_errors.append(err)

        # Thereotical bound
        bound = compute_theoretical_bound(singular_vals, j, epsilon)
        theoretical_bounds.append(bound)

    draw_diagram2(
        gaps, 
        (vector_errors, 'bo-', 'Actual error ‖v_j - v′_j‖₂'), 
        (theoretical_bounds, 'r--', 'Theoretical upper bound'), 
        xlabel  = 'Singular value gap', 
        ylabel  = 'Error(Euclidean norm)', 
        title   =  f'Singular vector error vs. gap (m = {m}, j = {j})', 
        figsize = (14, 5)
    )

if __name__ == "__main__":
    test_value_gap_j()
