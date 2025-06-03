"""
測試 Procrustes 對齊前後的 right sigular vectors 誤差。
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

from func import generate_matrix_with_singular_values, sketched_svd, procrustes_align, draw_diagram2

def test_proHun(N = 500, n = 20, k = 10, m = 40):
    # Generate the experiment data
    sigma = np.linspace(5, 1, k) # Singular values with small gap
    X, _, _, V_X = generate_matrix_with_singular_values(N, n, sigma=sigma)
    _, V_Y = sketched_svd(X, m)
    
    # Take the first k terms
    V_X_k = V_X[:, :k]
    V_Y_k = V_Y[:, :k]
    V_Y_k_aligned = procrustes_align(V_X_k.T, V_Y_k.T).T  # Align

    # Find the best match
    # We consider the similarity_matrix[i, j] = |cos(θ)| between v_i and aligned_v_j
    similarity = np.abs(V_X_k @ V_Y_k_aligned.T)
    cost = 1 - similarity  # Find the minimum cost
    # Hungarian
    row_ind, col_ind = linear_sum_assignment(cost)

    # Compute the error
    errors_before = []
    errors_after = []

    for i in range(k):
        vi = V_X_k[i]
        vi_sketch = V_Y_k[i]
        vi_aligned = V_Y_k_aligned[col_ind[i]]  

        # Ensure the nonnegative inner product
        if np.dot(vi, vi_sketch) < 0:
            vi_sketch = -vi_sketch
        if np.dot(vi, vi_aligned) < 0:
            vi_aligned = -vi_aligned

        errors_before.append(norm(vi - vi_sketch))
        errors_after.append(norm(vi - vi_aligned))
    
    # Draw diagram
    x = np.arange(k)
    draw_diagram2(
        x, 
        (errors_before, 'ro-', 'Before Procrustes'), 
        (errors_after, 'go--', 'After Procrustes (best matched)'), 
        xlabel  = 'Singular vector index (original)', 
        ylabel  = 'error(Euclidean norm)', 
        title   = f'Singular vector errors before/after Procrustes (matched) (m = {m})', 
    )

if __name__ == "__main__":
    test_proHun()
