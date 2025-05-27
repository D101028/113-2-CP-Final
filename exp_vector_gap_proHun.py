"""
測試在不同 singular value gap 下，經過 Procrustes 對齊與匈牙利匹配後，
奇異向量的誤差與理論上界的關係。
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

from func import generate_matrix_with_singular_values, sketched_svd, procrustes_align

def test_vector_gap_after_proHun(N = 500, n = 20, k = 10, m = 40, epsilon = 0.1, 
                                 gap_list = np.linspace(0.01, 2.0, 30), j_list = [0, 1, 2, 3, 4]):
    """
    # 參數設定
    N, n, k = 500, 20, 10
    m = 40
    epsilon = 0.1
    gap_list = np.linspace(0.01, 2.0, 30)
    j_list = [0, 1, 2, 3, 4]  # 要比較的 singular vector indices
    """
    # 初始化儲存
    errors_by_j = {j: [] for j in j_list}
    bounds_by_j = {j: [] for j in j_list}

    for gap in gap_list:
        # 建立 rank-k 矩陣 X
        s_vals = np.linspace(10, 10 - gap*(k-1), k)
        s_vals = np.maximum(s_vals, 0.1)
        S = np.diag(s_vals)
        X, S, _, V_X = generate_matrix_with_singular_values(N, n, k, sigma=s_vals)

        # Sketeched SVD
        _, V_Y = sketched_svd(X, m)

        # SVD
        V_X_k = V_X[:, :k]
        V_Y_k = V_Y[:, :k]

        # Procrustes 對齊
        V_Y_k_aligned = procrustes_align(V_X_k.T, V_Y_k.T).T  # Align

        # 匈牙利匹配
        similarity = np.abs(V_X_k @ V_Y_k_aligned.T)
        cost = 1 - similarity
        _, col_ind = linear_sum_assignment(cost)

        for j in j_list:
            # 誤差計算
            v = V_X[j]
            v_p = V_Y_k_aligned[col_ind[j]]
            if np.dot(v, v_p) < 0:
                v_p = -v_p
            errors_by_j[j].append(norm(v - v_p))

            # 理論上界
            sigma_j = S[j]
            numerator = epsilon * np.sqrt(1 + epsilon) / np.sqrt(1 - epsilon)
            max_term = 0
            for l in range(k):
                if l == j:
                    continue
                sigma_i = S[l]
                denom_candidates = [abs(sigma_i**2 - sigma_j**2 * (1 + c*epsilon)) for c in [-1, 0, 1]]
                denom = min(denom_candidates)
                if denom != 0:
                    ratio = np.sqrt(2) * sigma_i * sigma_j / denom
                    max_term = max(max_term, ratio)
            bounds_by_j[j].append(min(np.sqrt(2), numerator * max_term))

    # 畫圖
    plt.figure(figsize=(10, 6))
    for j in j_list:
        plt.plot(gap_list, errors_by_j[j], label=f'Error of v{j}')
        plt.plot(gap_list, bounds_by_j[j], '--', label=f'Bound for v{j}')
        
    plt.xlabel('Singular value gap')
    plt.ylabel('L2 error')
    plt.title('Singular vector error vs. gap (Procrustes + matching)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()