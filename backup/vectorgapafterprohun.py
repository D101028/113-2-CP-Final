import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm
from scipy.optimize import linear_sum_assignment

# 參數設定
N, n, k = 500, 20, 10
m = 40
epsilon = 0.1
gap_list = np.linspace(0.01, 2.0, 30)
j_list = [0, 1, 2, 3, 4]  # 要比較的 singular vector indices

# 初始化儲存
errors_by_j = {j: [] for j in j_list}
bounds_by_j = {j: [] for j in j_list}

for gap in gap_list:
    # 建立 rank-k 矩陣 X
    s_vals = np.linspace(10, 10 - gap*(k-1), k)
    s_vals = np.maximum(s_vals, 0.1)
    S = np.diag(s_vals)

    U = np.linalg.qr(np.random.randn(N, k))[0]
    V = np.linalg.qr(np.random.randn(n, k))[0]
    X = U @ S @ V.T

    # 壓縮
    Phi = np.random.randn(m, N) / np.sqrt(m)
    Y = Phi @ X

    # SVD
    _, S_X, V_X = svd(X, full_matrices=False)
    _, _, V_Y = svd(Y, full_matrices=False)
    V_X_k = V_X[:k]
    V_Y_k = V_Y[:k]

    # Procrustes 對齊
    U_p, _, VT_p = svd(V_X_k @ V_Y_k.T)
    Q = U_p @ VT_p
    V_Y_k_aligned = (V_Y_k.T @ Q).T

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
        sigma_j = S_X[j]
        numerator = epsilon * np.sqrt(1 + epsilon) / np.sqrt(1 - epsilon)
        max_term = 0
        for l in range(k):
            if l == j:
                continue
            sigma_i = S_X[l]
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