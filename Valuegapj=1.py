import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

# --- 計算論文中理論上界 ---
def compute_theoretical_bound(sigma_vals, j, epsilon):
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

# --- 參數設定 ---
N, n, k = 500, 20, 10
m = 40          # 故意用小的 m，加強壓縮誤差
j = 1           # 比較第 j 個奇異向量
epsilon = 0.1
gaps = np.linspace(0.01, 3.0, 30)

vector_errors = []
theoretical_bounds = []

# --- 主實驗迴圈 ---
for gap in gaps:
    # 生成奇異值（gap 越小越相近）
    s1 = 10
    singular_vals = [s1 - i * gap for i in range(k)]
    singular_vals = np.maximum(singular_vals, 0.1)  # 防止為負
    S = np.diag(singular_vals)

    # 建 rank-k 矩陣 X
    U = np.linalg.qr(np.random.randn(N, k))[0]
    V = np.linalg.qr(np.random.randn(n, k))[0]
    X = U @ S @ V.T

    # 壓縮
    Phi = np.random.randn(m, N) / np.sqrt(m)
    Y = Phi @ X

    # SVD
    _, _, V_X = svd(X, full_matrices=False)
    _, _, V_Y = svd(Y, full_matrices=False)

    # 第 j 個向量誤差
    v = V_X[j]
    v_p = V_Y[j]
    if np.dot(v, v_p) < 0:
        v_p = -v_p
    err = norm(v - v_p)
    vector_errors.append(err)

    # 理論上界
    bound = compute_theoretical_bound(singular_vals, j, epsilon)
    theoretical_bounds.append(bound)

# --- 繪圖 ---
plt.figure(figsize=(10, 5))
plt.plot(gaps, vector_errors, 'bo-', label='Actual error ‖v_j - v′_j‖₂')
plt.plot(gaps, theoretical_bounds, 'r--', label='Theoretical upper bound')
plt.xlabel('Singular value gap')
plt.ylabel('Error')
plt.title(f'Singular vector error vs. gap (m = {m}, j = {j})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()














