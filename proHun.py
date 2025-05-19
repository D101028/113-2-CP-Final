import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# --- 對齊函數 ---
def procrustes_align(V_ref, V_target):
    U, _, VT = svd(V_ref.T @ V_target)
    Q = U @ VT
    return V_target @ Q  # 對齊後的 V_target（列為向量）

# --- 參數 ---
N, n, k = 500, 20, 10
m = 40

# 建立 rank-k 矩陣 X
U = np.linalg.qr(np.random.randn(N, k))[0]
V = np.linalg.qr(np.random.randn(n, k))[0]
S = np.diag(np.linspace(5, 1, k))  # 奇異值下降但 gap 小
X = U @ S @ V.T

# 壓縮
Phi = np.random.randn(m, N) / np.sqrt(m)
Y = Phi @ X

# SVD
_, _, V_X = svd(X, full_matrices=False)
_, _, V_Y = svd(Y, full_matrices=False)

# 取前 k 個右奇異向量
V_X_k = V_X[:k]         # shape (k, n)
V_Y_k = V_Y[:k]
V_Y_k_aligned = procrustes_align(V_X_k.T, V_Y_k.T).T  # 對齊後回到 shape (k, n)

# --- 找最佳匹配（基於 cosine distance） ---
# similarity_matrix[i, j] = |cos(θ)| between v_i and aligned_v_j
similarity = np.abs(V_X_k @ V_Y_k_aligned.T)
cost = 1 - similarity  # 因為我們要找最小的 cost

# 使用匈牙利算法找最小總距離對應
row_ind, col_ind = linear_sum_assignment(cost)

# --- 計算誤差 ---
errors_before = []
errors_after = []

for i in range(k):
    vi = V_X_k[i]
    vi_sketch = V_Y_k[i]
    vi_aligned = V_Y_k_aligned[col_ind[i]]  # 對應最佳匹配後的向量

    # 對齊方向
    if np.dot(vi, vi_sketch) < 0:
        vi_sketch = -vi_sketch
    if np.dot(vi, vi_aligned) < 0:
        vi_aligned = -vi_aligned

    errors_before.append(norm(vi - vi_sketch))
    errors_after.append(norm(vi - vi_aligned))

# --- 畫圖 ---
x = np.arange(k)
plt.figure(figsize=(9, 5))
plt.plot(x, errors_before, 'ro-', label='Before Procrustes')
plt.plot(x, errors_after, 'go--', label='After Procrustes (best matched)')
plt.xlabel('Singular vector index (true)')
plt.ylabel('‖v_j - v′_j‖₂')
plt.title(f'Singular vector errors before/after Procrustes (matched) (m = {m})')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()