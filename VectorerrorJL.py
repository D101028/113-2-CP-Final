import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm

# --- Step 1: 模擬參數 ---
N, n = 500, 20   # X 為 N x n
k = 10            # True rank
epsilon = 0.1   # Distortion factor
delta=0.01
m = int(np.ceil((6 / (epsilon ** 2)) * (k * np.log(42 / epsilon) + np.log(2 / delta))))     # Compressed rows


# --- Step 2: 產生 rank-k 的矩陣 X ---
U = np.linalg.qr(np.random.randn(N, k))[0]        # N x k，正交列
V = np.linalg.qr(np.random.randn(n, k))[0]        # n x k，正交列
S = np.diag(np.linspace(10, 1, k))                # k 個非零奇異值

X = U @ S @ V.T

# --- Step 3: 隨機投影 Φ ---
Phi = np.random.randn(m, N) / np.sqrt(m)
Y = Phi @ X  # m x n

# --- Step 4: SVD ---
_, S_X, V_X = svd(X, full_matrices=False)  # V_X: n x n
_, S_Y, V_Y = svd(Y, full_matrices=False)

# --- Step 5: 比較誤差與理論上界 ---
errors = []
bounds = []

for j in range(k):
    v = V_X[j]
    v_p = V_Y[j]
    
    # 對齊符號
    if np.dot(v, v_p) < 0:
        v_p = -v_p

    err = norm(v - v_p)
    errors.append(err)

    # 計算理論上界
    sigma_j = S_X[j]
    numerator = epsilon * np.sqrt(1 + epsilon) / np.sqrt(1 - epsilon)
    max_term = 0

    for i in range(k):
        if i == j:
            continue
        sigma_i = S_X[i]
        denom_candidates = [abs(sigma_i**2 - sigma_j**2 * (1 + c * epsilon)) for c in [-1, 0, 1]]
        denom = min(denom_candidates)
        if denom != 0:
            ratio = np.sqrt(2) * sigma_i * sigma_j / denom
            max_term = max(max_term, ratio)

    bound = min(np.sqrt(2), numerator * max_term)
    bounds.append(bound)

# --- Step 6: 繪圖 ---
plt.figure(figsize=(8, 5))
plt.plot(errors, 'bo-', label='Error')
plt.plot(bounds, 'r--', label='upper bound')
plt.xlabel('j-th right singular vector')
plt.ylabel('error(L2 norm)')
plt.title('The estimate for right singular vector')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
