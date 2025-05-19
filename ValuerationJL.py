import numpy as np
import matplotlib.pyplot as plt

# 參數設定
n, d = 500, 20    # 原始矩陣維度 (n >> d)
k = 10             # 欲近似的秩
epsilon = 0.1      # 誤差容忍度
m=40

# 建立 rank-k 的資料矩陣 X = U Σ V^T
U = np.linalg.qr(np.random.randn(n, k))[0]        # N x k，正交列
V = np.linalg.qr(np.random.randn(d, k))[0]        # n x k，正交列
S = np.diag(np.linspace(10, 1, k))                # k 個非零奇異值

X = U @ S @ V.T

# Sketched matrix Y = Φ X, 使用 Gaussian JL matrix
Phi = np.random.randn(m, n) / np.sqrt(m)
Y = Phi @ X

# 計算奇異值
true_singular_values = np.linalg.svd(X, compute_uv=False)
sketch_singular_values = np.linalg.svd(Y, compute_uv=False)

# 取前 k 個奇異值
σ_X = true_singular_values[:k]
σ_Y = sketch_singular_values[:k]

# 計算比例與誤差
ratios = σ_Y / σ_X
lower_bound = np.sqrt(1 - epsilon)
upper_bound = np.sqrt(1 + epsilon)

# 顯示結果
#print("\nRatio of σ_j(Y) / σ_j(X):")
#for j, r in enumerate(ratios):
#    status = "✓" if lower_bound <= r <= upper_bound else "✗"
#    print(f"  σ_{j+1}: {r:.4f} ({status})")

# 繪圖比較
plt.figure(figsize=(8, 5))
plt.plot(range(1, k+1), ratios, marker='o', label='σ_Y / σ_X')
plt.axhline(y=lower_bound, color='r', linestyle='--', label='Theorem 1 lower bound')
plt.axhline(y=upper_bound, color='g', linestyle='--', label='Theorem 1 upper bound')
plt.title("Ratio of Sketched vs True Singular Values")
plt.xlabel("Singular Value Index (j)")
plt.ylabel("σ_j(Y) / σ_j(X)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()