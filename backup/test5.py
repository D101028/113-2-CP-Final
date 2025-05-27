import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds

# 初始化參數
n = 10         # 節點數
m = 4          # Sketch 壓縮維度
steps = 150     # 模擬加入邊的次數

# 初始化鄰接矩陣與 sketch 矩陣
A = np.zeros((n, n))
S = np.random.randn(m, n) / np.sqrt(m)

# 儲存每一步的奇異值（True SVD 與 Sketch SVD）
true_singular_values = []
sketch_singular_values = []

def true_svd(A):
    """
    直接對完整鄰接矩陣 A 做 SVD，回傳前幾大的奇異值
    """
    u, s, vt = svds(A, k=min(3, n - 1))
    return s[::-1]

def sketch_svd(A, S):
    """
    對 Sketch 矩陣 SA 做 SVD，回傳前幾大的奇異值（近似值）
    """
    SA = S @ A
    u, s, vt = svds(SA, k=min(3, m, n - 1))
    return s[::-1]

# 模擬串流加入邊
for step in range(steps):
    i, j = np.random.choice(n, 2, replace=False)
    A[i, j] = A[j, i] = 1  # 無向圖

    true_sv = true_svd(A)
    sketch_sv = sketch_svd(A, S)

    true_singular_values.append(true_sv)
    sketch_singular_values.append(sketch_sv)

# 畫圖：Sketch SVD
plt.figure(figsize=(10, 5))
for k in range(len(sketch_singular_values[0])):
    plt.plot([sv[k] for sv in sketch_singular_values], label=f"Sketch Top-{k+1} σ")
plt.title("Approximate Spectral Features via Sketch SVD")
plt.xlabel("Stream Step (new edge added)")
plt.ylabel("Singular Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 畫圖：True SVD
plt.figure(figsize=(10, 5))
for k in range(len(true_singular_values[0])):
    plt.plot([sv[k] for sv in true_singular_values], label=f"True Top-{k+1} σ")
plt.title("True Spectral Features via Full SVD")
plt.xlabel("Stream Step (new edge added)")
plt.ylabel("Singular Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
