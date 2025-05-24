import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Procrustes 
def procrustes_align(V_ref, V_target):
    U, _, VT = svd(V_ref.T @ V_target)
    Q = U @ VT
    return V_target @ Q  

# Variable setting
N, n, k = 500, 20, 10
m = 40

# Construct X with rank(X)=k
U = np.linalg.qr(np.random.randn(N, k))[0]
V = np.linalg.qr(np.random.randn(n, k))[0]
S = np.diag(np.linspace(5, 1, k))  # Singular values with small gap
X = U @ S @ V.T

# Select Phi satisfies JL
Phi = np.random.randn(m, N) / np.sqrt(m)

# Compute Y
Y = Phi @ X

# SVD
_, _, V_X_T = svd(X, full_matrices=False)
_, _, V_Y_T = svd(Y, full_matrices=False)

V_X = V_X_T.T
V_Y = V_Y_T.T

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

# Draw the diagram
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
