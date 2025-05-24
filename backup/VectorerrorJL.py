import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm

# Variable setting
N, n = 500, 20     # Dimension of X
k = 10             # rank
epsilon = 0.1      # Error
delta = 0.01       # Failure probability
m = int(np.ceil((6 / (epsilon ** 2)) * (k * np.log(42 / epsilon) + np.log(2 / delta))))     # Compute m


# Construct X with rank(X)=k 
U = np.linalg.qr(np.random.randn(N, k))[0]        # N x k, orthogonal
V = np.linalg.qr(np.random.randn(n, k))[0]        # n x k, orthogonal
S = np.diag(np.linspace(10, 1, k))                # k nonzero singular values

X = U @ S @ V.T

# Select Phi satisfies JL
Phi = np.random.randn(m, N) / np.sqrt(m)

# Compute Y
Y = Phi @ X  
# SVD
_, S_X, V_X_T = svd(X, full_matrices=False)  # V_X: n x n
_, S_Y, V_Y_T = svd(Y, full_matrices=False)

V_X = V_X_T.T
V_Y = V_Y_T.T

# Error(L2) and thereotical bound
errors = []
bounds = []

for j in range(k):
    v = V_X[:, j]
    v_p = V_Y[:, j]
    
    # Ensure the nonnegative inner product
    if np.dot(v, v_p) < 0:
        v_p = -v_p

    err = norm(v - v_p)
    errors.append(err)

    # Thereotical bound
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

# Draw the diagram
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
