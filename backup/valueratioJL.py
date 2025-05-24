import numpy as np
import matplotlib.pyplot as plt

# Variable setting
N, n = 500, 20     # Dimension of X
k = 10             # Rank 
epsilon = 0.1      # Error
delta = 0.01       # Failure probability

# Compute m
m = int(np.ceil((6 / (epsilon ** 2)) * (k * np.log(42 / epsilon) + np.log(2 / delta))))

# Construct X with rank(X)=k
U = np.linalg.qr(np.random.randn(N, k))[0]        # N x k, orthogonal
V = np.linalg.qr(np.random.randn(n, k))[0]        # n x k, orthogonal
S = np.diag(np.linspace(10, 1, k))                # k nonzero singular values

X = U @ S @ V.T

# Select Phi satisfies JL
Phi = np.random.randn(m, N) / np.sqrt(m)

# Compute Y
Y = Phi @ X

# Singular values
true_singular_values = np.linalg.svd(X, compute_uv=False)
sketch_singular_values = np.linalg.svd(Y, compute_uv=False)

# Take the first k terms
σ_X = true_singular_values[:k]
σ_Y = sketch_singular_values[:k]

# Compute the ratio with the corresponding bound
ratios = σ_Y / σ_X
lower_bound = np.sqrt(1 - epsilon)
upper_bound = np.sqrt(1 + epsilon)

# Draw the diagram
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