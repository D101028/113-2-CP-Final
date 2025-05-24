import numpy as np
from numpy.linalg import svd, norm
import matplotlib.pyplot as plt

# Compute the theoretical bound
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

# Variable setting
N, n, k = 500, 20, 10
m = 40          # Set m
j = 0           # We only consider the first right singular vector
epsilon = 0.1   # Error
gaps = np.linspace(0.01, 3.0, 30) # Gap

vector_errors = []
theoretical_bounds = []

# Select error and the theoretical bound
for gap in gaps:
    # Generate singular values
    s1 = 10
    singular_vals = [s1 - i * gap for i in range(k)]
    singular_vals = np.maximum(singular_vals, 0.1)  # Here we require singular to be nonnegative
    S = np.diag(singular_vals)

    # Construct X with rank(X)=k
    U = np.linalg.qr(np.random.randn(N, k))[0]
    V = np.linalg.qr(np.random.randn(n, k))[0]
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

    # The error of the j-th vector 
    v = V_X[:, j]
    v_p = V_Y[:, j]

    # Ensure the nonnegative inner product
    if np.dot(v, v_p) < 0:
        v_p = -v_p
    err = norm(v - v_p)
    vector_errors.append(err)

    # Thereotical bound
    bound = compute_theoretical_bound(singular_vals, j, epsilon)
    theoretical_bounds.append(bound)

# Draw the diagram
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
