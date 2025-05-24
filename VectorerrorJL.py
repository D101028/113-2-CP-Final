import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, norm

from func import generate_matrix_with_singular_values, sketched_svd, draw_diagram2

def test_vector_error_JL(N = 500, n = 20, k = 10, epsilon = 0.1, delta = 0.01):
    """
    >>> N, n = 500, 20     # Dimension of X
    >>> k = 10             # rank
    >>> epsilon = 0.1      # Error
    >>> delta = 0.01       # Failure probability
    """
    # Compute m
    m = int(np.ceil((6 / (epsilon ** 2)) * (k * np.log(42 / epsilon) + np.log(2 / delta))))

    # Generate the experiment data
    X, S_X, _, V_X = generate_matrix_with_singular_values(N, n, sigma=np.linspace(10, 1, k))
    _, V_Y = sketched_svd(X, m)

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
    x = np.arange(k)
    draw_diagram2(
        x, 
        (errors, 'bo-', 'Error'), 
        (bounds, 'r--', 'upper bound'), 
        xlabel  = 'j-th right singular vector', 
        ylabel  = 'error(L2 norm)', 
        title   = 'The estimate for right singular vector', 
        figsize = (8, 5)
    )

if __name__ == "__main__":
    ###################################################
    ### Danger! It May Be Very Resources Consuming! ###
    ###################################################
    test_vector_error_JL()
