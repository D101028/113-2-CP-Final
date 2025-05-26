import numpy as np
from numpy.linalg import norm

from func import generate_matrix_with_singular_values, sketched_svd, m_func, draw_diagram2

def compute_errors_and_bounds(N, n, k, m, epsilon):
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
    return errors, bounds

def test_vector_error_JL(N = 500, n = 20, k = 10, epsilon = 0.1, delta = 0.01):
    """
    Compute results from given epsilon and delta. 
    m is computed from epsilon and delta. 
    >>> N, n = 500, 20     # Dimension of X
    >>> k = 10             # rank
    >>> epsilon = 0.1      # Error
    >>> delta = 0.01       # Failure probability
    """
    # Compute m
    m = m_func(k, epsilon, delta)

    errors, bounds = compute_errors_and_bounds(N, n, k, m, epsilon)

    # Draw the diagram
    draw_diagram2(
        np.arange(k), 
        (errors, 'bo-', 'Error'), 
        (bounds, 'r--', 'upper bound'), 
        xlabel  = 'j-th right singular vector', 
        ylabel  = 'error(L2 norm)', 
        title   = 'The estimate for right singular vector', 
        figsize = (8, 5)
    )

def test_vector_error_n_JL(N = 500, n = 20, k = 10, epsilon = 0.1, m = 40):
    """
    Compute results from given epsilon and m. 
    >>> N, n = 500, 20    # Dimension of X
    >>> k = 10            # rank
    >>> m = 40            # Set m
    >>> epsilon = 0.1     # Error
    """
    
    errors, bounds = compute_errors_and_bounds(N, n, k, m, epsilon)

    # Draw the diagram
    draw_diagram2(
        np.arange(k), 
        (errors, 'bo-', 'Error'), 
        (bounds, 'r--', 'upper bound'), 
        xlabel  = 'j-th right singular vector', 
        ylabel  = 'error(L2 norm)', 
        title   = 'The estimate for right singular vector', 
        figsize = (8, 5)
    )

if __name__ == "__main__":
    test_vector_error_JL()
