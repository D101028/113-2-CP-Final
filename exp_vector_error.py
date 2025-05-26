import numpy as np

from func import generate_matrix_with_singular_values, sketched_svd, m_func, draw_diagram3

def test_value_ratio_JL(N = 500, n = 20, k = 10, epsilon = 0.1, delta = 0.01):
    ######################################################################
    ### Danger! Running This Function May Be Very Resources Consuming! ###
    ######################################################################
    """
    Compute results from given epsilon and delta. 
    m is computed from epsilon and delta. 
    >>> N, n = 500, 20     # Dimension of X
    >>> k = 10             # Rank 
    >>> epsilon = 0.1      # Error
    >>> delta = 0.01       # Failure probability
    """
    # Compute m
    m = m_func(k, epsilon, delta)

    # Generate the experiment data
    X, true_singular_values, _, _ = generate_matrix_with_singular_values(N, n, sigma=np.linspace(10, 1, k))
    sketch_singular_values, _ = sketched_svd(X, m)

    # Take the first k terms
    sigma_X = true_singular_values[:k]
    sigma_Y = sketch_singular_values[:k]

    # Compute the ratio with the corresponding bound
    ratios = sigma_Y / sigma_X
    lower_bound = np.sqrt(1 - epsilon)
    upper_bound = np.sqrt(1 + epsilon)

    # Draw the diagram
    draw_diagram3(range(1, k+1), 
        (ratios, 'σ_Y / σ_X'), 
        (lower_bound, 'Theorem 1 lower bound'), 
        (upper_bound, 'Theorem 1 upper bound'), 
        xlabel  = "Singular Value Index (j)", 
        ylable  = "σ_j(Y) / σ_j(X)", 
        title   = "Ratio of Sketched vs True Singular Values"
    )

def test_value_ratio_n_JL(N = 500, n = 20, k = 10, epsilon = 0.1, m = 40):
    """
    Compute results from given epsilon and m
    >>> N, n = 500, 20      # Dimension of X 
    >>> k = 10              # Rank
    >>> epsilon = 0.1       # Error
    >>> m = 40              # Compressed rows number
    """

    # Generate the experiment data
    X, true_singular_values, _, _ = generate_matrix_with_singular_values(N, n, sigma=np.linspace(10, 1, k))
    sketch_singular_values, _ = sketched_svd(X, m)

    # Take the first k terms
    sigma_X = true_singular_values[:k]
    sigma_Y = sketch_singular_values[:k]

    # Compute the ratio with the corresponding bound
    ratios = sigma_Y / sigma_X
    lower_bound = np.sqrt(1 - epsilon)
    upper_bound = np.sqrt(1 + epsilon)

    # Draw the diagram
    draw_diagram3(
        range(1, k+1), 
        (ratios, 'σ_Y / σ_X'), 
        (lower_bound, 'Theorem 1 lower bound'), 
        (upper_bound, 'Theorem 1 upper bound'), 
        xlabel  = "Singular Value Index (j)", 
        ylabel  = "σ_j(Y) / σ_j(X)", 
        title   = "Ratio of Sketched vs True Singular Values"
    )

if __name__ == "__main__":
    test_value_ratio_n_JL()
