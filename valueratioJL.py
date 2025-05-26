import numpy as np

from func import generate_matrix_with_singular_values, sketched_svd, draw_diagram3

def test_value_ratio_JL(N = 500, n = 20, k = 10, epsilon = 0.1, delta = 0.01):
    """
    >>> N, n = 500, 20     # Dimension of X
    >>> k = 10             # Rank 
    >>> epsilon = 0.1      # Error
    >>> delta = 0.01       # Failure probability
    """
    # Compute m
    m = int(np.ceil((6 / (epsilon ** 2)) * (k * np.log(42 / epsilon) + np.log(2 / delta))))

    # Generate the experiment data
    X, true_singular_values, _, _ = generate_matrix_with_singular_values(N, n, sigma=np.linspace(10, 1, k))
    sketch_singular_values, _ = sketched_svd(X, m)

    # Take the first k terms
    σ_X = true_singular_values[:k]
    σ_Y = sketch_singular_values[:k]

    # Compute the ratio with the corresponding bound
    ratios = σ_Y / σ_X
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

if __name__ == "__main__":
    ###################################################
    ### Danger! It May Be Very Resources Consuming! ###
    ###################################################
    test_value_ratio_JL()
