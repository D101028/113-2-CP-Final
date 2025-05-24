import matplotlib.pyplot as plt
import numpy as np

from func import generate_matrix_with_singular_values, sketched_svd

def test_value_ration_n_JL(N = 500, n = 20, k = 10, epsilon = 0.1, m = 40):
    """
    >>> N, n = 500, 20      # Dimension of X 
    >>> k = 10              # Rank
    >>> epsilon = 0.1       # Error
    >>> m = 40              # Compressed rows number
    """

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

if __name__ == "__main__":
    test_value_ration_n_JL()
