"""
MLE and MAP: Code Examples
Demonstrating MLE and MAP with practical examples
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from scipy.optimize import minimize

# ==================== MLE EXAMPLES ====================

def mle_bernoulli(data: np.ndarray) -> float:
    """
    MLE for Bernoulli distribution (coin flip)
    
    θ̂_MLE = k/n (number of successes / total trials)
    """
    return np.mean(data)

def mle_normal(data: np.ndarray) -> tuple:
    """
    MLE for Normal distribution
    
    μ̂_MLE = x̄ (sample mean)
    σ̂²_MLE = (1/n) Σ(xᵢ - μ̂)² (sample variance, biased)
    """
    mu_mle = np.mean(data)
    sigma2_mle = np.mean((data - mu_mle)**2)
    return mu_mle, sigma2_mle

def mle_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    MLE for Linear Regression (with Gaussian noise)
    
    ŵ_MLE = (XᵀX)⁻¹Xᵀy  (Ordinary Least Squares)
    """
    return np.linalg.solve(X.T @ X, X.T @ y)

# ==================== MAP EXAMPLES ====================

def map_bernoulli_beta(data: np.ndarray, alpha: float = 1.0, beta_param: float = 1.0) -> float:
    """
    MAP for Bernoulli with Beta prior
    
    θ̂_MAP = (k + α - 1) / (n + α + β - 2)
    
    Where:
    - k: number of successes
    - n: total trials
    - α, β: Beta prior parameters
    """
    k = np.sum(data)
    n = len(data)
    
    if alpha == 1.0 and beta_param == 1.0:
        # Uniform prior → MAP = MLE
        return k / n
    else:
        return (k + alpha - 1) / (n + alpha + beta_param - 2)

def map_linear_regression_ridge(X: np.ndarray, y: np.ndarray, 
                                lambda_reg: float = 1.0) -> np.ndarray:
    """
    MAP for Linear Regression with Gaussian prior (Ridge)
    
    ŵ_MAP = (XᵀX + λI)⁻¹Xᵀy
    
    Where λ = σ²/σ²_prior (regularization strength)
    """
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + lambda_reg * np.eye(n_features), X.T @ y)

# ==================== COMPARISON EXAMPLES ====================

def compare_mle_map_coin():
    """
    Compare MLE and MAP for coin flip example
    """
    print("Coin Flip: MLE vs MAP")
    print("=" * 60)
    
    # Small dataset: 3 flips, 3 heads
    data = np.array([1, 1, 1])  # 3 heads
    
    # MLE
    mle_estimate = mle_bernoulli(data)
    print(f"Data: {len(data)} flips, {np.sum(data)} heads")
    print(f"MLE: θ = {mle_estimate:.4f} (100% heads - extreme!)")
    print()
    
    # MAP with different priors
    priors = [
        ("Uniform (α=1, β=1)", 1.0, 1.0),
        ("Weak prior (α=2, β=2)", 2.0, 2.0),
        ("Moderate prior (α=5, β=5)", 5.0, 5.0),
        ("Strong prior (α=10, β=10)", 10.0, 10.0),
    ]
    
    print("MAP with different priors:")
    for name, alpha, beta_param in priors:
        map_estimate = map_bernoulli_beta(data, alpha, beta_param)
        print(f"  {name}: θ = {map_estimate:.4f}")
    
    print("\nKey Insight:")
    print("  - MLE: Only looks at data → extreme estimate")
    print("  - MAP: Incorporates prior → more reasonable")
    print("  - Stronger prior → closer to prior mean (0.5)")

def compare_mle_map_regression():
    """
    Compare MLE and MAP for linear regression
    """
    print("\nLinear Regression: MLE vs MAP (Ridge)")
    print("=" * 60)
    
    # Generate small dataset
    np.random.seed(42)
    n_samples = 10
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    true_w = np.array([1.0, 0.5, 0.0, 0.0, 0.0])  # Sparse true weights
    y = X @ true_w + 0.1 * np.random.randn(n_samples)
    
    # MLE (OLS)
    w_mle = mle_linear_regression(X, y)
    print(f"True weights: {true_w}")
    print(f"MLE (OLS): {w_mle}")
    print(f"MLE norm: {np.linalg.norm(w_mle):.4f}")
    print()
    
    # MAP (Ridge) with different lambda
    lambdas = [0.01, 0.1, 1.0, 10.0]
    print("MAP (Ridge) with different λ:")
    for lam in lambdas:
        w_map = map_linear_regression_ridge(X, y, lambda_reg=lam)
        print(f"  λ={lam:4.2f}: {w_map}, norm={np.linalg.norm(w_map):.4f}")
    
    print("\nKey Insight:")
    print("  - MLE: Can have large weights (overfitting)")
    print("  - MAP: Shrinks weights toward 0 (regularization)")
    print("  - Higher λ: More shrinkage, smaller weights")

def visualize_mle_map_coin():
    """
    Visualize MLE vs MAP for coin flip
    """
    print("\nVisualizing MLE vs MAP")
    print("=" * 60)
    
    # Data: 3 heads out of 3 flips
    k, n = 3, 3
    
    # MLE
    mle_theta = k / n
    
    # MAP with Beta(2, 2) prior
    alpha, beta_param = 2, 2
    map_theta = (k + alpha - 1) / (n + alpha + beta_param - 2)
    
    # Plot
    theta_range = np.linspace(0.01, 0.99, 100)
    
    # Likelihood
    likelihood = theta_range**k * (1 - theta_range)**(n - k)
    likelihood = likelihood / np.max(likelihood)  # Normalize
    
    # Prior
    prior = beta.pdf(theta_range, alpha, beta_param)
    prior = prior / np.max(prior)  # Normalize
    
    # Posterior
    posterior = beta.pdf(theta_range, k + alpha, n - k + beta_param)
    posterior = posterior / np.max(posterior)  # Normalize
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(theta_range, likelihood, 'b-', linewidth=2, label='Likelihood')
    plt.axvline(mle_theta, color='r', linestyle='--', linewidth=2, label=f'MLE = {mle_theta:.2f}')
    plt.xlabel('θ (probability of heads)')
    plt.ylabel('Normalized Likelihood')
    plt.title('MLE: Maximize Likelihood')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(theta_range, prior, 'g-', linewidth=2, label='Prior Beta(2,2)')
    plt.xlabel('θ')
    plt.ylabel('Normalized Prior')
    plt.title('Prior Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(theta_range, posterior, 'purple', linewidth=2, label='Posterior')
    plt.axvline(mle_theta, color='r', linestyle='--', linewidth=2, label=f'MLE = {mle_theta:.2f}')
    plt.axvline(map_theta, color='orange', linestyle='--', linewidth=2, label=f'MAP = {map_theta:.2f}')
    plt.xlabel('θ')
    plt.ylabel('Normalized Posterior')
    plt.title('MAP: Maximize Posterior')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mle_map_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'mle_map_comparison.png'")
    print(f"\nMLE estimate: {mle_theta:.4f}")
    print(f"MAP estimate: {map_theta:.4f}")
    print("MAP is pulled toward prior mean (0.5)")

# ==================== USAGE ====================

if __name__ == "__main__":
    print("MLE and MAP: Code Examples")
    print("=" * 60)
    
    # Coin flip comparison
    compare_mle_map_coin()
    
    # Regression comparison
    compare_mle_map_regression()
    
    # Visualization
    try:
        visualize_mle_map_coin()
    except Exception as e:
        print(f"\nVisualization skipped: {e}")
        print("(matplotlib may not be available)")

