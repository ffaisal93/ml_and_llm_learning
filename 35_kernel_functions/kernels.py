"""
Kernel Functions: Complete Implementation
All kernels with detailed explanations and examples
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# ==================== LINEAR KERNEL ====================

def linear_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """
    Linear Kernel: K(x, y) = x · y = x^T y
    
    What it does:
    - Computes standard dot product
    - Assumes data is linearly separable
    - No transformation (original space)
    
    When to use:
    - Linearly separable data
    - High-dimensional data (many features)
    - Baseline (start here)
    - When interpretability matters
    
    Advantages:
    - Fast (simple computation)
    - Interpretable
    - Less overfitting
    
    Disadvantages:
    - Can't handle non-linear relationships
    """
    return np.dot(x, y)

# ==================== POLYNOMIAL KERNEL ====================

def polynomial_kernel(x: np.ndarray, y: np.ndarray, 
                      degree: int = 2, 
                      gamma: float = 1.0, 
                      coef0: float = 0.0) -> float:
    """
    Polynomial Kernel: K(x, y) = (γ * x^T y + r)^d
    
    What it does:
    - Computes dot product in polynomial feature space
    - Implicitly creates features: x₁², x₂², x₁x₂, etc.
    - For degree=2: Creates quadratic features
    
    How it works:
    Instead of explicitly computing [x₁, x₂, x₁², x₂², x₁x₂],
    the kernel computes the dot product in this space efficiently.
    
    Parameters:
    - degree (d): Polynomial degree
      * d=2: Quadratic (most common)
      * d=3: Cubic
      * Higher: Rare (overfitting risk)
    - gamma (γ): Controls influence of higher-order terms
      * Higher: More polynomial influence
      * Lower: More like linear
    - coef0 (r): Bias term
      * Usually 0.0
    
    When to use:
    - Polynomial relationships (y = x², etc.)
    - Moderate non-linearity
    - When you know the relationship is polynomial
    
    Example:
    If true boundary is circular (x₁² + x₂² = r²),
    polynomial kernel with degree=2 can separate it.
    """
    return (gamma * np.dot(x, y) + coef0) ** degree

# ==================== RBF KERNEL ====================

def rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """
    RBF (Radial Basis Function) Kernel: K(x, y) = exp(-γ * ||x - y||²)
    
    Also called: Gaussian kernel
    
    What it does:
    - Measures similarity based on distance
    - Points close together → high similarity (≈1)
    - Points far apart → low similarity (≈0)
    - Creates infinite-dimensional feature space
    
    How it works:
    Each data point creates a "bump" (Gaussian) in feature space.
    Kernel value is high when two points are in same bump (close),
    low when in different bumps (far).
    
    Visual intuition:
    - Imagine each point has a Gaussian "influence zone"
    - Overlapping zones → high kernel value
    - Non-overlapping zones → low kernel value
    
    Parameters:
    - gamma (γ): Controls kernel width
      * High γ: Narrow kernel (small radius)
        - Each point only influences nearby points
        - More complex boundaries (risk overfitting)
        - More support vectors
      * Low γ: Wide kernel (large radius)
        - Each point influences many points
        - Simpler boundaries (risk underfitting)
        - Fewer support vectors
      * Rule of thumb: γ = 1 / (n_features * variance)
    
    When to use:
    - Non-linear problems (default choice)
    - Complex decision boundaries
    - Local structure (similar points should be close)
    - Most common kernel for non-linear SVM
    
    Advantages:
    - Very flexible (handles complex boundaries)
    - Works well for most non-linear problems
    - Only one parameter to tune
    - Smooth decision boundaries
    
    Disadvantages:
    - Can overfit with high gamma
    - Computationally more expensive
    - Less interpretable
    
    Example:
    If data is in concentric circles, RBF can separate them.
    Linear and polynomial kernels cannot.
    """
    distance_squared = np.sum((x - y) ** 2)
    return np.exp(-gamma * distance_squared)

# ==================== SIGMOID KERNEL ====================

def sigmoid_kernel(x: np.ndarray, y: np.ndarray,
                   gamma: float = 1.0, coef0: float = 0.0) -> float:
    """
    Sigmoid Kernel: K(x, y) = tanh(γ * x^T y + r)
    
    What it does:
    - Similar to neural network activation
    - Less commonly used
    - Not always positive definite (can cause issues)
    
    When to use:
    - Rarely used in practice
    - RBF is usually better
    - Specific use cases only
    
    Note: Usually not recommended. Use RBF instead.
    """
    return np.tanh(gamma * np.dot(x, y) + coef0)

# ==================== KERNEL COMPARISON ====================

def compare_kernels_example():
    """
    Compare different kernels on same data
    """
    print("Kernel Comparison Example")
    print("=" * 60)
    
    # Two data points
    x1 = np.array([0, 0])
    x2 = np.array([1, 1])
    x3 = np.array([0, 0])  # Same as x1
    
    print(f"Points: x1={x1}, x2={x2}, x3={x3} (same as x1)")
    print()
    
    # Linear kernel
    lin_12 = linear_kernel(x1, x2)
    lin_11 = linear_kernel(x1, x3)
    print(f"Linear Kernel:")
    print(f"  K(x1, x2) = {lin_12:.4f}")
    print(f"  K(x1, x1) = {lin_11:.4f}")
    print()
    
    # Polynomial kernel (degree=2)
    poly_12 = polynomial_kernel(x1, x2, degree=2)
    poly_11 = polynomial_kernel(x1, x3, degree=2)
    print(f"Polynomial Kernel (degree=2):")
    print(f"  K(x1, x2) = {poly_12:.4f}")
    print(f"  K(x1, x1) = {poly_11:.4f}")
    print()
    
    # RBF kernel
    rbf_12 = rbf_kernel(x1, x2, gamma=1.0)
    rbf_11 = rbf_kernel(x1, x3, gamma=1.0)
    print(f"RBF Kernel (gamma=1.0):")
    print(f"  K(x1, x2) = {rbf_12:.4f} (different points, low similarity)")
    print(f"  K(x1, x1) = {rbf_11:.4f} (same point, maximum similarity)")
    print()
    
    # RBF with different gamma
    rbf_12_low = rbf_kernel(x1, x2, gamma=0.1)
    rbf_12_high = rbf_kernel(x1, x2, gamma=10.0)
    print(f"RBF Kernel with different gamma:")
    print(f"  gamma=0.1: K(x1, x2) = {rbf_12_low:.4f} (wider kernel, higher similarity)")
    print(f"  gamma=10.0: K(x1, x2) = {rbf_12_high:.4f} (narrower kernel, lower similarity)")
    print()
    print("Key Insight:")
    print("  - Higher gamma: Points need to be closer to have high similarity")
    print("  - Lower gamma: Points can be farther and still have high similarity")

# ==================== WHEN TO USE WHICH ====================

def kernel_selection_guide():
    """
    Guide for choosing the right kernel
    """
    print("Kernel Selection Guide")
    print("=" * 60)
    
    print("\n1. Start with Linear Kernel:")
    print("   - Fast, interpretable")
    print("   - If it works, use it!")
    print("   - Good for high-dimensional data (text, images with many features)")
    print()
    
    print("2. If Linear Fails, Try RBF:")
    print("   - Most common for non-linear problems")
    print("   - Tune gamma parameter")
    print("   - Usually works well")
    print()
    
    print("3. If RBF Overfits, Try Polynomial:")
    print("   - Less flexible than RBF")
    print("   - More interpretable")
    print("   - Try degree=2 or 3")
    print()
    
    print("4. Never Use Sigmoid:")
    print("   - Unless you have specific reason")
    print("   - RBF is almost always better")
    print()
    
    print("Parameter Tuning (RBF):")
    print("   - Gamma too high: Overfitting (narrow kernel)")
    print("   - Gamma too low: Underfitting (wide kernel)")
    print("   - Try: 0.001, 0.01, 0.1, 1.0, 10.0")
    print("   - C (regularization): 0.1, 1, 10, 100, 1000")

# ==================== KERNEL TRICK EXPLANATION ====================

def kernel_trick_explanation():
    """
    Explain the kernel trick in detail
    """
    print("The Kernel Trick: Detailed Explanation")
    print("=" * 60)
    
    print("""
Problem: We want to use linear SVM on non-linear data.

Solution 1: Transform data to high dimensions
  - Original: x = [x₁, x₂]
  - Transform: φ(x) = [x₁, x₂, x₁², x₂², x₁x₂, √2x₁, √2x₂, ...]
  - Problem: Very expensive! High-dimensional computation is slow.

Solution 2: Kernel Trick
  - Don't compute φ(x) explicitly
  - Instead, use kernel K(x, y) = φ(x) · φ(y)
  - Kernel computes dot product in high-dimensional space directly
  - Much faster! We never compute high-dimensional features.

Example:
  Polynomial kernel (degree=2):
  - Instead of computing [x₁, x₂, x₁², x₂², x₁x₂, ...]
  - We compute K(x, y) = (x · y)²
  - This gives us the same result, but much faster!

Why it works:
  - SVM only needs dot products (not the features themselves)
  - Kernel computes dot products in transformed space
  - We get benefit of high dimensions without the cost

Key insight:
  - Kernel = dot product in some (possibly infinite) feature space
  - We never need to know what that space is
  - We just need the kernel function
    """)

# ==================== USAGE ====================

if __name__ == "__main__":
    print("Kernel Functions: Complete Guide")
    print("=" * 60)
    print()
    
    # Examples
    x = np.array([1, 2])
    y = np.array([3, 4])
    
    print("Example: x = [1, 2], y = [3, 4]")
    print()
    
    # Linear
    print(f"Linear Kernel: K(x, y) = {linear_kernel(x, y)}")
    print("  (Just dot product: 1*3 + 2*4 = 11)")
    print()
    
    # Polynomial
    print(f"Polynomial Kernel (degree=2): K(x, y) = {polynomial_kernel(x, y, degree=2)}")
    print("  ((1*3 + 2*4)² = 11² = 121)")
    print()
    
    # RBF
    distance = np.sqrt(np.sum((x - y)**2))
    print(f"RBF Kernel (gamma=1.0): K(x, y) = {rbf_kernel(x, y, gamma=1.0):.4f}")
    print(f"  (Distance = {distance:.4f}, exp(-distance²) = {rbf_kernel(x, y, gamma=1.0):.4f})")
    print()
    
    # Same point
    print(f"RBF Kernel (same point): K(x, x) = {rbf_kernel(x, x, gamma=1.0):.4f}")
    print("  (Same point, distance=0, exp(0) = 1.0)")
    print()
    
    # Comparison
    compare_kernels_example()
    print()
    
    # Selection guide
    kernel_selection_guide()
    print()
    
    # Kernel trick
    kernel_trick_explanation()

