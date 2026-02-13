"""
PCA: Principal Component Analysis
Interview question: "Implement PCA from scratch"

Mathematical Formulation:
1. Center data: X_centered = X - mean(X)
2. Covariance matrix: C = (1/n) X_centered^T @ X_centered
3. Eigenvalue decomposition: C = Q Λ Q^T
   - Q: Eigenvectors (principal components)
   - Λ: Eigenvalues (explained variance)
4. Select top k components: Q_k = Q[:, :k]
5. Project: Y = X_centered @ Q_k

Why it works:
- Principal components are directions of maximum variance
- First PC captures most variance
- PCs are orthogonal (uncorrelated)
- Minimizes reconstruction error
"""
import numpy as np

def pca(X: np.ndarray, n_components: int) -> tuple:
    """
    PCA from scratch
    
    Steps:
    1. Center data (subtract mean)
    2. Compute covariance matrix
    3. Eigenvalue decomposition
    4. Select top n_components eigenvectors
    5. Project data
    
    Args:
        X: Data matrix (n_samples, n_features)
        n_components: Number of components to keep
    
    Returns:
        (transformed_data, components, explained_variance)
    """
    # Center data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Covariance matrix: C = (1/n) X^T X
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Eigenvalue decomposition: C = Q Λ Q^T
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    # Project data: Y = X @ components
    X_transformed = X_centered @ components
    
    return X_transformed, components, explained_variance

def explained_variance_ratio(explained_variance: np.ndarray) -> np.ndarray:
    """Compute explained variance ratio"""
    total_variance = np.sum(explained_variance)
    return explained_variance / total_variance

def reconstruct(X_transformed: np.ndarray, components: np.ndarray, 
                X_mean: np.ndarray) -> np.ndarray:
    """
    Reconstruct original data from PCA components
    
    X_reconstructed = X_transformed @ components^T + mean
    """
    return X_transformed @ components.T + X_mean


# Usage Example
if __name__ == "__main__":
    print("PCA from Scratch")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create correlated data
    X = np.random.randn(n_samples, n_features)
    # Add correlation
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n_samples)
    
    print(f"Original data shape: {X.shape}")
    
    # Apply PCA
    n_components = 2
    X_transformed, components, explained_variance = pca(X, n_components)
    
    print(f"\nTransformed data shape: {X_transformed.shape}")
    print(f"Components shape: {components.shape}")
    print(f"Explained variance: {explained_variance}")
    
    # Explained variance ratio
    variance_ratio = explained_variance_ratio(explained_variance)
    print(f"\nExplained variance ratio: {variance_ratio}")
    print(f"Total variance explained: {np.sum(variance_ratio):.4f}")
    
    # Reconstruction
    X_mean = np.mean(X, axis=0)
    X_reconstructed = reconstruct(X_transformed, components, X_mean)
    reconstruction_error = np.mean((X - X_reconstructed)**2)
    print(f"\nReconstruction error (MSE): {reconstruction_error:.6f}")

