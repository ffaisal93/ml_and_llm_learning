# Topic 21: Dimensionality Reduction

## What You'll Learn

This topic teaches you dimensionality reduction:
- PCA (Principal Component Analysis)
- t-SNE
- UMAP
- Theory, math, and code

## Why We Need This

### Interview Importance
- **Common question**: "Implement PCA from scratch"
- **Math knowledge**: Shows mathematical understanding
- **Practical skill**: Used in many ML pipelines

### Real-World Application
- **Visualization**: Reduce to 2D/3D
- **Feature reduction**: Remove redundancy
- **Preprocessing**: Before other algorithms

## Industry Use Cases

### 1. **PCA**
**Use Case**: Most common
- Feature reduction
- Visualization
- Noise reduction

### 2. **t-SNE**
**Use Case**: Visualization
- 2D/3D visualization
- Cluster visualization
- Data exploration

### 3. **UMAP**
**Use Case**: Modern alternative
- Better than t-SNE
- Faster
- Preserves global structure

## Industry-Standard Boilerplate Code

### PCA from Scratch

```python
"""
PCA: Principal Component Analysis
Interview question: "Implement PCA"
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
    
    Returns: (transformed_data, components, explained_variance)
    """
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    # Project data
    X_transformed = X_centered @ components
    
    return X_transformed, components, explained_variance
```

### PCA Math

**Mathematical Formulation:**

1. **Covariance Matrix**: C = (1/n) X^T X
2. **Eigenvalue Decomposition**: C = Q Λ Q^T
3. **Principal Components**: Columns of Q (eigenvectors)
4. **Explained Variance**: Eigenvalues
5. **Projection**: Y = X Q_k (where Q_k is top k components)

**Why it works:**
- Maximizes variance in projected space
- Minimizes reconstruction error
- Orthogonal components

## Theory

### PCA Properties
- **Variance**: First PC has max variance
- **Orthogonality**: PCs are orthogonal
- **Reconstruction**: Can reconstruct from PCs

### When to Use
- **High-dimensional data**: Reduce dimensions
- **Visualization**: 2D/3D plots
- **Noise reduction**: Remove low-variance components
- **Feature selection**: Use top PCs

## Exercises

1. Implement PCA
2. Visualize in 2D
3. Compute explained variance
4. Reconstruct from PCs

## Next Steps

- **Topic 22**: Recommendation systems
- **Topic 23**: Clustering evaluation

