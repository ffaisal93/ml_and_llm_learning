# Topic 21: Dimensionality Reduction

> 🔥 **For interviews, read these first:**
> - **`DIMENSIONALITY_REDUCTION_DEEP_DIVE.md`** — frontier-lab interview deep dive: PCA derivation (variance maximization → eigendecomposition), SVD connection, Eckart-Young, kernel PCA, t-SNE (KL with Student-t), UMAP (fuzzy simplicial complex), autoencoders, VAE, ICA, NMF, method-selection guide.
> - **`INTERVIEW_GRILL.md`** — 60+ active-recall questions.

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

## Core Intuition

Dimensionality reduction is about compressing data while keeping the most useful structure.

That structure might be:
- variance
- neighborhood geometry
- cluster separation
- visualization-friendly layout

Different methods preserve different notions of structure.

### PCA

PCA is the most important method to understand deeply because it is mathematically clean and frequently asked in interviews.

Its intuition is:
- find orthogonal directions of greatest variance
- project the data onto those directions

This is useful when redundant dimensions can be replaced by a smaller number of informative directions.

### t-SNE

t-SNE is mostly a visualization method.

Its goal is to preserve local neighborhoods rather than give a globally faithful geometric map.

That is why pretty t-SNE plots can be useful but also misleading if over-interpreted.

### UMAP

UMAP is also mainly used for visualization and low-dimensional structure discovery.

Compared with t-SNE, it is often faster and may preserve more global structure, but it is still not a drop-in replacement for linear methods like PCA.

## Technical Details Interviewers Often Want

### Why PCA Uses Eigenvectors

The covariance matrix tells you how directions in feature space vary together.

Its eigenvectors give principal directions, and eigenvalues tell you how much variance each direction explains.

### PCA Also Minimizes Reconstruction Error

This is a key follow-up.

PCA is not only "maximize variance." It also gives the best low-rank linear approximation in the least-squares sense.

### PCA Needs Centering

If you do not center the data first, the first component can be dominated by the mean offset rather than the true variation structure.

That is a common interview implementation bug.

## Common Failure Modes

- using PCA without centering
- interpreting t-SNE distances globally as if they were metric-faithful
- using PCA when the key structure is strongly nonlinear
- assuming explained variance always equals downstream usefulness
- choosing dimensionality only by visualization aesthetics

## Edge Cases and Follow-Up Questions

1. Why must PCA center the data first?
2. Why can PCA fail on nonlinear manifolds?
3. Why is t-SNE mainly a visualization tool rather than a general feature extractor?
4. Why can the first two PCs fail to separate classes even when the full space is predictive?
5. What is the difference between preserving variance and preserving neighborhoods?

## What to Practice Saying Out Loud

1. Why PCA is both a variance-maximization and reconstruction-minimization method
2. Why PCA is linear and what that implies
3. Why visualization methods can be useful but also misleading

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
