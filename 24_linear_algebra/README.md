# Topic 24: Linear Algebra Q&A

## What You'll Learn

This topic covers common linear algebra questions:
- Matrix operations
- Eigenvalues and eigenvectors
- SVD (Singular Value Decomposition)
- Matrix inverses
- Common interview questions with easy explanations

## Why We Need This

### Interview Importance
- **Common questions**: Linear algebra is fundamental
- **Math foundation**: Many ML concepts use linear algebra
- **Problem-solving**: Shows mathematical maturity

### Real-World Application
- **All ML**: Uses linear algebra
- **Neural networks**: Matrix operations
- **PCA, SVD**: Dimensionality reduction

## Common Interview Questions

### Q1: What are eigenvalues and eigenvectors?

**Answer:**
- **Eigenvector**: Vector that doesn't change direction when matrix is applied
- **Eigenvalue**: Scaling factor
- **Formula**: Av = λv (where A is matrix, v is eigenvector, λ is eigenvalue)

**Example:**
```
A = [[2, 1],
     [1, 2]]

Eigenvalues: λ₁ = 3, λ₂ = 1
Eigenvectors: v₁ = [1, 1], v₂ = [1, -1]
```

### Q2: What is SVD?

**Answer:**
- **SVD**: Singular Value Decomposition
- **Formula**: A = U Σ V^T
- **U**: Left singular vectors (eigenvectors of AA^T)
- **Σ**: Singular values (diagonal matrix)
- **V**: Right singular vectors (eigenvectors of A^T A)

**Use Cases:**
- Dimensionality reduction
- Matrix approximation
- PCA (related to SVD)

### Q3: When is a matrix invertible?

**Answer:**
- Matrix is invertible if determinant ≠ 0
- Equivalent: Full rank (no linearly dependent rows/columns)
- Equivalent: All eigenvalues ≠ 0

### Q4: What is the rank of a matrix?

**Answer:**
- **Rank**: Number of linearly independent rows/columns
- Maximum number of linearly independent vectors
- Dimension of column space (or row space)

### Q5: What is a positive definite matrix?

**Answer:**
- **Positive definite**: All eigenvalues > 0
- **Semi-definite**: All eigenvalues ≥ 0
- **Properties**: x^T A x > 0 for all x ≠ 0

## Code Implementation

See `linear_algebra_qa.py` for implementations.

## Theory

### Matrix Operations
- **Multiplication**: (A @ B)[i,j] = Σ A[i,k] × B[k,j]
- **Transpose**: (A^T)[i,j] = A[j,i]
- **Inverse**: A × A^(-1) = I

### Eigenvalue Decomposition
- **Formula**: A = Q Λ Q^(-1)
- **Q**: Eigenvectors (columns)
- **Λ**: Eigenvalues (diagonal)

### SVD vs Eigenvalue Decomposition
- **Eigenvalue**: Only for square matrices
- **SVD**: Works for any matrix
- **Relation**: For square matrices, related but different

## Exercises

1. Compute eigenvalues/eigenvectors
2. Implement SVD
3. Check matrix invertibility
4. Compute matrix rank

## Next Steps

- **Topic 25**: Final review
- Practice all topics

