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

## Core Intuition

Linear algebra matters in ML because models operate on vectors, matrices, and linear transformations almost everywhere.

If you understand the geometry, many formulas become easier to reason about.

### Eigenvectors and Eigenvalues

An eigenvector is a direction that a matrix transforms without changing its direction.

The eigenvalue tells you how much that direction is scaled.

This matters because eigenvectors often reveal the most important directions of action of a matrix.

### SVD

SVD is powerful because it works for any matrix, not just square ones.

A good intuition is:
- one rotation
- one scaling
- another rotation

That is why SVD appears in PCA, compression, denoising, and low-rank approximation.

### Rank

Rank tells you how many independent directions or independent pieces of information are really present.

Low rank often means:
- redundancy
- compressibility
- reduced intrinsic dimensionality

## Technical Details Interviewers Often Want

### Why SVD Is So Useful

SVD provides the best low-rank approximation of a matrix in a least-squares sense.

This is one of the most important conceptual reasons it shows up in ML.

### Invertibility

A matrix is invertible only if it has full rank.

Equivalent views:
- determinant nonzero
- no zero eigenvalues
- columns are linearly independent

Interviewers often like hearing those equivalences cleanly.

### Positive Definite Matrices

Positive definite matrices matter in optimization and covariance reasoning.

They are associated with:
- strictly positive quadratic forms
- positive curvature
- stable covariance structure

## Common Failure Modes

- memorizing definitions without geometric meaning
- confusing eigenvalue decomposition with SVD
- forgetting that eigenvalue decomposition does not apply generally to all matrices
- treating rank as just a formula rather than a measure of independent information

## Edge Cases and Follow-Up Questions

1. Why does SVD work for non-square matrices while eigen decomposition does not in the same way?
2. Why does low rank imply redundancy?
3. Why does invertibility require full rank?
4. Why are positive definite matrices important in optimization?
5. Why is SVD central to PCA?

## What to Practice Saying Out Loud

1. A geometric explanation of eigenvectors
2. Why SVD is more general than eigendecomposition
3. Why rank matters in ML and numerical stability

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
