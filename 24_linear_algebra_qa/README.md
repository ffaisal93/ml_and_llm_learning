# Topic 24: Linear Algebra Q&A

> 🔥 **For interviews, read these first:**
> - **`LINEAR_ALGEBRA_DEEP_DIVE.md`** — frontier-lab deep dive: rank, eigendecomposition, SVD (with Eckart-Young), positive (semi)definiteness, matrix calculus (OLS gradient + Hessian), conditioning, projections.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic covers the linear algebra you actually need for ML interviews:
- Rank, four fundamental subspaces, rank-nullity
- Eigendecomposition and the spectral theorem
- SVD as the universal factorization
- Positive (semi)definiteness — covariance, Hessians, kernel matrices
- Matrix calculus — derivatives that show up in OLS, ridge, neural nets
- Condition number and why it matters for optimization
- Projections and the geometric view of OLS

## Why This Matters

Almost every ML algorithm is linear algebra at scale. PCA = eigendecomposition of covariance. Ridge = solving a regularized linear system. Neural networks = stacked linear maps with nonlinearities. SVD shows up in compression, recommender systems, low-rank adaptation (LoRA), embedding spaces.

Senior interviews probe whether you understand the *operations* — not just the names.

## Next Steps

- **Topic 37**: MLE and MAP — links squared loss to Gaussian MLE, ridge to Gaussian MAP
- **Topic 21**: Dimensionality reduction — direct SVD application
- **Topic 35**: Kernel functions — Gram matrices and PSD
