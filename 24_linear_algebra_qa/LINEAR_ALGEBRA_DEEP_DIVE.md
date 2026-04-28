# Linear Algebra for ML — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

ML is linear algebra at scale plus calculus. Senior interviews probe whether you understand the *operations* you're doing — not just the syntax — and whether you can reason about properties (rank, conditioning, definiteness) that determine whether a method works or fails.

---

## 1. Matrices as linear maps

A matrix $A \in \mathbb{R}^{m \times n}$ is a linear map $\mathbb{R}^n \to \mathbb{R}^m$. Five fundamental subspaces (the four classical, plus null space of $A^\top$):

- **Column space** $\mathrm{Col}(A) \subseteq \mathbb{R}^m$: outputs $A$ can produce.
- **Null space** $\mathrm{Null}(A) \subseteq \mathbb{R}^n$: $\{x : Ax = 0\}$.
- **Row space** $\mathrm{Row}(A) = \mathrm{Col}(A^\top)$.
- **Left null space** $\mathrm{Null}(A^\top) \subseteq \mathbb{R}^m$.

**Rank-nullity**: $\mathrm{rank}(A) + \dim(\mathrm{Null}(A)) = n$.

**Rank facts**:
- $\mathrm{rank}(A) = \mathrm{rank}(A^\top)$ (row rank = column rank).
- $\mathrm{rank}(AB) \leq \min(\mathrm{rank}(A), \mathrm{rank}(B))$.
- For $A \in \mathbb{R}^{m \times n}$: full rank means $\mathrm{rank} = \min(m,n)$.

---

## 2. Eigendecomposition

For square $A \in \mathbb{R}^{n \times n}$:

$$
A v = \lambda v
$$

$\lambda$ is an eigenvalue, $v$ a (right) eigenvector. The characteristic polynomial $\det(A - \lambda I) = 0$ gives eigenvalues.

**Diagonalization**: if $A$ has $n$ linearly independent eigenvectors, then $A = V \Lambda V^{-1}$ where $\Lambda$ is diagonal of eigenvalues.

### Symmetric matrices — special

If $A = A^\top$:
- All eigenvalues are real.
- Eigenvectors of distinct eigenvalues are orthogonal.
- $A$ is diagonalizable: $A = Q \Lambda Q^\top$ with $Q$ orthogonal.

This is the **spectral theorem**. It's why PCA (covariance is symmetric), kernel methods, and tons of ML rely on it.

### Powers and functions of matrices

$A^k = V \Lambda^k V^{-1}$. So $\Lambda^k$ raises eigenvalues to the $k$-th power. This is why repeated multiplication by $A$ converges (or explodes) based on the largest $|\lambda|$ — the spectral radius.

For symmetric $A$: $f(A) = Q f(\Lambda) Q^\top$ for any analytic $f$.

---

## 3. SVD — the universal factorization

For any $A \in \mathbb{R}^{m \times n}$:

$$
A = U \Sigma V^\top
$$

- $U \in \mathbb{R}^{m \times m}$, orthogonal. Columns are left singular vectors.
- $\Sigma \in \mathbb{R}^{m \times n}$, "diagonal" with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.
- $V \in \mathbb{R}^{n \times n}$, orthogonal. Columns are right singular vectors.

**Geometric intuition**: $A$ rotates ($V^\top$), scales axes ($\Sigma$), then rotates again ($U$). Any linear map decomposes this way.

### Connections to other things

- **Rank**: number of nonzero singular values.
- **$\|A\|_2$ (operator norm)**: largest singular value $\sigma_1$.
- **$\|A\|_F$ (Frobenius)**: $\sqrt{\sum \sigma_i^2}$.
- **Condition number**: $\kappa(A) = \sigma_1/\sigma_r$.
- **Pseudoinverse**: $A^+ = V \Sigma^+ U^\top$ where $\Sigma^+$ inverts nonzero singular values.

### Eckart-Young theorem

The truncated SVD $A_k = U_k \Sigma_k V_k^\top$ (top-$k$ singular components) is the best rank-$k$ approximation to $A$ in both operator and Frobenius norm. Foundation of PCA, low-rank matrix completion, model compression.

### Connection to eigendecomposition

For symmetric PSD $A$: SVD = eigendecomposition (singular values = eigenvalues, left = right singular vectors = eigenvectors).

For general $A$:
- $A^\top A = V \Sigma^\top \Sigma V^\top$ — eigendecomposition of $A^\top A$ has eigenvalues $\sigma_i^2$ and eigenvectors $V$.
- $A A^\top = U \Sigma \Sigma^\top U^\top$ — eigendecomp gives eigenvectors $U$.

This is how SVD is computed numerically (in practice via more stable bidiagonalization, but conceptually).

---

## 4. Positive (semi)definiteness

A symmetric matrix $A$ is:
- **Positive definite (PD)** if $x^\top A x > 0$ for all $x \neq 0$. Equivalent: all eigenvalues $> 0$.
- **Positive semidefinite (PSD)** if $x^\top A x \geq 0$ for all $x$. Equivalent: all eigenvalues $\geq 0$.

### Why PD/PSD matters in ML

- **Covariance matrices** are PSD.
- **Hessian** at a local minimum is PSD; PD at a strict local min.
- **Convex quadratic** $\frac{1}{2}x^\top A x + b^\top x$ is convex iff $A$ is PSD.
- **Kernel matrices** (Gram matrices) must be PSD (Mercer's condition).
- **PD allows Cholesky**: $A = L L^\top$ with $L$ lower-triangular. Numerically efficient for solving.

### Quick PSD check
- $A = B^\top B$ for any $B$ → PSD.
- All principal minors $\geq 0$ → PSD (Sylvester's criterion: leading principal minors $> 0$ for PD).

---

## 5. Matrix calculus — the four core formulas

These come up constantly in derivations.

**Scalar-by-vector** (gradient):

$$
\nabla_x (b^\top x) = b, \quad \nabla_x (x^\top A x) = (A + A^\top) x
$$

For symmetric $A$: $\nabla_x(x^\top A x) = 2 A x$.

**Vector-by-vector** (Jacobian): for $f(x) \in \mathbb{R}^m$, $f$ from $\mathbb{R}^n$, $J_{ij} = \partial f_i / \partial x_j$.

**Scalar-by-matrix**: $\nabla_W \mathrm{tr}(W^\top A) = A$, $\nabla_W \mathrm{tr}(A W^\top B) = B^\top A^\top$.

**Chain rule for Jacobians**: $J_{f \circ g}(x) = J_f(g(x)) \cdot J_g(x)$.

### OLS gradient — derive it once

$\mathcal{L}(w) = \frac{1}{2}\|y - Xw\|^2 = \frac{1}{2}(y - Xw)^\top(y - Xw)$.

$\nabla_w \mathcal{L} = -X^\top(y - Xw) = X^\top X w - X^\top y$.

Setting to zero: $\hat{w} = (X^\top X)^{-1} X^\top y$ (when $X^\top X$ invertible).

Hessian: $\nabla^2 \mathcal{L} = X^\top X$ — PSD always; PD if $X$ has full column rank.

---

## 6. Matrix norms

| Norm | Formula | Property |
|---|---|---|
| Frobenius | $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2}$ | Sum of squared entries |
| Operator (spectral) | $\|A\|_2 = \sigma_{\max}$ | Largest stretch |
| Nuclear | $\|A\|_* = \sum \sigma_i$ | Convex relaxation of rank |
| 1-norm | $\|A\|_1 = \max_j \sum_i |a_{ij}|$ | Max column abs-sum |
| $\infty$-norm | $\|A\|_\infty = \max_i \sum_j |a_{ij}|$ | Max row abs-sum |

Frobenius is the default in ML (it's just $\ell_2$ on the vectorized matrix). Nuclear norm is used as a convex relaxation of rank — the workhorse of low-rank matrix completion.

---

## 7. Condition number — why training breaks

For a square invertible $A$:

$$
\kappa(A) = \|A\| \|A^{-1}\| = \sigma_1 / \sigma_n
$$

When solving $Ax = b$, perturbations in $b$ are amplified by $\kappa$. Large condition number = ill-conditioned = numerically unstable.

### Why ML cares
- **Hessian conditioning** controls gradient descent convergence rate. Convex quadratic with Hessian $H$: GD with optimal step $\eta = 2/(L + \mu)$ contracts at rate $((\kappa - 1)/(\kappa + 1))^k$; with simpler step $1/L$, contracts at $(1 - \mu/L)^k$. Bad conditioning → slow.
- **Adaptive optimizers** (Adam, RMSprop) approximate per-parameter rescaling — implicitly handle bad conditioning.
- **Normalization** (BN, LN) reduces internal-layer condition number, which is one explanation for why it speeds up training.

### Improving conditioning
- Standardize features (subtract mean, divide by SD).
- Whiten data.
- Add diagonal: $A + \lambda I$ — ridge regression bumps small eigenvalues, lowers $\kappa$.

---

## 8. Projections and least squares

A projection $P$ satisfies $P^2 = P$. Orthogonal if also $P = P^\top$.

For a matrix $X$ with linearly independent columns:

$$
P = X(X^\top X)^{-1} X^\top
$$

projects onto $\mathrm{Col}(X)$. The OLS solution $\hat{w} = (X^\top X)^{-1} X^\top y$ gives $\hat{y} = P y$ — fitted values are the projection of $y$ onto column space.

**Geometric view of OLS:** find the closest point in $\mathrm{Col}(X)$ to $y$. The residual $y - \hat{y}$ is orthogonal to $\mathrm{Col}(X)$ — the *normal equations*: $X^\top(y - X\hat{w}) = 0$.

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Is rank always $\min(m, n)$? | Yes | Only if full rank — rank can be lower |
| Is $X^\top X$ always invertible? | Yes | Only if $X$ has full column rank |
| Are eigenvectors of a symmetric matrix unique? | Yes | Only up to sign and degenerate-eigenvalue rotation |
| What's the difference between rank and dimension? | Same thing | Dimension is for spaces; rank is for matrices (= dim of column/row space) |
| Largest eigenvalue = operator norm? | Yes | For symmetric matrices yes; in general operator norm is largest *singular* value |
| Does Adam fix bad conditioning? | Yes | Approximately — it rescales per-coordinate, which helps when curvature varies axis-by-axis |
| PSD + PSD = PSD? | Maybe | Yes, sum of PSD is PSD |
| PSD × PSD = PSD? | Yes | Not in general — only if they commute |

---

## 10. Eight most-asked interview questions

1. **Derive OLS gradient and prove the Hessian is PSD.** (Vectorized chain rule + $X^\top X \succeq 0$.)
2. **What's the SVD of a matrix and why is it unique?** (Up to sign of singular vectors when SVs are distinct; up to a rotation when degenerate.)
3. **Why does PCA work? Connect to SVD.** (Eigendecomposition of covariance = SVD of centered data; top-$k$ approx via Eckart-Young.)
4. **What's a condition number and when does it matter?** (Sensitivity of solution; affects GD convergence; normalization helps.)
5. **What does it mean for a matrix to be PSD? List 3 equivalent characterizations.** (All eigenvalues $\geq 0$; $x^\top A x \geq 0$; $A = B^\top B$.)
6. **Compute the gradient of $\|Ax - b\|^2$ w.r.t. $x$.** (Should take 30 seconds: $2A^\top(Ax - b)$.)
7. **Why is $X^\top X$ used instead of $X X^\top$ in OLS?** (Solves for $w \in \mathbb{R}^d$, dim of features. Use $XX^\top$ when $n < d$ — kernel trick.)
8. **What's the geometric meaning of the rank of a matrix?** (Dim of column space = "number of independent output directions"; if $A$ is a linear map, $\mathrm{rank} = $ dim of image.)

---

## 11. Drill plan

- Derive OLS gradient + Hessian + closed form on paper. Repeat until 2 minutes.
- Recite SVD definition, properties, connection to eigendecomp.
- For a $3 \times 3$ symmetric matrix, compute eigenvalues and eigenvectors by hand.
- For each ML method (PCA, ridge, OLS, kernel ridge), state the relevant linear algebra fact it relies on.
- Recite three equivalent definitions of PSD; derive Cholesky for a $2 \times 2$ PD.

---

## 12. Further reading

- Strang, *Introduction to Linear Algebra* — the canonical undergrad text.
- Trefethen & Bau, *Numerical Linear Algebra* — focused on what actually breaks numerically.
- Petersen & Pedersen, *The Matrix Cookbook* — quick reference for matrix calculus.
- Boyd & Vandenberghe, *Convex Optimization*, Appendix A — concise linear algebra refresher.
