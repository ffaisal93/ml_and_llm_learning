# Linear Algebra for ML — Interview Grill

> 50 questions on rank, eigendecomp, SVD, PSD, matrix calculus, conditioning, projections. Drill until you can answer 35+ cold.

---

## A. Rank and subspaces

**1. Define rank of a matrix.**
Dimension of the column space (= dimension of the row space). Equivalently, number of linearly independent rows or columns.

**2. State the rank-nullity theorem.**
For $A \in \mathbb{R}^{m \times n}$: $\mathrm{rank}(A) + \dim(\mathrm{Null}(A)) = n$.

**3. What does row rank = column rank mean intuitively?**
A counterintuitive fact. Both give the same number; this is a deep theorem proved via SVD or RREF arguments.

**4. Inequality for $\mathrm{rank}(AB)$?**
$\mathrm{rank}(AB) \leq \min(\mathrm{rank}(A), \mathrm{rank}(B))$.

**5. When is $X^\top X$ invertible?**
When $X$ has full column rank (columns linearly independent).

**6. What if $X^\top X$ is singular in OLS?**
Use pseudoinverse, or add ridge ($X^\top X + \lambda I$), or remove redundant columns.

**7. What's the four fundamental subspaces?**
$\mathrm{Col}(A)$, $\mathrm{Null}(A)$, $\mathrm{Row}(A) = \mathrm{Col}(A^\top)$, $\mathrm{Null}(A^\top)$. $\mathrm{Col}(A) \perp \mathrm{Null}(A^\top)$, $\mathrm{Row}(A) \perp \mathrm{Null}(A)$.

---

## B. Eigendecomposition

**8. Define eigenvalue and eigenvector.**
$Av = \lambda v$ with $v \neq 0$. $\lambda$ is the eigenvalue, $v$ the eigenvector.

**9. How do you find eigenvalues?**
Roots of characteristic polynomial: $\det(A - \lambda I) = 0$.

**10. State the spectral theorem.**
Real symmetric matrix $A$ has $n$ real eigenvalues and an orthonormal basis of eigenvectors. $A = Q\Lambda Q^\top$ with $Q$ orthogonal.

**11. Why are eigenvectors of distinct eigenvalues orthogonal (for symmetric $A$)?**
$\lambda_1 v_1^\top v_2 = (A v_1)^\top v_2 = v_1^\top A v_2 = \lambda_2 v_1^\top v_2$. If $\lambda_1 \neq \lambda_2$, must have $v_1^\top v_2 = 0$.

**12. Which matrices are NOT diagonalizable?**
Defective matrices — those without a full set of linearly independent eigenvectors. E.g., $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has only one eigenvector (up to scaling).

**13. Eigenvalues of $A^k$?**
$\lambda^k$ for each eigenvalue $\lambda$ of $A$.

**14. Eigenvalues of $A^{-1}$?**
$1/\lambda$ for each $\lambda \neq 0$.

**15. What's the spectral radius?**
$\rho(A) = \max_i |\lambda_i|$ — largest absolute eigenvalue. Determines convergence/divergence of $A^k$.

---

## C. SVD

**16. State the SVD theorem.**
Any $A \in \mathbb{R}^{m \times n}$ factors as $A = U \Sigma V^\top$ with $U, V$ orthogonal and $\Sigma$ diagonal with non-negative singular values.

**17. Geometric interpretation of SVD?**
Rotation ($V^\top$) → axis-aligned scaling ($\Sigma$) → rotation ($U$). Any linear map decomposes this way.

**18. SVD vs eigendecomposition?**
SVD works for any matrix; eigendecomposition only for diagonalizable square matrices. For symmetric PSD, they coincide. SVD = eigendecomposition of $A^\top A$ (or $A A^\top$).

**19. What's the operator norm of $A$ in terms of SVD?**
Largest singular value: $\|A\|_2 = \sigma_1$.

**20. Frobenius norm in terms of SVD?**
$\|A\|_F = \sqrt{\sum_i \sigma_i^2}$.

**21. How do you compute rank from SVD?**
Number of nonzero singular values (in practice, number greater than some tolerance).

**22. State Eckart-Young.**
The truncated SVD $A_k = U_k \Sigma_k V_k^\top$ is the best rank-$k$ approximation in operator and Frobenius norms.

**23. Why does PCA reduce to SVD?**
Centered data $X$. Covariance $\Sigma_X = X^\top X / n$. Eigendecomp of $\Sigma_X$ = right singular vectors $V$ of $X$. PCA scores = $US$.

**24. SVD of a low-rank matrix?**
Rank-$r$ matrix has only $r$ nonzero singular values. Truncated SVD with $k=r$ recovers exactly.

**25. What's the pseudoinverse via SVD?**
$A^+ = V \Sigma^+ U^\top$ where $\Sigma^+$ inverts the nonzero singular values. Solves least-squares for any $A$.

---

## D. PSD / definiteness

**26. Define positive semidefinite.**
Symmetric and $x^\top A x \geq 0$ for all $x$. Equivalently, all eigenvalues $\geq 0$.

**27. Define positive definite.**
PSD + $x^\top A x > 0$ for $x \neq 0$. All eigenvalues $> 0$.

**28. Three equivalent characterizations of PSD?**
(1) $x^\top A x \geq 0 \forall x$. (2) All eigenvalues $\geq 0$. (3) $A = B^\top B$ for some $B$.

**29. Why is the Hessian PSD at a local minimum?**
Necessary second-order condition: at a local min, the function curves upward (or flat) in every direction.

**30. Why is covariance always PSD?**
$\mathrm{Cov}(X) = \mathbb{E}[(X - \mu)(X - \mu)^\top]$. For any $w$: $w^\top \mathrm{Cov}(X) w = \mathrm{Var}(w^\top X) \geq 0$.

**31. Why must kernel matrices be PSD?**
Mercer's theorem: a kernel function corresponds to an inner product in some Hilbert space iff its Gram matrix is PSD for any data.

**32. Sum of two PSD matrices?**
PSD: $x^\top(A+B)x = x^\top A x + x^\top B x \geq 0$.

**33. Product of two PSD matrices — always PSD?**
No (in general). $AB$ may not even be symmetric. PSD only if $A, B$ commute.

**34. Cholesky decomposition — when does it exist?**
For PD matrices: $A = L L^\top$ with $L$ lower triangular and positive diagonal. For PSD, need to allow zeros (semi-Cholesky).

---

## E. Matrix calculus

**35. $\nabla_x (b^\top x) = ?$**
$b$.

**36. $\nabla_x (x^\top A x) = ?$**
$(A + A^\top) x$. For symmetric $A$: $2Ax$.

**37. $\nabla_x \|y - Ax\|^2 = ?$**
$-2A^\top(y - Ax) = 2A^\top A x - 2 A^\top y$.

**38. Hessian of $\|y - Ax\|^2$?**
$2 A^\top A$. PSD always; PD iff $A$ has full column rank.

**39. Closed-form OLS?**
$\hat{x} = (A^\top A)^{-1} A^\top y$.

**40. What's the chain rule for matrix functions?**
$d(f \circ g)/dx = (df/dg)(dg/dx)$ — Jacobian product. Backprop is exactly this.

**41. Derivative of $\log \det A$ w.r.t. $A$?**
$A^{-T}$. Used in VAEs, normalizing flows, GMM.

---

## F. Conditioning

**42. Definition of condition number?**
$\kappa(A) = \sigma_{\max}/\sigma_{\min}$ for invertible $A$. Measures sensitivity to perturbations.

**43. Why does it matter for gradient descent?**
GD on a quadratic with Hessian $H$ converges at rate $\propto (\kappa - 1)/(\kappa + 1)$. Large $\kappa$ → slow.

**44. How does Adam help with bad conditioning?**
Per-coordinate adaptive learning rates approximate diagonal preconditioning. Effectively rescales axes — not perfect, but helps when curvature varies axis-by-axis.

**45. How does normalization (BN/LN) help with conditioning?**
Renormalizes activations → reduces conditioning of intermediate Jacobians/Hessians. One reason normalization speeds up training.

**46. What does adding $\lambda I$ to a matrix do to its condition number?**
Reduces $\kappa$. New eigenvalues $\lambda_i + \lambda$. Smallest eigenvalue boosted from $\lambda_n$ to $\lambda_n + \lambda$. Ridge regression's stabilizing effect.

---

## G. Projections and OLS

**47. Define a projection matrix.**
$P^2 = P$. Orthogonal projection: also $P = P^\top$.

**48. Projection onto column space of $X$?**
$P = X(X^\top X)^{-1} X^\top$.

**49. Geometric view of OLS solution?**
$\hat{y} = Py$ — projection of $y$ onto $\mathrm{Col}(X)$. Residual $y - \hat{y}$ is orthogonal to columns of $X$ (normal equations).

**50. Trace of the hat matrix $P$?**
$\mathrm{tr}(P) = \mathrm{rank}(X)$ = degrees of freedom of the fit.

---

## Quick fire

**51.** *Operator norm of $A$?* $\sigma_{\max}$.
**52.** *Frobenius norm via SVD?* $\sqrt{\sum \sigma_i^2}$.
**53.** *Best rank-k approximation?* Truncated SVD.
**54.** *Eigenvalues of $A^\top A$?* $\sigma_i^2$ of $A$.
**55.** *Hessian of $\frac{1}{2}\|Xw - y\|^2$?* $X^\top X$.
**56.** *Trace of $AB$ vs $BA$?* Equal.
**57.** *Determinant of an orthogonal matrix?* $\pm 1$.
**58.** *Inverse of an orthogonal matrix?* Its transpose.
**59.** *PSD allows what decomposition?* Cholesky.
**60.** *Rank of an outer product $uv^\top$?* 1 (unless $u$ or $v$ is zero).

---

## Self-grading

If you can't answer 1-15, you don't know basic linear algebra. If you can't answer 16-35, you'll get tripped up on PCA/SVD/optimization questions. If you can't answer 36-50, frontier-lab interviews on matrix calculus / numerical methods will go past you.

Aim for 40+/60 cold.
