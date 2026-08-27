# Linear Algebra for ML — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

ML is linear algebra at scale plus calculus. Senior interviews probe whether you understand the *operations* you're doing — not just the syntax — and whether you can reason about properties (rank, conditioning, definiteness) that determine whether a method works or fails.

---

## 1. Matrices as linear maps

A matrix $A \in \mathbb{R}^{m \times n}$ is a linear map $\mathbb{R}^n \to \mathbb{R}^m$. Four fundamental subspaces:

- **Column space** $\mathrm{Col}(A) \subseteq \mathbb{R}^m$: outputs $A$ can produce.
- **Null space** $\mathrm{Null}(A) \subseteq \mathbb{R}^n$: $\{x : Ax = 0\}$.
- **Row space** $\mathrm{Row}(A) = \mathrm{Col}(A^\top)$.
- **Left null space** $\mathrm{Null}(A^\top) \subseteq \mathbb{R}^m$.

**Rank-nullity**: $\mathrm{rank}(A) + \dim(\mathrm{Null}(A)) = n$.

> **Saying it out loud.** A matrix is a machine that turns vectors into vectors, and the four subspaces just bookkeep what it can produce and what it destroys. The column space is everything it can output. The nullspace is everything it crushes to zero. Rank-nullity says those two have to account for all your input dimensions: whatever doesn't survive got annihilated. That's why collinear features are a problem, since they put a direction in the nullspace that your model literally cannot distinguish.

**Rank facts**:
- $\mathrm{rank}(A) = \mathrm{rank}(A^\top)$ (row rank = column rank).
- $\mathrm{rank}(AB) \leq \min(\mathrm{rank}(A), \mathrm{rank}(B))$.
- For $A \in \mathbb{R}^{m \times n}$: full rank means $\mathrm{rank} = \min(m,n)$.

> **Saying it out loud.** Three rank facts worth having on instant recall. Row rank equals column rank, which is genuinely surprising because rows and columns live in different-sized spaces, and the reason is that both count the nonzero singular values. The rank of a product is bounded by the smaller of the two ranks, because you can lose dimensions by multiplying but never create them. And full rank just means as high as the shape allows. That second fact is the entire mathematical justification for LoRA: a $d$-by-$r$ times an $r$-by-$d$ can never exceed rank $r$.

---

## 2. Eigendecomposition

> **In plain language.** Eigenvectors are the directions a matrix leaves pointing the same way. Push most vectors through a matrix and they get rotated and stretched; eigenvectors only get stretched, by a factor called the eigenvalue. Finding them means finding the coordinate system in which the matrix does nothing but scale each axis independently, which turns a messy transformation into simple multiplication.

For square $A \in \mathbb{R}^{n \times n}$:

$$
A v = \lambda v
$$

$\lambda$ is an eigenvalue, $v$ a (right) eigenvector. The characteristic polynomial $\det(A - \lambda I) = 0$ gives eigenvalues.

**Diagonalization**: if $A$ has $n$ linearly independent eigenvectors, then $A = V \Lambda V^{-1}$ where $\Lambda$ is diagonal of eigenvalues.

> **Saying it out loud.** An eigenvector is a direction the matrix doesn't turn, it only stretches, and the eigenvalue is how much. If you can find $n$ independent such directions, you can rewrite the matrix as a change of basis, a diagonal scaling, and a change back, which is diagonalization. In that basis a complicated transformation becomes $n$ independent multiplications. The formal way to find eigenvalues is the characteristic polynomial, though nobody does that in code past three-by-three because polynomial roots are numerically awful.

### Symmetric matrices — special

If $A = A^\top$:
- All eigenvalues are real.
- Eigenvectors of distinct eigenvalues are orthogonal.
- $A$ is diagonalizable: $A = Q \Lambda Q^\top$ with $Q$ orthogonal.

This is the **spectral theorem**. It's why PCA (covariance is symmetric), kernel methods, and tons of ML rely on it.

> **Saying it out loud.** Symmetric matrices are the well-behaved case and that's why ML lives in them. All the eigenvalues come out real, eigenvectors for different eigenvalues are automatically perpendicular, and you always get a full orthonormal basis, so diagonalization never fails. That's the spectral theorem, and it means every symmetric matrix is just rotate, stretch along axes, rotate back. Covariance matrices, Hessians, and kernel matrices are all symmetric, which is exactly why PCA and kernel methods have clean closed-form answers instead of iterative approximations.

### Powers and functions of matrices

$A^k = V \Lambda^k V^{-1}$. So $\Lambda^k$ raises eigenvalues to the $k$-th power. This is why repeated multiplication by $A$ converges (or explodes) based on the largest $|\lambda|$ — the spectral radius.

For symmetric $A$: $f(A) = Q f(\Lambda) Q^\top$ for any analytic $f$.

> **Saying it out loud.** Raising a matrix to a power raises its eigenvalues to that power while the eigenvectors stay fixed, so long-run behavior is decided entirely by the largest eigenvalue in absolute value. Above one, repeated application explodes; below one, it decays to nothing. That's exactly the vanishing and exploding gradient story in recurrent networks, where backprop through time is repeated multiplication by a Jacobian. And for symmetric matrices you can apply any analytic function eigenvalue-by-eigenvalue, which is how matrix square roots and matrix exponentials get computed.

---

## 3. SVD — the universal factorization

> **In plain language.** SVD is the one decomposition that works on absolutely any matrix, square or not. It says every matrix does the same three things in sequence: rotate, stretch along the new axes, rotate again. The stretch factors are the singular values, sorted biggest first, and they tell you which directions the matrix actually cares about.

For any $A \in \mathbb{R}^{m \times n}$:

$$
A = U \Sigma V^\top
$$

- $U \in \mathbb{R}^{m \times m}$, orthogonal. Columns are left singular vectors.
- $\Sigma \in \mathbb{R}^{m \times n}$, "diagonal" with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.
- $V \in \mathbb{R}^{n \times n}$, orthogonal. Columns are right singular vectors.

**Geometric intuition**: $A$ rotates ($V^\top$), scales axes ($\Sigma$), then rotates again ($U$). Any linear map decomposes this way.

> **Saying it out loud.** SVD says every matrix, no exceptions, factors into a rotation, a stretch along the axes, and another rotation. Take the unit sphere, push it through the matrix, and you always get an ellipsoid: the singular values are the lengths of its axes and the columns of $U$ say which way those axes point. That universality is why it's the most useful factorization in applied math, since eigendecomposition demands square and non-defective while SVD demands nothing. And the singular values come out sorted, so you get a free ranking of which directions matter.

### Connections to other things

- **Rank**: number of nonzero singular values.
- **$\|A\|_2$ (operator norm)**: largest singular value $\sigma_1$.
- **$\|A\|_F$ (Frobenius)**: $\sqrt{\sum \sigma_i^2}$.
- **Condition number**: $\kappa(A) = \sigma_1/\sigma_r$.
- **Pseudoinverse**: $A^+ = V \Sigma^+ U^\top$ where $\Sigma^+$ inverts nonzero singular values.

> **Saying it out loud.** Almost every matrix quantity you care about reads straight off the SVD. Rank is the count of nonzero singular values. The operator norm is the largest one, the maximum stretch. Frobenius is the square root of their sum of squares. The condition number is the ratio of largest to smallest, which is how elongated that ellipsoid is. And the pseudoinverse is the same factorization with the nonzero singular values flipped over. So one decomposition answers rank, norms, conditioning, and inversion all at once.

### Eckart-Young theorem

The truncated SVD $A_k = U_k \Sigma_k V_k^\top$ (top-$k$ singular components) is the best rank-$k$ approximation to $A$ in both operator and Frobenius norm. Foundation of PCA, low-rank matrix completion, model compression.

> **Saying it out loud.** Eckart-Young says the best rank-$k$ approximation to any matrix is just its truncated SVD, keep the top $k$ pieces and drop the rest. What makes it remarkable is that it's optimal in both the operator norm and the Frobenius norm at the same time, and that a problem which sounds combinatorial has a closed-form answer. It's the foundation under PCA, low-rank matrix completion, and model compression. The concrete payoff is that the leftover error is exactly the singular values you discarded, so you know your approximation error before you commit to $k$.

### Connection to eigendecomposition

For symmetric PSD $A$: SVD = eigendecomposition (singular values = eigenvalues, left = right singular vectors = eigenvectors).

For general $A$:
- $A^\top A = V \Sigma^\top \Sigma V^\top$ — eigendecomposition of $A^\top A$ has eigenvalues $\sigma_i^2$ and eigenvectors $V$.
- $A A^\top = U \Sigma \Sigma^\top U^\top$ — eigendecomp gives eigenvectors $U$.

This is how SVD is computed numerically (in practice via more stable bidiagonalization, but conceptually).

> **Saying it out loud.** SVD and eigendecomposition are the same thing for symmetric positive semidefinite matrices, and related but distinct in general. The bridge is that $A^\top A$ is always symmetric PSD, and its eigenvectors are your right singular vectors $V$ with eigenvalues equal to the singular values squared; do the same on the other side and you get $U$. That's the conceptual recipe, but you should add that real implementations never form $A^\top A$, they use bidiagonalization, because forming that product squares the condition number and costs you half your digits of precision.

---

## 4. Positive (semi)definiteness

A symmetric matrix $A$ is:
- **Positive definite (PD)** if $x^\top A x > 0$ for all $x \neq 0$. Equivalent: all eigenvalues $> 0$.
- **Positive semidefinite (PSD)** if $x^\top A x \geq 0$ for all $x$. Equivalent: all eigenvalues $\geq 0$.

> **Saying it out loud.** Positive definite means the quadratic form is strictly positive in every direction; semidefinite allows zero. Geometrically, the matrix never sends a vector to point backwards from where it started, so the associated bowl always curves upward. The equivalent eigenvalue statement, all eigenvalues nonnegative or all strictly positive, is usually the easier one to check. The practical difference between the two is invertibility, which is exactly what ridge regression buys by adding $\lambda I$.

### Why PD/PSD matters in ML

- **Covariance matrices** are PSD.
- **Hessian** at a local minimum is PSD; PD at a strict local min.
- **Convex quadratic** $\frac{1}{2}x^\top A x + b^\top x$ is convex iff $A$ is PSD.
- **Kernel matrices** (Gram matrices) must be PSD (Mercer's condition).
- **PD allows Cholesky**: $A = L L^\top$ with $L$ lower-triangular. Numerically efficient for solving.

> **Saying it out loud.** PSD shows up everywhere in ML because it's the algebraic signature of convexity and of squared quantities. Covariance matrices are PSD because variance can't be negative. A quadratic is convex exactly when its matrix is PSD. Hessians are PSD at minima. Kernel matrices have to be PSD or there's no feature space they correspond to, which is Mercer's condition. And PD lets you do Cholesky, which is the fastest way to solve a system or sample from a Gaussian. If you see PSD in a problem, the underlying claim is almost always convexity or a valid inner product.

### Quick PSD check
- $A = B^\top B$ for any $B$ → PSD.
- All principal minors $\geq 0$ → PSD (Sylvester's criterion: leading principal minors $> 0$ for PD).

> **Saying it out loud.** Two quick ways to check PSD. If the matrix is $B^\top B$ for any $B$, it's automatically PSD, no computation needed, and this covers covariance matrices and Gram matrices at a glance. Otherwise use Sylvester's criterion on the leading principal minors, or just attempt a Cholesky factorization, which fails precisely when the matrix isn't positive definite. The Cholesky attempt is the cheap practical test, and it's why Gaussian-process code adds a small jitter to the diagonal when it fails.

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

> **Saying it out loud.** Four formulas cover almost every derivation you'll be asked to do. The gradient of a linear form is just the vector. The gradient of a quadratic form is $(A + A^\top)x$, which is $2Ax$ when $A$ is symmetric, which it almost always is. The Jacobian is the matrix of all partials for a vector-valued function. And the chain rule is Jacobian multiplication, which is literally what backpropagation does. The one thing that trips people up is layout convention, so decide up front whether gradients are rows or columns and stay consistent, because mixing conventions is how you get a mysterious stray transpose.

### OLS gradient — derive it once

$\mathcal{L}(w) = \frac{1}{2}\|y - Xw\|^2 = \frac{1}{2}(y - Xw)^\top(y - Xw)$.

$\nabla_w \mathcal{L} = -X^\top(y - Xw) = X^\top X w - X^\top y$.

Setting to zero: $\hat{w} = (X^\top X)^{-1} X^\top y$ (when $X^\top X$ invertible).

Hessian: $\nabla^2 \mathcal{L} = X^\top X$ — PSD always; PD if $X$ has full column rank.

> **Saying it out loud.** Derive OLS once and you own it. Write the loss as a half squared residual, differentiate to get minus $X^\top$ times the residual, set it to zero and you have the normal equations, giving $\hat w$ equals $X^\top X$ inverse times $X^\top y$. The Hessian is $X^\top X$, constant, always PSD, and positive definite exactly when $X$ has full column rank, which is why the problem is convex with a unique answer when your features are independent. And you should add that you'd never actually invert that matrix in code, since its condition number is the square of $X$'s, so you'd use QR or SVD instead.

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

> **Saying it out loud.** Norms differ in what kind of error they care about. Frobenius treats the matrix as one long vector and averages over everything, which is why it's the default in ML loss functions. The operator norm is worst-case, the biggest stretch in any direction, which is what you constrain for Lipschitz and stability guarantees. And the nuclear norm, the sum of the singular values, is the convex relaxation of rank, playing the same role for matrices that L1 plays for vectors: it's the penalty that actually produces low-rank solutions, and it's the workhorse of matrix completion.

---

## 7. Condition number — why training breaks

For a square invertible $A$:

$$
\kappa(A) = \|A\| \|A^{-1}\| = \sigma_1 / \sigma_n
$$

When solving $Ax = b$, perturbations in $b$ are amplified by $\kappa$. Large condition number = ill-conditioned = numerically unstable.

> **Saying it out loud.** The condition number is the ratio of the largest singular value to the smallest, and it tells you how much error gets amplified when you solve a system. A condition number of a million means a tiny perturbation in your data can move the solution by a factor of a million, so you lose about six of your sixteen digits. Geometrically it's how needle-shaped the ellipsoid is. The rule of thumb worth quoting: you lose roughly $\log_{10} \kappa$ digits of accuracy.

### Why ML cares
- **Hessian conditioning** controls gradient descent convergence rate. Convex quadratic with Hessian $H$: GD with optimal step $\eta = 2/(L + \mu)$ contracts at rate $((\kappa - 1)/(\kappa + 1))^k$; with simpler step $1/L$, contracts at $(1 - \mu/L)^k$. Bad conditioning → slow.
- **Adaptive optimizers** (Adam, RMSprop) approximate per-parameter rescaling — implicitly handle bad conditioning.
- **Normalization** (BN, LN) reduces internal-layer condition number, which is one explanation for why it speeds up training.

> **Saying it out loud.** ML cares about conditioning because it sets the speed limit on gradient descent. On a quadratic, the contraction per step depends on the condition number, so a $\kappa$ of a thousand means creeping progress, and the picture is a long narrow valley where the gradient points across rather than along. Adam partly rescues this by rescaling each coordinate, which is a crude diagonal preconditioner. Normalization layers help by keeping intermediate Jacobians better conditioned, which is the modern explanation for why they let you raise the learning rate by an order of magnitude.

### Improving conditioning
- Standardize features (subtract mean, divide by SD).
- Whiten data.
- Add diagonal: $A + \lambda I$ — ridge regression bumps small eigenvalues, lowers $\kappa$.

> **Saying it out loud.** Three ways to improve conditioning, in increasing order of aggressiveness. Standardize your features, which alone fixes most of the damage, since a feature in the thousands next to a feature in the hundredths creates a huge eigenvalue spread. Whiten the data, which fully decorrelates and equalizes, at the cost of amplifying noise directions. Or add $\lambda I$, which lifts every eigenvalue and therefore shrinks the ratio, which is exactly ridge regression's stabilizing effect. All three trade a bit of bias or fidelity for numerical sanity.

---

## 8. Projections and least squares

A projection $P$ satisfies $P^2 = P$. Orthogonal if also $P = P^\top$.

For a matrix $X$ with linearly independent columns:

$$
P = X(X^\top X)^{-1} X^\top
$$

projects onto $\mathrm{Col}(X)$. The OLS solution $\hat{w} = (X^\top X)^{-1} X^\top y$ gives $\hat{y} = P y$ — fitted values are the projection of $y$ onto column space.

**Geometric view of OLS:** find the closest point in $\mathrm{Col}(X)$ to $y$. The residual $y - \hat{y}$ is orthogonal to $\mathrm{Col}(X)$ — the *normal equations*: $X^\top(y - X\hat{w}) = 0$.

> **Saying it out loud.** Least squares is a projection, and once you see that the algebra stops being mysterious. Your target vector almost never lies in the span of your feature columns, so the best you can do is drop a perpendicular onto that span; the foot of the perpendicular is your fitted vector and the hat matrix does the dropping. Requiring the residual to be perpendicular to every column is literally the normal equations. One nice consequence to name: with an intercept column of ones in the model, the residuals must sum to zero, because they have to be orthogonal to that column.

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

> **Saying it out loud.** The gotchas that catch people: rank is only $\min(m,n)$ if the matrix is full rank, $X^\top X$ is invertible only with full column rank, and eigenvectors are unique only up to sign, and not even that when eigenvalues repeat. The operator norm is the largest singular value, which coincides with the largest eigenvalue magnitude only for symmetric matrices. And the PSD pair: sums of PSD matrices are PSD, but products generally aren't, since they aren't even symmetric unless the two matrices commute.

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
