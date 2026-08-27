# Linear algebra

Linear algebra is tested as a proxy for whether you understand what your model actually computes. The interviewer wants shapes, costs, and one geometric picture per object. The common failure is reciting definitions without the picture: saying "SVD factorises a matrix" without rotate-scale-rotate, or "PCA finds directions of variance" without saying those directions are eigenvectors of the covariance matrix. The second common failure is not knowing why the normal equation is numerically worse than SVD, which is a two-line condition-number argument.

## The equations

**Matrix multiplication, shapes and cost.**

$$C = AB, \qquad C_{ij} = \sum_{k=1}^{m} A_{ik}B_{kj}, \qquad A \in \mathbb{R}^{n \times m},\; B \in \mathbb{R}^{m \times p},\; C \in \mathbb{R}^{n \times p}$$

The inner dimensions must match at $m$ and they vanish from the result; each of the $np$ output entries costs $m$ multiplies and $m$ adds, so the cost is $O(nmp)$ multiply-accumulates, or $2nmp$ floating-point operations.

**Dot product and its geometric form.**

$$\mathbf{a}^\top\mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\|_2\,\|\mathbf{b}\|_2\cos\theta, \qquad \cos\theta = \frac{\mathbf{a}^\top\mathbf{b}}{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}$$

$\theta$ is the angle between the two vectors, so the dot product mixes magnitude and direction; dividing out both norms leaves cosine similarity, which is direction only.

**Vector and matrix norms.**

$$\|\mathbf{x}\|_1 = \sum_i |x_i|, \quad \|\mathbf{x}\|_2 = \sqrt{\textstyle\sum_i x_i^2}, \quad \|A\|_F = \sqrt{\textstyle\sum_{ij}A_{ij}^2} = \sqrt{\textstyle\sum_i \sigma_i^2}, \quad \|A\|_2 = \max_{\mathbf{x} \neq 0}\frac{\|A\mathbf{x}\|_2}{\|\mathbf{x}\|_2} = \sigma_{\max}$$

$\sigma_i$ are the singular values of $A$; the Frobenius norm treats the matrix as one long vector, and the induced spectral norm is the largest factor by which $A$ can stretch any vector.

**Rank and rank-nullity.**

$$\operatorname{rank}(A) = \dim(\operatorname{col}(A)) = \dim(\operatorname{row}(A)), \qquad \operatorname{rank}(A) + \dim(\ker(A)) = m \;\;\text{for}\;\; A \in \mathbb{R}^{n \times m}$$

Rank is the number of genuinely independent directions in the output; the null space holds the input directions $A$ crushes to zero, and the two dimensions must add up to the number of columns.

**Eigenvalue equation.**

$$A\mathbf{v} = \lambda\mathbf{v}, \qquad \det(A - \lambda I) = 0$$

$\mathbf{v} \neq \mathbf{0}$ is an eigenvector and $\lambda$ its eigenvalue; these are the directions $A$ only stretches and never rotates, and they exist for square $A$ only.

**Spectral decomposition of a symmetric matrix.**

$$A = Q\Lambda Q^\top = \sum_{i=1}^{d}\lambda_i \mathbf{q}_i\mathbf{q}_i^\top, \qquad Q^\top Q = I$$

For symmetric real $A$ the eigenvalues $\lambda_i$ are real and the eigenvectors $\mathbf{q}_i$ can be chosen orthonormal, so $A$ is a sum of rank-one pieces along perpendicular axes.

**Singular value decomposition.**

$$A = U\Sigma V^\top, \quad U \in \mathbb{R}^{n \times n},\; V \in \mathbb{R}^{m \times m} \;\text{orthogonal}, \quad A^\top A = V\Sigma^\top\Sigma V^\top, \quad \sigma_i = \sqrt{\lambda_i(A^\top A)}$$

Every matrix, square or not, has an SVD; the right singular vectors are eigenvectors of the Gram matrix $A^\top A$, the left ones are eigenvectors of $AA^\top$, and the singular values are the square roots of those shared eigenvalues.

**Pseudo-inverse and least squares.**

$$A^{+} = V\Sigma^{+}U^\top, \quad \Sigma^{+}_{ii} = \begin{cases}1/\sigma_i & \sigma_i > \tau\\ 0 & \text{otherwise}\end{cases}, \qquad \mathbf{w}^\star = A^{+}\mathbf{b} \;\;\overset{\text{full rank}}{=}\;\; (A^\top A)^{-1}A^\top\mathbf{b}$$

The normal equation $A^\top A\mathbf{w} = A^\top\mathbf{b}$ comes from setting the gradient of $\|A\mathbf{w} - \mathbf{b}\|_2^2$ to zero; the pseudo-inverse gives the same answer when $A$ has full column rank and the minimum-norm answer when it does not.

**Covariance matrix and the PCA objective.**

$$C = \frac{1}{n-1}X_c^\top X_c \in \mathbb{R}^{d \times d}, \qquad \mathbf{w}_1 = \arg\max_{\|\mathbf{w}\|_2 = 1} \mathbf{w}^\top C\mathbf{w}$$

$X_c$ is the data matrix with the column means subtracted; the constrained maximum of that quadratic form is the top eigenvector of $C$, with value the top eigenvalue, so PCA is exactly an eigendecomposition of the covariance.

**Positive semi-definiteness, three equivalent statements.**

$$A \succeq 0 \iff \mathbf{x}^\top A\mathbf{x} \ge 0 \;\;\forall \mathbf{x} \iff \lambda_i \ge 0 \;\;\forall i \iff A = B^\top B \;\;\text{for some } B$$

$A$ must be symmetric for this to be meaningful; the three forms are the quadratic-form test, the spectrum test, and the factorisation test, and each is the useful one in a different proof.

**Determinant as a volume factor.**

$$\det(A) = \prod_{i=1}^{d}\lambda_i, \qquad \operatorname{vol}(A\,S) = |\det(A)|\cdot\operatorname{vol}(S), \qquad \det(AB) = \det(A)\det(B)$$

$S$ is any region of $\mathbb{R}^d$; the determinant is the signed factor by which $A$ scales volume, and a negative sign means the map flips orientation.

**Condition number.**

$$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} = \|A\|_2\|A^{-1}\|_2, \qquad \kappa_2(A^\top A) = \kappa_2(A)^2$$

$\kappa$ bounds how much a relative error in the input is amplified in the output; forming the Gram matrix squares it, which is the whole numerical case against the normal equation.

**Trace and the cyclic property.**

$$\operatorname{tr}(A) = \sum_i A_{ii} = \sum_i \lambda_i, \qquad \operatorname{tr}(ABC) = \operatorname{tr}(BCA) = \operatorname{tr}(CAB), \qquad \mathbf{x}^\top A\mathbf{x} = \operatorname{tr}(A\mathbf{x}\mathbf{x}^\top)$$

Trace is the sum of the diagonal and also the sum of the eigenvalues; cyclic permutation is legal whenever the shapes still conform, and it lets you move a small factor to the outside and avoid forming a large product.

**Jacobian and the gradients used in backprop.**

$$J_{ij} = \frac{\partial f_i}{\partial x_j}, \qquad \nabla_\mathbf{x}(\mathbf{w}^\top\mathbf{x}) = \mathbf{w}, \qquad \nabla_\mathbf{x}(\mathbf{x}^\top A\mathbf{x}) = (A + A^\top)\mathbf{x}, \qquad \nabla_\mathbf{x}\|A\mathbf{x} - \mathbf{b}\|_2^2 = 2A^\top(A\mathbf{x} - \mathbf{b})$$

The Jacobian of $f:\mathbb{R}^m \to \mathbb{R}^n$ is $n \times m$; the quadratic-form gradient simplifies to $2A\mathbf{x}$ when $A$ is symmetric, and the third result is the one backprop uses for every linear layer under squared error.

## Code from memory

Power iteration for the dominant eigenpair, checked against `numpy.linalg.eig`.

```python
import numpy as np

def power_iteration(A, steps=500, tol=1e-12):
    n = A.shape[0]
    v = np.ones(n) / np.sqrt(n)
    lam = 0.0
    for _ in range(steps):
        # one matrix-vector product, written as explicit loops
        w = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i, j] * v[j]
            w[i] = s
        norm = np.sqrt(sum(w[i] ** 2 for i in range(n)))
        w = w / norm
        # Rayleigh quotient gives the eigenvalue for the current vector
        lam_new = sum(w[i] * sum(A[i, j] * w[j] for j in range(n)) for i in range(n))
        if abs(lam_new - lam) < tol:
            v, lam = w, lam_new
            break
        v, lam = w, lam_new
    return lam, v

rng = np.random.default_rng(0)
B = rng.normal(size=(5, 5))
A = B @ B.T                                   ## symmetric, so eigenvalues are real

lam, v = power_iteration(A)
ev, EV = np.linalg.eig(A)
k = np.argmax(np.abs(ev))
print("power iteration lambda =", round(lam, 8))
print("numpy.linalg.eig  lambda =", round(float(ev[k].real), 8))
print("eigenvector alignment |v.u| =", round(abs(float(v @ EV[:, k].real)), 10))
```

Output: power iteration gives `8.07932277` and `numpy.linalg.eig` gives `8.07932277`, an exact match to eight decimals, and the eigenvector alignment is `1.0`. Convergence rate is $|\lambda_2/\lambda_1|$ per step, so a near-tie between the top two eigenvalues makes it slow.

PCA from scratch, checked against `sklearn.decomposition.PCA` on the explained-variance ratio.

```python
import numpy as np
from sklearn.decomposition import PCA

def pca_from_scratch(X, k):
    n, d = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu                                   ## centre: PCA is undefined without this
    C = (Xc.T @ Xc) / (n - 1)                     ## covariance, d x d, symmetric PSD
    vals, vecs = np.linalg.eigh(C)                ## eigh: ascending order, orthonormal vecs
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    Z = Xc @ vecs[:, :k]                          ## project onto top-k eigenvectors
    ratio = vals[:k] / vals.sum()
    return Z, vals, ratio

rng = np.random.default_rng(1)
L = rng.normal(size=(6, 3))
X = rng.normal(size=(400, 3)) @ L.T + 5.0         ## 6-D data with 3-D structure, off-centre

Z, vals, ratio = pca_from_scratch(X, 3)
sk = PCA(n_components=3).fit(X)
print("mine   ", np.round(ratio, 6))
print("sklearn", np.round(sk.explained_variance_ratio_, 6))
print("max abs diff", float(np.max(np.abs(ratio - sk.explained_variance_ratio_))))
```

Output: both print `[0.653702 0.281613 0.064685]`, with maximum absolute difference `1.08e-14`. Use `eigh` and not `eig` on a covariance matrix, because `eigh` exploits symmetry and returns real ordered eigenvalues.

Least squares three ways on well-conditioned and then ill-conditioned data.

```python
import numpy as np

def ls_normal(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)      ## normal equation

def ls_svd(A, b, rcond=1e-12):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.array([1.0 / si if si > rcond * s[0] else 0.0 for si in s])
    return Vt.T @ (s_inv * (U.T @ b))             ## pseudo-inverse A+ = V S+ U^T

def compare(A, b, tag):
    w1 = ls_normal(A, b)
    w2 = np.linalg.lstsq(A, b, rcond=None)[0]
    w3 = ls_svd(A, b)
    print(tag, "cond(A) = %.3e  cond(A^T A) = %.3e" % (np.linalg.cond(A), np.linalg.cond(A.T @ A)))
    print("  normal ", np.round(w1, 6))
    print("  lstsq  ", np.round(w2, 6))
    print("  svd    ", np.round(w3, 6))
    print("  max|normal-svd| = %.3e   max|lstsq-svd| = %.3e"
          % (np.max(np.abs(w1 - w3)), np.max(np.abs(w2 - w3))))

rng = np.random.default_rng(2)
A = rng.normal(size=(50, 3)); w_true = np.array([1.0, -2.0, 0.5]); b = A @ w_true
compare(A, b, "well-conditioned:")

## ill-conditioned: third column is nearly a copy of the first
A2 = A.copy(); A2[:, 2] = A2[:, 0] + 1e-7 * rng.normal(size=50)
b2 = A2 @ w_true
compare(A2, b2, "ill-conditioned: ")
```

Output. Well-conditioned, $\kappa_2(A) = 1.158$ and $\kappa_2(A^\top A) = 1.342$: all three return `[1. -2. 0.5]` and the largest pairwise difference is `6.66e-16`, which is machine precision. Ill-conditioned, $\kappa_2(A) = 2.147 \times 10^{7}$ and $\kappa_2(A^\top A) = 4.358 \times 10^{14}$: the normal equation returns `[0.911663 -2. 0.588337]`, wrong in the first and third coefficients by `8.83e-02`, while `lstsq` and the SVD pseudo-inverse both still return `[1. -2. 0.5]` and agree with each other to `1.46e-09`. The squared condition number has eaten roughly half the available digits.

Gram-Schmidt with explicit loops, verified by $Q^\top Q = I$.

```python
import numpy as np

def gram_schmidt(A):
    n, d = A.shape
    Q = np.zeros((n, d))
    for j in range(d):
        v = A[:, j].copy()
        # subtract the projection onto every earlier q, one at a time (modified GS)
        for i in range(j):
            r = 0.0
            for k in range(n):
                r += Q[k, i] * v[k]
            for k in range(n):
                v[k] -= r * Q[k, i]
        norm = np.sqrt(sum(v[k] ** 2 for k in range(n)))
        if norm < 1e-12:
            raise ValueError("column %d is linearly dependent" % j)
        Q[:, j] = v / norm
    return Q

rng = np.random.default_rng(4)
A = rng.normal(size=(8, 5))
Q = gram_schmidt(A)
G = Q.T @ Q
dev = float(np.max(np.abs(G - np.eye(5))))
print("max |Q^T Q - I| =", "%.3e" % dev)
print("orthonormal to 1e-12:", dev < 1e-12)
print("span preserved (residual of A on Q):",
      "%.3e" % float(np.max(np.abs(Q @ (Q.T @ A) - A))))
```

Output: `max |Q^T Q - I| = 2.220e-16`, so the columns are orthonormal to machine precision, and the residual of $A$ against its own projection $QQ^\top A$ is `6.661e-16`, so the span is preserved. Subtracting each projection as it is computed, rather than all at once from the original column, is modified Gram-Schmidt and it is markedly more stable.

## Questions

### Q1. What does rank mean in plain words, and why is a low-rank matrix compressible?

Rank is the number of independent directions the matrix actually uses. If $A$ is $n \times m$ with rank $r$, then every column is a linear combination of the same $r$ basis columns, so the other $m - r$ columns carry no new information. That is why rank-$r$ means compressible: you can write $A = BC$ with $B \in \mathbb{R}^{n \times r}$ and $C \in \mathbb{R}^{r \times m}$, so storage drops from $nm$ numbers to $r(n + m)$ numbers. At $n = m = 4096$ and $r = 16$ that is 16.8 million numbers down to 131 thousand, a factor of 128. LoRA uses exactly this. It freezes the pretrained weight $W_0$ and learns an update $\Delta W = BA$ with inner dimension $r$, on the empirical claim that the useful fine-tuning update has low intrinsic rank. You train $r(n+m)$ parameters instead of $nm$, and at inference you can fold $W_0 + BA$ into one matrix, so there is no added latency.

> **Say it.** Rank is how many independent directions a matrix really uses. If the rank is $r$, every column is a combination of the same $r$ basis columns, so you can factor the matrix as an $n$ by $r$ times an $r$ by $m$ and store $r(n+m)$ numbers instead of $nm$. For a 4096-square matrix at rank 16 that is a 128-fold saving. LoRA is this idea applied to fine-tuning: freeze the pretrained weight and learn a low-rank update $BA$, betting that the useful update has low intrinsic rank. At inference you fold it back in, so no extra latency.

### Q2. What does the SVD give you geometrically?

Rotate, scale, rotate. Write $A = U\Sigma V^\top$. Applying $A$ to a vector does three things in order. First $V^\top$ rotates, or rotates and reflects, the input into a new orthonormal frame; it changes the axes but not any length. Then $\Sigma$ is diagonal, so it stretches each new axis $i$ by the singular value $\sigma_i \ge 0$; this is the only step that changes size, and a zero $\sigma_i$ collapses that axis entirely. Then $U$ rotates the result into the output space. So every linear map, of any shape, is a rotation, then an axis-aligned scaling, then another rotation. The image of the unit sphere under $A$ is an ellipsoid whose semi-axis lengths are the singular values and whose axis directions are the columns of $U$. The number of nonzero $\sigma_i$ is the rank, $\sigma_{\max}$ is the spectral norm, and the columns of $V$ with $\sigma_i = 0$ span the null space.

> **Say it.** Rotate, scale, rotate. $V^\top$ turns the input into a new orthonormal frame without changing any length, $\Sigma$ stretches each of those axes by its singular value, and $U$ rotates into the output space. So every linear map, square or not, is a rotation, an axis-aligned scaling, and another rotation. The unit sphere goes to an ellipsoid with semi-axes equal to the singular values along the columns of $U$. The count of nonzero singular values is the rank, the largest is the spectral norm, and the right singular vectors with zero singular value span the null space.

### Q3. Why is SVD preferred over the normal equation for least squares?

Because forming $A^\top A$ squares the condition number. The relative error in a solved linear system scales with the condition number of the matrix you solve, and $\kappa_2(A^\top A) = \kappa_2(A)^2$ because the eigenvalues of the Gram matrix are the squared singular values of $A$. In double precision you have about 16 decimal digits. If $\kappa_2(A) = 10^{8}$, the SVD route loses about 8 digits and keeps 8, while the normal equation solves a system with $\kappa = 10^{16}$ and keeps essentially none. My third code block shows this: at $\kappa_2(A) = 2.1 \times 10^{7}$ and $\kappa_2(A^\top A) = 4.4 \times 10^{14}$, the normal equation is wrong by $8.8 \times 10^{-2}$ in the coefficients while the SVD route is still correct to $1.5 \times 10^{-9}$. The SVD also degrades gracefully when $A$ is rank-deficient: you truncate the tiny singular values and get the minimum-norm solution, whereas $A^\top A$ is then singular and the solve simply fails.

> **Say it.** Because $A^\top A$ squares the condition number: the eigenvalues of the Gram matrix are the squared singular values, so $\kappa$ becomes $\kappa^2$. Double precision gives you sixteen digits, so at $\kappa$ of ten to the eight the SVD keeps eight digits and the normal equation keeps none. I measured it: at $\kappa(A)$ of two times ten to the seven the normal equation was off by nine parts in a hundred while the SVD was correct to a billionth. SVD also handles rank deficiency by truncating small singular values and returning the minimum-norm solution, where the Gram matrix is just singular.

### Q4. What are the eigenvectors of a covariance matrix, and why is PCA exactly that eigendecomposition?

The covariance matrix $C = \frac{1}{n-1}X_c^\top X_c$ is symmetric and positive semi-definite, so it has real non-negative eigenvalues and orthonormal eigenvectors. The variance of the data projected onto a unit direction $\mathbf{w}$ is exactly $\mathbf{w}^\top C\mathbf{w}$. PCA asks for the unit direction of maximum projected variance, so it maximises $\mathbf{w}^\top C\mathbf{w}$ subject to $\mathbf{w}^\top\mathbf{w} = 1$. Form the Lagrangian $\mathbf{w}^\top C\mathbf{w} - \lambda(\mathbf{w}^\top\mathbf{w} - 1)$ and set the gradient to zero: $2C\mathbf{w} - 2\lambda\mathbf{w} = 0$, which is $C\mathbf{w} = \lambda\mathbf{w}$. So the stationary points are the eigenvectors and the objective value at each is $\mathbf{w}^\top C\mathbf{w} = \lambda$. The maximum is therefore the top eigenvector, and its eigenvalue is the variance it captures. The next component repeats the problem restricted to the orthogonal complement, which gives the second eigenvector. Explained-variance ratio is $\lambda_i / \sum_j \lambda_j$.

> **Say it.** The covariance is symmetric and positive semi-definite, so it has real eigenvalues and orthonormal eigenvectors. The projected variance along a unit direction $\mathbf{w}$ is the quadratic form $\mathbf{w}^\top C\mathbf{w}$, so PCA maximises that under a unit-norm constraint. Take the Lagrangian, differentiate, and you get $C\mathbf{w} = \lambda\mathbf{w}$ directly. So the stationary directions are the eigenvectors and the objective at each equals its eigenvalue. The top eigenvector is the first component and its eigenvalue is the variance it captures; each later component is the same problem in the orthogonal complement.

### Q5. How does PCA relate to the SVD of the centred data matrix?

They are the same computation. Take the SVD of the centred data, $X_c = U\Sigma V^\top$. Then $C = \frac{1}{n-1}X_c^\top X_c = \frac{1}{n-1}V\Sigma^\top\Sigma V^\top$. That is already an eigendecomposition of $C$: the right singular vectors $V$ are the principal directions, and the eigenvalues are $\lambda_i = \sigma_i^2/(n-1)$. The projected scores are $X_cV = U\Sigma$, so you never need $C$ at all. In practice you always take the SVD route, for three reasons. It avoids forming $X_c^\top X_c$, which squares the condition number and loses half your digits. It costs $O(nd\min(n,d))$ rather than $O(nd^2)$ to form the covariance plus $O(d^3)$ to decompose it, which matters when $d$ is large. And when $d \gg n$, as with text features, $C$ is $d \times d$ and may not even fit in memory while $X_c$ does. That is why `sklearn.decomposition.PCA` calls an SVD internally.

> **Say it.** They are the same thing. If the centred data has SVD $U\Sigma V^\top$, then the covariance is $V\Sigma^2V^\top$ over $n-1$, which is already its eigendecomposition. So the right singular vectors are the principal directions, the eigenvalues are the squared singular values over $n-1$, and the scores are $U\Sigma$. You always take the SVD route, because forming the covariance squares the condition number, costs more when $d$ is large, and needs a $d$ by $d$ matrix that may not fit when features outnumber samples. That is what sklearn does internally.

### Q6. What does positive semi-definite mean, and where does it show up in ML?

A symmetric matrix $A$ is positive semi-definite when $\mathbf{x}^\top A\mathbf{x} \ge 0$ for every $\mathbf{x}$. Two equivalent statements: all eigenvalues are non-negative, and $A$ can be written as $B^\top B$. Geometrically the quadratic form is a bowl that never dips below zero, though it may be flat in some directions. Three places it appears. A covariance matrix is $\frac{1}{n-1}X_c^\top X_c$, which is $B^\top B$ by construction, so it is PSD; this is why variance along any direction is never negative. A kernel matrix must be PSD by Mercer's condition, because $K_{ij} = \phi(x_i)^\top\phi(x_j)$ is a Gram matrix; if your similarity matrix is not PSD there is no feature space behind it and the SVM dual is not convex. The Hessian is PSD at any local minimum, which is the second-order condition, and a function whose Hessian is PSD everywhere is convex. Positive definite, with strictly positive eigenvalues, additionally means invertible and a strict minimum.

> **Say it.** Positive semi-definite means the quadratic form $\mathbf{x}^\top A\mathbf{x}$ is never negative, equivalently all eigenvalues are at least zero, equivalently the matrix factors as $B^\top B$. Covariance matrices are PSD because they are literally a Gram matrix, which is why no direction has negative variance. Kernel matrices must be PSD, because that is what guarantees a feature space exists and keeps the SVM dual convex. And the Hessian is PSD at any local minimum; if it is PSD everywhere the function is convex. Strictly positive eigenvalues give you invertibility and a strict minimum.

### Q7. What does the condition number tell you about optimisation difficulty?

For a quadratic objective the Hessian $H$ is constant and the level sets are ellipsoids with axis lengths set by $1/\sqrt{\lambda_i}$. The condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ is how elongated that bowl is. Gradient descent is stable only when the step size satisfies $\eta < 2/\lambda_{\max}$, because the steepest direction diverges otherwise. However, progress along the flattest direction goes as $(1 - \eta\lambda_{\min})$ per step. So the largest eigenvalue caps the learning rate while the smallest eigenvalue sets how fast you actually move, and the error contracts by roughly $\frac{\kappa - 1}{\kappa + 1}$ per step. The number of iterations to a fixed accuracy therefore scales linearly with $\kappa$. At $\kappa = 10^4$ you need on the order of ten thousand steps where a well-conditioned problem needs ten. The practical fixes all reduce $\kappa$: feature normalisation, batch or layer normalisation, momentum which gives $\sqrt{\kappa}$ instead of $\kappa$, and per-coordinate scaling as in Adam.

> **Say it.** The condition number is the ratio of the largest to the smallest Hessian eigenvalue, and it is how elongated the loss bowl is. The largest eigenvalue caps your learning rate at two over $\lambda_{\max}$ for stability, and the smallest eigenvalue sets how fast you move along the flat direction, so the contraction per step is about $\kappa-1$ over $\kappa+1$ and iterations scale linearly with $\kappa$. At a condition number of ten thousand you need thousands of steps for what a round bowl does in ten. That is why we normalise features, use normalisation layers, momentum, and per-coordinate scaling in Adam.

### Q8. Why does a matrix multiply cost $O(nmp)$, and how does that give the transformer FLOP count?

For $A \in \mathbb{R}^{n \times m}$ times $B \in \mathbb{R}^{m \times p}$ there are $np$ output entries, and each is a dot product of length $m$, so $nmp$ multiply-accumulates or about $2nmp$ floating-point operations. Apply that to a transformer. A linear layer mapping $d_{\text{in}}$ to $d_{\text{out}}$ over $T$ tokens is a $T \times d_{\text{in}}$ by $d_{\text{in}} \times d_{\text{out}}$ product, so $2Td_{\text{in}}d_{\text{out}}$ FLOPs, which is $2T$ times the parameter count of that layer. Summing over all weight matrices gives the standard rule: forward pass costs about $2N$ FLOPs per token for $N$ parameters. The backward pass costs about twice the forward, because it computes a gradient with respect to both the input and the weights, so training costs about $6N$ per token. Attention adds a term the parameter count does not cover: the $QK^\top$ and score-times-$V$ products cost about $4T^2d$ per layer, which is quadratic in sequence length and dominates at long context.

> **Say it.** An $n$ by $m$ times $m$ by $p$ product has $np$ outputs, each a length-$m$ dot product, so $nmp$ multiply-accumulates and about $2nmp$ FLOPs. A transformer linear layer over $T$ tokens is $2T$ times that layer's parameter count, so the forward pass is roughly $2N$ FLOPs per token for $N$ parameters, the backward is twice that, and training is about $6N$ per token. Attention is the exception, because the score matrix is $T$ by $T$: those two products cost about $4T^2d$ per layer, which is quadratic in sequence length and takes over at long context.

### Q9. What does the trace trick buy you?

Two things. First, cyclic permutation, $\operatorname{tr}(ABC) = \operatorname{tr}(BCA)$, lets you reorder a product so the largest intermediate matrix is never formed. If $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{d}$, then $\operatorname{tr}(\mathbf{u}\mathbf{v}^\top) = \mathbf{v}^\top\mathbf{u}$: the left side builds a $d \times d$ outer product for $O(d^2)$ and the right side is $O(d)$. Similarly $\operatorname{tr}(X^\top AX)$ with $X$ tall is cheaper as $\operatorname{tr}(AXX^\top)$ or the other way round, depending on which dimension is smaller. Second, it turns scalars into traces so matrix calculus applies. Any scalar equals its own trace, so $\mathbf{x}^\top A\mathbf{x} = \operatorname{tr}(A\mathbf{x}\mathbf{x}^\top)$, and then the standard identity $\frac{\partial}{\partial A}\operatorname{tr}(AB) = B^\top$ gives derivatives with no index bookkeeping. It also gives identities you use constantly: $\|A\|_F^2 = \operatorname{tr}(A^\top A)$, $\operatorname{tr}(A) = \sum_i\lambda_i$, and the Gaussian log-likelihood's quadratic term written as a trace against the sample covariance.

> **Say it.** Two things. Cyclic permutation lets you reorder a product to avoid forming the big intermediate: the trace of an outer product $\mathbf{u}\mathbf{v}^\top$ is just $\mathbf{v}^\top\mathbf{u}$, which is order $d$ instead of $d$ squared. And because any scalar equals its own trace, you can rewrite a quadratic form as a trace and then use matrix-calculus identities like the derivative of trace $AB$ being $B$ transposed, so you differentiate without index gymnastics. It also gives the identities you use daily: Frobenius norm squared is trace of $A^\top A$, and trace equals the sum of eigenvalues.

### Q10. What is the difference between an orthogonal and an orthonormal basis, and why does orthogonality help numerically?

An orthogonal set has vectors that are mutually perpendicular, $\mathbf{v}_i^\top\mathbf{v}_j = 0$ for $i \neq j$, but with arbitrary lengths. An orthonormal set adds unit length, $\|\mathbf{v}_i\|_2 = 1$, so the Gram matrix is exactly the identity. A square matrix $Q$ with orthonormal columns satisfies $Q^\top Q = I$, so $Q^{-1} = Q^\top$ and the inverse is free. Orthonormality is what makes things numerically safe. Coefficients come from a single dot product, $c_i = \mathbf{q}_i^\top\mathbf{x}$, with no linear solve. All singular values of $Q$ are 1, so $\kappa_2(Q) = 1$ and multiplying by $Q$ neither amplifies error nor loses it. Lengths and angles are preserved, $\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2$, so $Q$ is a rotation or reflection. That is why stable algorithms are built from orthogonal transforms: QR by Householder reflections, and the SVD itself. In deep learning it is why orthogonal weight initialisation helps in deep or recurrent stacks, since repeated multiplication neither explodes nor vanishes.

> **Say it.** Orthogonal means mutually perpendicular; orthonormal adds unit length, so the Gram matrix is exactly the identity and $Q^{-1}$ equals $Q^\top$. That gives you three numerical wins. Coefficients are a single dot product with no solve. Every singular value is one, so the condition number is one and multiplying by $Q$ neither amplifies nor loses error. And lengths and angles are preserved, so $Q$ is a pure rotation or reflection. That is why QR and the SVD are built from orthogonal transforms, and why orthogonal initialisation helps very deep or recurrent networks.

### Q11. Why do we normalise embeddings before cosine similarity, and what happens if we do not?

Cosine similarity is defined as $\frac{\mathbf{a}^\top\mathbf{b}}{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}$, so the normalisation is part of the definition. If you divide each vector by its own L2 norm once, in advance, then the raw dot product is already the cosine, which is the point: dot products are a single fast matrix multiply and every vector database, FAISS included, is built around inner-product search. If you skip normalisation and use the raw dot product, you are ranking by $\|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$, so long vectors win regardless of direction. Embedding norms correlate with things you did not want to rank on, such as token count, word frequency, and how typical the text is, so retrieval starts returning long or generic passages instead of relevant ones. The failure is quiet, because the results still look plausible. Two notes: normalisation must use the same convention at index time and query time, and after normalisation cosine and squared Euclidean distance are monotonically related, $\|\mathbf{a} - \mathbf{b}\|_2^2 = 2 - 2\cos\theta$, so they rank identically.

> **Say it.** Because the cosine is the dot product divided by both norms, so if you normalise once up front, the plain dot product is already the cosine, and that is what inner-product search in a vector database computes. Skip it and you rank by norm times norm times cosine, so long vectors win on length alone. Embedding norms track passage length, word frequency and typicality, so you quietly start retrieving long generic chunks instead of relevant ones. Normalise at index time and query time the same way. After normalising, squared Euclidean distance is two minus twice the cosine, so the two rankings agree.

### Q12. What is a projection matrix, and what is the idempotence property?

A projection matrix $P$ maps any vector onto a subspace and leaves vectors already in that subspace untouched. That second requirement is idempotence: $P^2 = P$, because projecting something that is already projected changes nothing. An orthogonal projection additionally satisfies $P^\top = P$, which means the residual $\mathbf{x} - P\mathbf{x}$ is perpendicular to the subspace. For a subspace spanned by the columns of $A$ with full column rank, $P = A(A^\top A)^{-1}A^\top$, and if the columns are orthonormal, as in $Q$, this collapses to $P = QQ^\top$. The eigenvalues of a projection are only 0 and 1, so the trace equals the rank, which equals the dimension of the subspace. This is exactly least squares: $\hat{\mathbf{b}} = P\mathbf{b}$ is the point in the column space of $A$ closest to $\mathbf{b}$, and the normal equation is just the statement that the residual is orthogonal to every column, $A^\top(A\mathbf{w} - \mathbf{b}) = \mathbf{0}$. Attention heads, PCA reconstruction, and gradient projection all use the same object.

> **Say it.** A projection maps a vector onto a subspace and leaves anything already in that subspace alone, which is exactly idempotence: $P^2 = P$. If it is also symmetric it is an orthogonal projection, so the residual is perpendicular to the subspace. For the column space of $A$ it is $A(A^\top A)^{-1}A^\top$, and for orthonormal columns it collapses to $QQ^\top$. Eigenvalues are only zero and one, so the trace equals the rank. Least squares is precisely this: the fitted values are the projection of $\mathbf{b}$ onto the column space, and the normal equation says the residual is orthogonal to every column.

### Q13. What does a determinant of zero mean?

It means the matrix collapses volume to zero, so it is singular. Equivalently, and these are all the same fact: the columns are linearly dependent, the rank is less than $d$, the null space contains a nonzero vector, at least one eigenvalue is zero, at least one singular value is zero, and no inverse exists. Geometrically, $|\det(A)|$ is the factor by which $A$ scales the volume of any region, so a determinant of zero means the unit cube is flattened into a lower-dimensional slab with no volume. The map is not injective: distinct inputs land on the same output, so it cannot be undone. In practice you almost never test $\det(A) = 0$, for two reasons. The determinant is a product of $d$ numbers, so it underflows or overflows badly at large $d$, and it is scale dependent since $\det(cA) = c^d\det(A)$. Use the condition number or the smallest singular value instead; those tell you how close to singular you are, which is the question that actually matters.

> **Say it.** Zero determinant means the matrix squashes volume to nothing, so it is singular. Equivalently the columns are dependent, the rank is deficient, the null space is nontrivial, an eigenvalue is zero, a singular value is zero, and there is no inverse. The unit cube gets flattened into something with no volume, so the map cannot be undone. In practice never test the determinant against zero: it is a product of $d$ numbers so it overflows or underflows, and it scales as $c^d$. Look at the smallest singular value or the condition number, which tell you how near-singular you are.

### Q14. What is the relation between the Frobenius norm and the singular values?

$\|A\|_F^2 = \sum_{ij}A_{ij}^2 = \operatorname{tr}(A^\top A) = \sum_i \sigma_i^2$. The middle step is just the definition of the trace of the Gram matrix, and the last step holds because the trace equals the sum of eigenvalues and the eigenvalues of $A^\top A$ are the squared singular values. Two consequences. First, the Frobenius norm is invariant under orthogonal transforms, since $\|UAV^\top\|_F = \|A\|_F$; rotations do not change total energy. Second, this is what makes truncated SVD optimal. The Eckart-Young theorem says the best rank-$k$ approximation in Frobenius norm is $A_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$, and the error is exactly $\|A - A_k\|_F^2 = \sum_{i>k}\sigma_i^2$, the tail energy you discarded. So the fraction of energy retained is $\sum_{i \le k}\sigma_i^2 / \sum_i \sigma_i^2$, which is the same quantity PCA calls the explained-variance ratio. Compare the spectral norm, $\|A\|_2 = \sigma_{\max}$, which sees only the largest singular value.

> **Say it.** The Frobenius norm squared is the sum of all squared entries, which equals the trace of $A^\top A$, which equals the sum of the squared singular values. So it is invariant under rotations, because orthogonal transforms do not change total energy. It also explains truncated SVD: by Eckart-Young the best rank-$k$ approximation is the top $k$ singular triplets, and the squared Frobenius error is exactly the sum of the discarded squared singular values. That energy ratio is the same number PCA reports as explained variance. The spectral norm, by contrast, only sees the largest singular value.

### Q15. Derive $\nabla_\mathbf{x}\|A\mathbf{x} - \mathbf{b}\|_2^2$ step by step.

Write the residual $\mathbf{r} = A\mathbf{x} - \mathbf{b}$ and expand the squared norm as an inner product:

$$f(\mathbf{x}) = \mathbf{r}^\top\mathbf{r} = (A\mathbf{x} - \mathbf{b})^\top(A\mathbf{x} - \mathbf{b}) = \mathbf{x}^\top A^\top A\mathbf{x} - 2\mathbf{b}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{b}$$

The two cross terms combined because $\mathbf{x}^\top A^\top\mathbf{b}$ is a scalar, so it equals its own transpose $\mathbf{b}^\top A\mathbf{x}$. Now differentiate term by term. The first term is a quadratic form $\mathbf{x}^\top M\mathbf{x}$ with $M = A^\top A$, whose gradient is $(M + M^\top)\mathbf{x}$; here $M$ is symmetric, so that is $2A^\top A\mathbf{x}$. The second term is linear, $\mathbf{w}^\top\mathbf{x}$ with $\mathbf{w} = A^\top\mathbf{b}$, whose gradient is $\mathbf{w}$, giving $-2A^\top\mathbf{b}$. The third term is constant. Add them:

$$\nabla_\mathbf{x} f = 2A^\top A\mathbf{x} - 2A^\top\mathbf{b} = 2A^\top(A\mathbf{x} - \mathbf{b}) = 2A^\top\mathbf{r}$$

Check the shapes: $A^\top$ is $m \times n$ and $\mathbf{r}$ is $n$, so the gradient is $m$-dimensional like $\mathbf{x}$. Setting it to zero gives the normal equation $A^\top A\mathbf{x} = A^\top\mathbf{b}$.

> **Say it.** Expand the squared norm as $\mathbf{x}^\top A^\top A\mathbf{x}$ minus $2\mathbf{b}^\top A\mathbf{x}$ plus a constant; the cross terms merge because a scalar equals its own transpose. The quadratic form differentiates to $M$ plus $M^\top$ times $\mathbf{x}$, and $A^\top A$ is symmetric, so that is $2A^\top A\mathbf{x}$. The linear term gives minus $2A^\top\mathbf{b}$. Add them and factor: $2A^\top(A\mathbf{x} - \mathbf{b})$, which is features transposed against the residual. Shapes check, since $A^\top$ is $m$ by $n$ and the residual is $n$. Set it to zero and you have the normal equation.

## Done when

- You can write the SVD, say rotate-scale-rotate, and state in one sentence why the right singular vectors are eigenvectors of $A^\top A$ with $\sigma_i = \sqrt{\lambda_i}$.
- You can code power iteration, PCA from scratch, and Gram-Schmidt in NumPy from memory, and each runs first try and matches the library.
- You can give the condition-number argument against the normal equation in under a minute, including the squaring and the digit count in double precision.
- You can derive $\nabla_\mathbf{x}\|A\mathbf{x} - \mathbf{b}\|_2^2 = 2A^\top(A\mathbf{x} - \mathbf{b})$ on a whiteboard, and say why the Hessian eigenvalue ratio caps the learning rate.
