# Linear Algebra for ML — Interview Grill

> 50 questions on rank, eigendecomp, SVD, PSD, matrix calculus, conditioning, projections. Drill until you can answer 35+ cold.

---

## A. Rank and subspaces

**1. Define rank of a matrix.**
Dimension of the column space (= dimension of the row space). Equivalently, number of linearly independent rows or columns.

> **Saying it out loud.** Rank is how many independent directions a matrix actually spans. Think of the matrix as a machine that eats vectors and spits out vectors: the rank is the dimension of the space of things it can produce. If you have a hundred columns but they're all combinations of three underlying patterns, the rank is three and the other ninety-seven are redundant. That's exactly why low-rank structure is everywhere in machine learning, from PCA to LoRA.

**2. State the rank-nullity theorem.**
For $A \in \mathbb{R}^{m \times n}$: $\mathrm{rank}(A) + \dim(\mathrm{Null}(A)) = n$.

> **Saying it out loud.** Rank-nullity says the input dimensions have to go somewhere: each one either survives into the output or gets crushed to zero, and the two counts add up to the number of columns. So rank plus nullity equals $n$, always. Practically, a nullspace of dimension bigger than zero means there are directions your model literally cannot see, which is exactly what happens with collinear features. It's a conservation law for dimensions.

**3. What does row rank = column rank mean intuitively?**
A counterintuitive fact. Both give the same number; this is a deep theorem proved via SVD or RREF arguments.

> **Saying it out loud.** Row rank equals column rank means the number of independent rows and the number of independent columns always come out the same, which is not obvious at all. Look at a ten-by-three matrix: the rows live in three-dimensional space and the columns live in ten-dimensional space, completely different worlds, yet the counts match. The reason is that both equal the number of nonzero singular values in the SVD. It's the cleanest example of a fact that seems like a coincidence until you see the right decomposition.

**4. Inequality for $\mathrm{rank}(AB)$?**
$\mathrm{rank}(AB) \leq \min(\mathrm{rank}(A), \mathrm{rank}(B))$.

> **Saying it out loud.** The rank of a product can't exceed the rank of either factor. Intuitively, if $B$ already collapses everything into a three-dimensional space, nothing $A$ does afterward can create new dimensions; you can only lose information, never manufacture it. That inequality is the entire mathematical basis for LoRA: multiplying a $d$-by-$r$ by an $r$-by-$d$ matrix guarantees the update has rank at most $r$, no matter what you train.

**5. When is $X^\top X$ invertible?**
When $X$ has full column rank (columns linearly independent).

> **Saying it out loud.** $X^\top X$ is invertible exactly when $X$ has full column rank, that is, no feature is a linear combination of the others. If two of your columns are duplicates or one is a sum of two others, you get a singular matrix and OLS has infinitely many solutions. The connection worth stating: the nullspace of $X^\top X$ is the same as the nullspace of $X$, so this is really a statement about your features being independent. With more features than samples, it's automatically singular.

**6. What if $X^\top X$ is singular in OLS?**
Use pseudoinverse, or add ridge ($X^\top X + \lambda I$), or remove redundant columns.

> **Saying it out loud.** If $X^\top X$ is singular, the normal equations have infinitely many solutions and you have to pick one. The pseudoinverse picks the minimum-norm solution, which is a clean, principled default. Ridge, adding $\lambda I$, makes the matrix invertible again and shrinks the solution, which is what people actually do in practice. Or you find the collinear columns and drop them. Ridge is usually the right answer because near-singularity is far more common than exact singularity, and ridge handles both.

**7. What's the four fundamental subspaces?**
$\mathrm{Col}(A)$, $\mathrm{Null}(A)$, $\mathrm{Row}(A) = \mathrm{Col}(A^\top)$, $\mathrm{Null}(A^\top)$. $\mathrm{Col}(A) \perp \mathrm{Null}(A^\top)$, $\mathrm{Row}(A) \perp \mathrm{Null}(A)$.

> **Saying it out loud.** There are four subspaces and they pair up into two orthogonal complements. The column space, everything the matrix can output, is perpendicular to the nullspace of the transpose. The row space is perpendicular to the nullspace. Every vector in the input space splits uniquely into a row-space part, which gets mapped somewhere, and a nullspace part, which gets annihilated. This is the picture underneath least squares: you project your target onto the column space because that's the only place the model can reach.

---

## B. Eigendecomposition

**8. Define eigenvalue and eigenvector.**
$Av = \lambda v$ with $v \neq 0$. $\lambda$ is the eigenvalue, $v$ the eigenvector.

> **Saying it out loud.** An eigenvector is a direction that the matrix doesn't rotate, only stretches. Feed it in, get the same direction back, just scaled by the eigenvalue. Everything else gets turned as well as stretched, so eigenvectors are the special axes where the transformation is simplest. That's why they show up everywhere: they're the coordinate system in which a complicated linear map becomes plain multiplication.

**9. How do you find eigenvalues?**
Roots of characteristic polynomial: $\det(A - \lambda I) = 0$.

> **Saying it out loud.** Formally, you solve the characteristic polynomial, setting the determinant of $A - \lambda I$ to zero, because $\lambda$ being an eigenvalue means $A - \lambda I$ crushes some nonzero vector, which means it's singular, which means its determinant vanishes. That's the right answer for a two-by-two on a whiteboard. In practice nobody ever does this beyond three-by-three, because polynomial roots are numerically horrible; real code uses the QR algorithm. Saying both is the good answer.

**10. State the spectral theorem.**
Real symmetric matrix $A$ has $n$ real eigenvalues and an orthonormal basis of eigenvectors. $A = Q\Lambda Q^\top$ with $Q$ orthogonal.

> **Saying it out loud.** The spectral theorem says that if a real matrix is symmetric, it has all real eigenvalues and an orthonormal basis of eigenvectors, so you can write it as an orthogonal matrix times a diagonal times that orthogonal matrix transposed. In plain terms, every symmetric matrix is just a rotation, a stretch along the new axes, and a rotation back. That's why covariance matrices and Hessians are so tractable: they're symmetric, so they always have clean perpendicular axes.

**11. Why are eigenvectors of distinct eigenvalues orthogonal (for symmetric $A$)?**
$\lambda_1 v_1^\top v_2 = (A v_1)^\top v_2 = v_1^\top A v_2 = \lambda_2 v_1^\top v_2$. If $\lambda_1 \neq \lambda_2$, must have $v_1^\top v_2 = 0$.

> **Saying it out loud.** Sandwich the matrix between the two eigenvectors and evaluate it two ways. Acting leftward gives you $\lambda_1$ times the inner product, acting rightward gives $\lambda_2$ times the same inner product, using symmetry to move $A$ across. So $(\lambda_1 - \lambda_2)$ times the inner product is zero, and if the eigenvalues differ, the inner product must be zero. It's a three-line proof and it's a common ask, so it's worth being able to write it without thinking.

**12. Which matrices are NOT diagonalizable?**
Defective matrices — those without a full set of linearly independent eigenvectors. E.g., $\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ has only one eigenvector (up to scaling).

> **Saying it out loud.** Defective matrices, meaning ones without enough independent eigenvectors to form a basis. The canonical example is a two-by-two with ones on the diagonal and a one in the corner: it has eigenvalue one repeated twice but only a single eigenvector direction. Note it's not about repeated eigenvalues by themselves, since the identity matrix has a repeated eigenvalue and is perfectly diagonalizable. It's about geometric multiplicity falling short of algebraic multiplicity, and symmetric matrices are never defective.

**13. Eigenvalues of $A^k$?**
$\lambda^k$ for each eigenvalue $\lambda$ of $A$.

> **Saying it out loud.** Raising the matrix to a power raises each eigenvalue to that power while the eigenvectors stay put. You can see it immediately: apply $A$ twice to an eigenvector and you scale by $\lambda$ twice. This is why the spectral radius controls whether repeated application blows up or dies out, and it's exactly why deep networks suffer exploding or vanishing gradients, since backprop is repeated multiplication by Jacobians.

**14. Eigenvalues of $A^{-1}$?**
$1/\lambda$ for each $\lambda \neq 0$.

> **Saying it out loud.** Inverting the matrix inverts each eigenvalue while keeping the same eigenvectors. Undoing a stretch by three means shrinking by a third along that same direction. Notice what happens if an eigenvalue is nearly zero: its reciprocal is enormous, which is precisely why nearly-singular matrices amplify noise so badly when you invert them. That's the condition number story in one sentence.

**15. What's the spectral radius?**
$\rho(A) = \max_i |\lambda_i|$ — largest absolute eigenvalue. Determines convergence/divergence of $A^k$.

> **Saying it out loud.** The spectral radius is the largest eigenvalue in absolute value, and it decides the long-run behavior of repeated multiplication. Below one, powers of the matrix decay to zero; above one, they explode; right at one, you're on a knife edge. That's the criterion for whether a linear dynamical system is stable, and it's the same math behind why RNN gradients vanish or explode over long sequences.

---

## C. SVD

**16. State the SVD theorem.**
Any $A \in \mathbb{R}^{m \times n}$ factors as $A = U \Sigma V^\top$ with $U, V$ orthogonal and $\Sigma$ diagonal with non-negative singular values.

> **Saying it out loud.** SVD says any matrix at all, square or not, symmetric or not, factors into an orthogonal matrix, a nonnegative diagonal, and another orthogonal matrix transposed. That universality is what makes it the most useful decomposition in applied math: no conditions, no exceptions, it always exists. The singular values come out sorted, so you get a built-in ranking of which directions carry the most action. Everything from PCA to pseudoinverses to low-rank compression falls out of it.

**17. Geometric interpretation of SVD?**
Rotation ($V^\top$) → axis-aligned scaling ($\Sigma$) → rotation ($U$). Any linear map decomposes this way.

> **Saying it out loud.** Geometrically, every linear map is a rotation, then a stretch along the new axes, then another rotation. That's it, that's the whole content of SVD. Take the unit sphere, push it through any matrix, and you always get an ellipsoid; the singular values are the lengths of the ellipsoid's axes and the columns of $U$ point along them. Once you have that picture, the operator norm being the largest singular value is obvious: it's the longest axis of the ellipsoid.

**18. SVD vs eigendecomposition?**
SVD works for any matrix; eigendecomposition only for diagonalizable square matrices. For symmetric PSD, they coincide. SVD = eigendecomposition of $A^\top A$ (or $A A^\top$).

> **Saying it out loud.** SVD works on any matrix; eigendecomposition needs a square matrix and only works if it has a full set of independent eigenvectors. Also SVD's singular values are always real and nonnegative, whereas eigenvalues can be negative or complex. They coincide for symmetric positive semidefinite matrices. And the connection is that the singular values of $A$ are the square roots of the eigenvalues of $A^\top A$, with $V$ holding those eigenvectors.

**19. What's the operator norm of $A$ in terms of SVD?**
Largest singular value: $\|A\|_2 = \sigma_1$.

> **Saying it out loud.** The operator norm is the largest singular value, which is the most a matrix can stretch any unit vector. Geometrically it's the longest axis of that ellipsoid the unit sphere maps onto. This is the number that governs stability: it's why spectral normalization in GANs constrains the top singular value, and why Lipschitz bounds for networks are products of layer operator norms.

**20. Frobenius norm in terms of SVD?**
$\|A\|_F = \sqrt{\sum_i \sigma_i^2}$.

> **Saying it out loud.** The Frobenius norm is the square root of the sum of squared singular values, which is also just the square root of the sum of all squared entries. So it's the Euclidean length of the matrix flattened into a vector, and the SVD gives you the same number a different way. It's the norm in Eckart-Young, and the practical contrast is that the operator norm cares only about the worst-case direction while Frobenius averages over all of them.

**21. How do you compute rank from SVD?**
Number of nonzero singular values (in practice, number greater than some tolerance).

> **Saying it out loud.** Count the nonzero singular values. Numerically nothing is ever exactly zero, so you count the ones above a tolerance, typically machine epsilon times the largest singular value times the matrix dimension. That's what numpy's matrix_rank does. And this is the right way to compute rank in floating point, because Gaussian elimination will happily report full rank on a matrix that's numerically rank-deficient.

**22. State Eckart-Young.**
The truncated SVD $A_k = U_k \Sigma_k V_k^\top$ is the best rank-$k$ approximation in operator and Frobenius norms.

> **Saying it out loud.** Eckart-Young says the best low-rank approximation of a matrix is just its truncated SVD: keep the top $k$ singular values and throw away the rest. It's optimal in both the Frobenius and the operator norm, which is remarkable because those measure error very differently. Low-rank approximation could have been a hard combinatorial problem and instead it has a closed form. The leftover error equals exactly the discarded singular values, so you know your error before you commit.

**23. Why does PCA reduce to SVD?**
Centered data $X$. Covariance $\Sigma_X = X^\top X / n$. Eigendecomp of $\Sigma_X$ = right singular vectors $V$ of $X$. PCA scores = $US$.

> **Saying it out loud.** PCA wants the eigenvectors of the covariance matrix, and the covariance is the centered data transposed times itself, divided by $n$. But the SVD of the centered data already gives you the eigenvectors of that product for free, as the right singular vectors, with eigenvalues equal to the squared singular values over $n$. So you skip building the covariance entirely. The reason it matters is numerical: forming $X^\top X$ squares the condition number and costs you half your precision.

**24. SVD of a low-rank matrix?**
Rank-$r$ matrix has only $r$ nonzero singular values. Truncated SVD with $k=r$ recovers exactly.

> **Saying it out loud.** A rank-$r$ matrix has exactly $r$ nonzero singular values and the rest are zero, so truncating at $k$ equal to $r$ reconstructs it exactly with no error. That's the ideal case. The interesting real-world version is approximate low rank, where singular values decay quickly without hitting zero, and then truncation gives you a great approximation cheaply. That decay profile is what you plot to decide how many components to keep.

**25. What's the pseudoinverse via SVD?**
$A^+ = V \Sigma^+ U^\top$ where $\Sigma^+$ inverts the nonzero singular values. Solves least-squares for any $A$.

> **Saying it out loud.** The pseudoinverse is $V$ times an inverted diagonal times $U^\top$, where you invert the nonzero singular values and leave the zeros alone. It's the closest thing to an inverse that always exists, and applying it solves least squares for any matrix, singular or rectangular or both. Among all least-squares solutions it returns the one with smallest norm. The numerical caution: tiny singular values become huge reciprocals, so in practice you threshold and zero out anything below a tolerance.

---

## D. PSD / definiteness

**26. Define positive semidefinite.**
Symmetric and $x^\top A x \geq 0$ for all $x$. Equivalently, all eigenvalues $\geq 0$.

> **Saying it out loud.** Positive semidefinite means symmetric, and the quadratic form $x^\top A x$ is never negative for any $x$. The intuition is that the matrix never flips a vector to point more than ninety degrees away from where it started. Equivalently all eigenvalues are at least zero, which is usually the easier characterization to check. Covariance matrices, Gram matrices, and kernel matrices are all PSD, which is why the concept keeps appearing.

**27. Define positive definite.**
PSD + $x^\top A x > 0$ for $x \neq 0$. All eigenvalues $> 0$.

> **Saying it out loud.** Positive definite is the strict version: the quadratic form is strictly positive for every nonzero vector, so all eigenvalues are strictly greater than zero. The practical difference from PSD is invertibility, since PD matrices are invertible and PSD ones may not be. That's exactly what ridge regression buys you: adding $\lambda I$ moves a PSD matrix to PD, so the inverse exists.

**28. Three equivalent characterizations of PSD?**
(1) $x^\top A x \geq 0 \forall x$. (2) All eigenvalues $\geq 0$. (3) $A = B^\top B$ for some $B$.

> **Saying it out loud.** Three equivalent ways to say PSD. The quadratic form is never negative. All eigenvalues are at least zero. And the matrix can be written as $B^\top B$ for some $B$. The third one is the most useful in practice, because it means every PSD matrix is a Gram matrix of some set of vectors, which is exactly what a kernel is. Being able to move between the three characterizations depending on what you're proving is the actual skill being tested.

**29. Why is the Hessian PSD at a local minimum?**
Necessary second-order condition: at a local min, the function curves upward (or flat) in every direction.

> **Saying it out loud.** At a local minimum the function has to curve upward, or at worst be flat, in every direction you could step. The Hessian's quadratic form measures curvature along a direction, so requiring it to be nonnegative in every direction is exactly the PSD condition. It's necessary but not sufficient, because a flat direction with zero curvature could be a saddle or a plateau. And it's why saddle points, which have both positive and negative eigenvalues, are the dominant obstacle in high-dimensional non-convex optimization, not local minima.

**30. Why is covariance always PSD?**
$\mathrm{Cov}(X) = \mathbb{E}[(X - \mu)(X - \mu)^\top]$. For any $w$: $w^\top \mathrm{Cov}(X) w = \mathrm{Var}(w^\top X) \geq 0$.

> **Saying it out loud.** Take any direction $w$ and compute $w^\top \mathrm{Cov}(X) w$: it's the variance of the data projected onto $w$. Variance is an average of squares, so it can't be negative, so the quadratic form is nonnegative in every direction, which is the definition of PSD. It's zero in a direction only when the data has no spread there at all, meaning perfectly collinear features. That's the geometric reason collinearity gives you singular covariance.

**31. Why must kernel matrices be PSD?**
Mercer's theorem: a kernel function corresponds to an inner product in some Hilbert space iff its Gram matrix is PSD for any data.

> **Saying it out loud.** A kernel is supposed to be an inner product in some feature space, and inner products always produce PSD Gram matrices, so if your kernel matrix has a negative eigenvalue there is no feature space it corresponds to. That's Mercer's condition. Practically it matters because SVM training is a convex problem only when the kernel matrix is PSD; feed it an indefinite similarity matrix and the optimizer can diverge or return nonsense. This is why people hand-rolling a custom similarity function get bitten.

**32. Sum of two PSD matrices?**
PSD: $x^\top(A+B)x = x^\top A x + x^\top B x \geq 0$.

> **Saying it out loud.** Yes, PSD is closed under addition: just add the two quadratic forms and both pieces are nonnegative. This is why you can freely add a regularizer to a covariance matrix or combine kernels by summing them and stay in the valid family. Nonnegative scaling works too, so PSD matrices form a convex cone. That cone structure is the basis of semidefinite programming.

**33. Product of two PSD matrices — always PSD?**
No (in general). $AB$ may not even be symmetric. PSD only if $A, B$ commute.

> **Saying it out loud.** No. The product of two PSD matrices usually isn't even symmetric, and symmetry is part of the definition, so it typically fails immediately. If the two matrices commute, the product is symmetric and does come out PSD. A related fact worth knowing: the eigenvalues of the product are still nonnegative even when the product isn't symmetric, which surprises people. This is the standard trap question in the PSD section.

**34. Cholesky decomposition — when does it exist?**
For PD matrices: $A = L L^\top$ with $L$ lower triangular and positive diagonal. For PSD, need to allow zeros (semi-Cholesky).

> **Saying it out loud.** Cholesky factors a positive definite matrix into a lower triangular matrix times its own transpose, with strictly positive entries on the diagonal. It exists exactly when the matrix is positive definite, and for merely semidefinite matrices you need a pivoted or perturbed variant that tolerates zeros. It's the workhorse for sampling from multivariate Gaussians and for solving linear systems, being about twice as fast as LU. A neat practical trick: a failing Cholesky is the cheapest test for whether a matrix is actually positive definite, which is why people add a tiny jitter to the diagonal of covariance matrices in Gaussian processes.

---

## E. Matrix calculus

**35. $\nabla_x (b^\top x) = ?$**
$b$.

> **Saying it out loud.** The gradient of a linear function $b^\top x$ is just $b$. It's the multivariate version of the derivative of $bx$ being $b$. This is the base case you build every other matrix-calculus identity on. The one thing to keep straight is your layout convention, whether gradients are columns or rows, because mixing conventions mid-derivation is how people end up with a stray transpose.

**36. $\nabla_x (x^\top A x) = ?$**
$(A + A^\top) x$. For symmetric $A$: $2Ax$.

> **Saying it out loud.** The gradient of a quadratic form is $(A + A^\top)x$, which collapses to $2Ax$ when $A$ is symmetric. Since almost every quadratic form you meet in ML involves a symmetric matrix, a covariance or a Hessian, $2Ax$ is the version you'll actually use. It's the matrix analogue of differentiating $ax^2$ to get $2ax$. Say the general form first and then simplify, because that shows you know why the symmetric case is special.

**37. $\nabla_x \|y - Ax\|^2 = ?$**
$-2A^\top(y - Ax) = 2A^\top A x - 2 A^\top y$.

> **Saying it out loud.** Differentiating the squared residual gives $-2A^\top(y - Ax)$, or equivalently $2A^\top A x - 2A^\top y$. Setting that to zero gives the normal equations directly. The structure worth noticing is $A^\top$ times the residual: the gradient is the residual pushed back through the transpose, which is exactly what backpropagation does at every layer. That's the whole idea of the backward pass in one expression.

**38. Hessian of $\|y - Ax\|^2$?**
$2 A^\top A$. PSD always; PD iff $A$ has full column rank.

> **Saying it out loud.** The Hessian is $2A^\top A$, which is constant, since the objective is quadratic. It's always PSD because it has the $B^\top B$ form, and it's strictly positive definite exactly when $A$ has full column rank. That's why least squares is convex and has a unique minimum when your features are independent, and a flat valley of equally-good solutions when they aren't. And the condition number of $A^\top A$ is the square of the condition number of $A$, which is exactly why gradient descent on ill-conditioned regression crawls.

**39. Closed-form OLS?**
$\hat{x} = (A^\top A)^{-1} A^\top y$.

> **Saying it out loud.** The closed form is $(A^\top A)^{-1} A^\top y$. You should say it and then immediately say you'd never compute it that way, because forming and inverting $A^\top A$ squares the condition number. In practice you solve via QR decomposition or SVD, which is what every library does under the hood. Knowing the formula gets you half credit; knowing not to implement it literally is what gets you the rest.

**40. What's the chain rule for matrix functions?**
$d(f \circ g)/dx = (df/dg)(dg/dx)$ — Jacobian product. Backprop is exactly this.

> **Saying it out loud.** The chain rule for vector functions is just multiplying Jacobians: the derivative of a composition is the derivative of the outer times the derivative of the inner. Backpropagation is exactly this applied to a deep composition of layers. The reason it's done right-to-left, propagating a vector backwards rather than building full Jacobian matrices, is efficiency: with a scalar loss, vector-times-matrix products cost far less than matrix-times-matrix. That choice is the whole difference between reverse-mode and forward-mode autodiff.

**41. Derivative of $\log \det A$ w.r.t. $A$?**
$A^{-T}$. Used in VAEs, normalizing flows, GMM.

> **Saying it out loud.** The derivative of $\log \det A$ with respect to $A$ is the inverse transpose. It shows up any time a log-determinant lands in a loss, which is constantly: the Jacobian term in normalizing flows, the entropy of a Gaussian, and the log-likelihood of a GMM all contain one. The practical note is that you compute $\log \det$ through a Cholesky factorization by summing twice the log of the diagonal, never by computing the determinant and then taking a log, because the determinant of anything large overflows immediately.

---

## F. Conditioning

**42. Definition of condition number?**
$\kappa(A) = \sigma_{\max}/\sigma_{\min}$ for invertible $A$. Measures sensitivity to perturbations.

> **Saying it out loud.** The condition number is the largest singular value over the smallest, and it tells you how much a small perturbation in the input can be amplified in the output. A condition number of a million means you can lose six digits of accuracy, which matters a lot when you only have about sixteen in double precision. Geometrically it's how elongated that ellipsoid is: a well-conditioned matrix maps the sphere to something round, an ill-conditioned one maps it to a needle.

**43. Why does it matter for gradient descent?**
GD on a quadratic with Hessian $H$ converges at rate $\propto (\kappa - 1)/(\kappa + 1)$. Large $\kappa$ → slow.

> **Saying it out loud.** Gradient descent on a quadratic converges at a rate governed by $(\kappa-1)/(\kappa+1)$, so a condition number of a thousand means each step shrinks the error by only about a fifth of a percent. The picture is a long narrow valley: the gradient points mostly across the valley rather than along it, so you zigzag. Your step size is limited by the steepest direction while your progress is limited by the flattest one. That single ratio is why preconditioning, normalization, and momentum all exist.

**44. How does Adam help with bad conditioning?**
Per-coordinate adaptive learning rates approximate diagonal preconditioning. Effectively rescales axes — not perfect, but helps when curvature varies axis-by-axis.

> **Saying it out loud.** Adam keeps a running estimate of each coordinate's gradient magnitude and divides by it, which is a crude diagonal preconditioner. If one parameter direction has curvature a thousand times another's, Adam roughly equalizes their effective step sizes, so you stop being hostage to the worst-conditioned axis. It's only diagonal, so it can't fix conditioning that comes from rotated or correlated directions, which is what full second-order methods would handle. In practice that's most of the benefit for a fraction of the cost.

**45. How does normalization (BN/LN) help with conditioning?**
Renormalizes activations → reduces conditioning of intermediate Jacobians/Hessians. One reason normalization speeds up training.

> **Saying it out loud.** Normalization keeps activations at a controlled scale and mean throughout the network, which stops the layer-to-layer Jacobians from becoming wildly stretched. That improves the conditioning of the loss surface, so larger learning rates become stable and training goes faster. The original internal-covariate-shift explanation has largely been replaced by this smoother-loss-landscape one. The practical evidence is that with normalization you can often raise the learning rate by an order of magnitude without diverging.

**46. What does adding $\lambda I$ to a matrix do to its condition number?**
Reduces $\kappa$. New eigenvalues $\lambda_i + \lambda$. Smallest eigenvalue boosted from $\lambda_n$ to $\lambda_n + \lambda$. Ridge regression's stabilizing effect.

> **Saying it out loud.** Adding $\lambda I$ shifts every eigenvalue up by $\lambda$, so the new condition number is $(\sigma_{\max}^2+\lambda)/(\sigma_{\min}^2+\lambda)$, which is always smaller than before. The small eigenvalues get proportionally the biggest boost, and those were the ones causing the trouble. That's ridge regression's stabilizing effect, and it's the same trick as adding jitter to a covariance matrix before Cholesky. You're trading a little bias for a large gain in numerical stability.

---

## G. Projections and OLS

**47. Define a projection matrix.**
$P^2 = P$. Orthogonal projection: also $P = P^\top$.

> **Saying it out loud.** A projection matrix is one that does nothing extra when you apply it twice: $P^2 = P$. Once you've landed in the subspace, projecting again leaves you alone. If it's also symmetric, it's an orthogonal projection, meaning it drops each point straight down onto the subspace along the shortest path. Non-symmetric idempotent matrices are oblique projections, which do land in the subspace but along a slanted direction, and least squares specifically needs the orthogonal kind.

**48. Projection onto column space of $X$?**
$P = X(X^\top X)^{-1} X^\top$.

> **Saying it out loud.** The projection onto the column space of $X$ is $X(X^\top X)^{-1}X^\top$, the hat matrix. Read it right to left: $X^\top$ takes coordinates in the column space, the inverse solves for the right combination, and $X$ maps back out. You can check it's idempotent in one line, since the middle terms cancel. And it requires $X$ to have full column rank, otherwise you use the pseudoinverse.

**49. Geometric view of OLS solution?**
$\hat{y} = Py$ — projection of $y$ onto $\mathrm{Col}(X)$. Residual $y - \hat{y}$ is orthogonal to columns of $X$ (normal equations).

> **Saying it out loud.** Geometrically OLS is just a projection. Your target vector generally doesn't lie in the span of the columns of $X$, so the best you can do is drop a perpendicular onto that span, and the foot of that perpendicular is the fitted vector. The residual is orthogonal to every column, and writing that orthogonality down is literally the normal equations. This picture makes it obvious why residuals sum to zero when you include an intercept: the residual has to be perpendicular to the all-ones column.

**50. Trace of the hat matrix $P$?**
$\mathrm{tr}(P) = \mathrm{rank}(X)$ = degrees of freedom of the fit.

> **Saying it out loud.** The trace of the hat matrix equals the rank of $X$, which is the number of independent parameters you fit, that is, the degrees of freedom. It works because a projection matrix has eigenvalues that are all zero or one, and the trace counts the ones. This is what generalizes to effective degrees of freedom for regularized models: for ridge, the trace comes out less than the parameter count, which quantifies how much the penalty shrank your model's flexibility.

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
