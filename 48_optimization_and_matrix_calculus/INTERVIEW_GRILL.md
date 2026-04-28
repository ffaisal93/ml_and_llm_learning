# Optimization & Matrix Calculus — Interview Grill

> 50 questions on convexity, gradient descent, second-order methods, constrained optimization, deep learning loss landscapes. Drill until you can answer 35+ cold.

---

## A. Convexity

**1. Define a convex function.**
$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ for $\lambda \in [0,1]$ on a convex domain.

**2. Hessian condition for convexity?**
$\nabla^2 f \succeq 0$ everywhere. Strict: $\succ 0$.

**3. First-order characterization?**
$f(y) \geq f(x) + \nabla f(x)^\top (y - x)$. Tangent line below graph.

**4. Strong convexity?**
$f(y) \geq f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2}\|y-x\|^2$. Hessian $\succeq \mu I$.

**5. Why is convexity nice?**
Every local min is global. GD converges. Strong convexity → geometric convergence.

**6. Is deep learning convex?**
No — non-convex in parameters. But most local minima are reasonable in over-parameterized regime.

**7. $\ell_1$ norm — convex?**
Yes. Even non-smooth (kink at 0), but convex.

**8. Composition of convex functions — always convex?**
No. Convex of convex isn't necessarily convex. Convex + non-decreasing of convex is convex (e.g., $\exp(\|x\|)$ is convex).

---

## B. Smoothness and gradient descent

**9. Define $L$-smooth.**
Gradient is $L$-Lipschitz: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$. Equivalently $\nabla^2 f \preceq L I$.

**10. Descent lemma?**
$f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2$.

**11. Optimal step size for $L$-smooth GD?**
$\eta = 1/L$. Gives steepest decrease per step.

**12. GD rate for smooth convex?**
$f(x_k) - f^* = O(1/k)$.

**13. GD rate for smooth strongly convex?**
$\|x_k - x^*\|^2 \leq (1 - \mu/L)^k \|x_0 - x^*\|^2$. Geometric.

**14. GD rate for non-convex smooth?**
$\min_{k' \leq k}\|\nabla f(x_{k'})\|^2 = O(1/k)$. Convergence to stationary point only.

**15. What's Nesterov acceleration?**
Modified momentum that achieves $O(1/k^2)$ for convex (vs $O(1/k)$ for GD). Quadratic improvement.

**16. Condition number $\kappa$?**
$L/\mu$ for strongly convex + smooth. Bigger $\kappa$ = worse conditioning = slower convergence.

**17. Why is Nesterov better for big $\kappa$?**
GD: $O(\kappa \log 1/\epsilon)$ iterations. Nesterov: $O(\sqrt{\kappa} \log 1/\epsilon)$. Quadratic improvement in conditioning dependence.

---

## C. Second-order methods

**18. Newton's method update?**
$x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)$.

**19. Convergence rate of Newton's near optimum?**
Quadratic — number of correct digits doubles per iteration. Requires: starting close enough to the optimum, $H$ Lipschitz-continuous, and $H \succ 0$ at the optimum (strong convexity locally). Far from the optimum, Newton can diverge or step in the wrong direction.

**20. Cost of Newton per step?**
$O(n^3)$ to invert Hessian (or $O(n^2)$ to solve).

**21. Why doesn't Newton work for deep learning?**
$n \sim 10^9$ → can't form/invert. Loss non-convex → Hessian not PSD → can step in wrong direction. Stochastic batches → noisy Hessian.

**22. BFGS vs L-BFGS?**
BFGS: stores full $n \times n$ Hessian approximation. L-BFGS: stores last $m$ gradient differences ($O(mn)$ memory). L-BFGS standard for medium-scale convex problems.

**23. Gauss-Newton — when?**
Least-squares with residual $r$: approximate Hessian by $J^\top J$. Always PSD. Cheap when $J$ is reasonable size.

**24. K-FAC, Shampoo — what are they?**
Block-diagonal / Kronecker-factored approximations to the Hessian. Cheap second-order methods for deep networks.

---

## D. Stochastic methods

**25. SGD update?**
$x_{k+1} = x_k - \eta \nabla f_{i_k}(x_k)$ for random index $i_k$.

**26. Why is SGD faster than full GD per epoch?**
Each step costs $O(1)$ instead of $O(n)$. With $n$ huge, SGD is way more iterations per dataset pass.

**27. SGD vs GD convergence rate?**
GD strongly convex: $O(\kappa \log 1/\epsilon)$. SGD strongly convex: $O(1/\epsilon)$. SGD has worse asymptotic rate but each step cheaper.

**28. Linear scaling rule (Goyal et al.)?**
When you scale batch size by $k$, scale LR by $k$ — keeps the same effective update.

**29. Up to what batch size?**
Critical batch size — beyond that, returns diminish (McCandlish et al., 2018). Different per task.

---

## E. Constrained optimization

**30. State the Lagrangian.**
$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum \lambda_i g_i(x) + \sum \nu_j h_j(x)$, $\lambda \geq 0$.

**31. Four KKT conditions?**
Stationarity ($\nabla_x \mathcal{L} = 0$), primal feasibility, dual feasibility ($\lambda \geq 0$), complementary slackness ($\lambda_i g_i = 0$).

**32. What's complementary slackness?**
Either constraint is active ($g_i = 0$) or its multiplier is zero. Encodes: only active constraints have non-trivial influence.

**33. SVM support vectors via KKT?**
Hard-margin: $\lambda_i = 0$ for non-support vectors; $\lambda_i > 0$ only for points exactly on the margin. Soft-margin (with box constraint $0 \leq \lambda_i \leq C$): $\lambda_i = 0$ off-margin; $0 < \lambda_i < C$ exactly on margin; $\lambda_i = C$ for margin violators.

**34. Strong duality — when?**
For convex problems satisfying constraint qualifications (e.g., Slater's condition: strictly feasible point exists). $g(\lambda^*) = f(x^*)$.

**35. Why is SVM dual easier than primal?**
Many fewer variables (one $\lambda$ per training example, but most are zero). Plus kernel trick fits naturally into the dual.

---

## F. Deep learning loss landscape

**36. Saddle points vs local minima in high dim?**
Saddle points dominate. With many dimensions, the chance all eigenvalues of a random Hessian are positive is low.

**37. How does SGD escape saddle points?**
Stochastic noise in gradient provides random kicks; one of them usually has component in negative-curvature direction.

**38. Flat vs sharp minima — generalization?**
Flat → better generalization empirically. Hessian eigenvalues correlate with train-test gap.

**39. Why does SGD prefer flat minima?**
Stochastic noise can't keep you in a sharp basin (small fluctuations push you out). Flat basins are more "stable" under noise.

**40. Edge of stability?**
With full-batch GD, top Hessian eigenvalue grows until $\lambda_{\max} \approx 2/\eta$, then oscillates. Loss bounces but decreases. Classical stability bound violated.

**41. Why does loss decrease at edge of stability despite oscillation?**
Implicit regularization toward flat regions. Even with oscillation, the average trajectory progresses.

**42. Lottery ticket hypothesis (Frankle & Carbin)?**
Dense networks contain sparse subnetworks ("winning tickets") that match performance when trained from same init. Implies optimization finds good solutions specific to init.

---

## G. Conditioning in deep learning

**43. Why does normalization help conditioning?**
Renormalizes activations → reduces variance in Hessian eigenvalues across layers → better conditioned.

**44. Why does Adam help conditioning?**
Per-parameter step ≈ diagonal preconditioning. Approximately rescales each axis by historical gradient magnitude.

**45. Standardize input features — why?**
Inputs of different scales cause Hessian to have very different eigenvalues across input directions. Standardize → balanced.

**46. What's a "stiff" direction in parameter space?**
Direction with small Hessian eigenvalue — function changes slowly along it. Hard for GD; needs many steps.

---

## H. Subtleties

**47. Subgradient — what is it?**
Generalization of gradient for non-smooth convex functions. For $\ell_1$: $\partial |x| = \mathrm{sign}(x)$ if $x \neq 0$; $\partial |0| = [-1, 1]$.

**48. Proximal gradient — when?**
Composite objectives like $f + g$ where $f$ smooth, $g$ non-smooth (e.g., $\ell_1$). Step: $x_{k+1} = \mathrm{prox}_{\eta g}(x_k - \eta \nabla f(x_k))$. Used for lasso (ISTA).

**49. ADMM?**
Alternating Direction Method of Multipliers. Splits an objective into easier subproblems via auxiliary variables. Used for distributed convex optimization.

**50. Lagrangian dual is concave — true?**
Yes. The dual function $g(\lambda) = \inf_x \mathcal{L}(x, \lambda)$ is concave (infimum of affine functions is concave). So dual problem is convex regardless of primal.

---

## Quick fire

**51.** *Convex def?* Tangent below graph.
**52.** *GD rate strongly convex?* Geometric, $(1-\mu/L)^k$.
**53.** *Newton convergence?* Quadratic.
**54.** *L-smooth Hessian bound?* $\preceq L I$.
**55.** *Strong convex Hessian bound?* $\succeq \mu I$.
**56.** *Optimal step for $L$-smooth?* $1/L$.
**57.** *Condition number?* $L/\mu$.
**58.** *Nesterov rate?* $O(1/k^2)$ convex.
**59.** *KKT conditions count?* 4.
**60.** *Why SGD escapes saddle?* Noise.

---

## Self-grading

If you can't answer 1-15, you don't know optimization. If you can't answer 16-35, you'll struggle on convex / second-order interview questions. If you can't answer 36-50, frontier-lab questions on deep learning training dynamics will go past you.

Aim for 40+/60 cold.
