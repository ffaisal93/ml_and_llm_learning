# Optimization and Matrix Calculus — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Optimization is what training is. Senior interviews go beyond "use Adam" — they probe whether you understand convexity, second-order behavior, conditioning, and the trade-offs between methods. This deep dive complements the linear-algebra deep dive by focusing on what *changes* during training: gradients, Hessians, step sizes, and convergence behavior.

---

## 1. Convex sets and convex functions

A set $C \subseteq \mathbb{R}^n$ is **convex** if for all $x, y \in C$ and $\lambda \in [0,1]$, $\lambda x + (1-\lambda)y \in C$. Lines, balls, half-spaces, polytopes, intersections of these.

A function $f$ is **convex** on a convex domain if for all $x, y$ and $\lambda \in [0, 1]$:

$$
f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)
$$

Equivalent characterizations:
- $f$ is twice-differentiable and Hessian $\nabla^2 f \succeq 0$ everywhere.
- The epigraph $\{(x, t) : f(x) \leq t\}$ is a convex set.
- For all $x, y$: $f(y) \geq f(x) + \nabla f(x)^\top (y - x)$ (first-order condition).

**Strict convexity**: strict inequality. Strictly convex + smooth $\Rightarrow$ unique minimum.

**Strong convexity** ($\mu$-strongly convex): $f(y) \geq f(x) + \nabla f(x)^\top (y - x) + \frac{\mu}{2}\|y - x\|^2$. Equivalently, Hessian $\succeq \mu I$.

### Why convexity matters

For convex $f$:
- Every local minimum is global.
- Gradient descent converges (with decreasing step size).
- Strong convexity gives geometric convergence rate.

Most ML problems are NOT convex (deep nets, GMMs, K-means), but convex theory still informs:
- Convex sub-problems (e.g., per-step Adam updates).
- Loss landscape locally behaves like quadratic near a minimum (Taylor) — convexity intuition transfers.

---

## 2. Smoothness and Lipschitz gradients

A function is **$L$-smooth** if its gradient is $L$-Lipschitz:

$$
\|\nabla f(x) - \nabla f(y)\| \leq L \|x - y\|
$$

Equivalent (for twice-differentiable): $\nabla^2 f \preceq L I$.

This bounds how fast the gradient can change. Gives the descent lemma:

$$
f(y) \leq f(x) + \nabla f(x)^\top (y - x) + \frac{L}{2}\|y - x\|^2
$$

Setting $y = x - \frac{1}{L} \nabla f(x)$ (gradient descent with step $1/L$):

$$
f(y) \leq f(x) - \frac{1}{2L}\|\nabla f(x)\|^2
$$

Each GD step decreases $f$ by at least $\|\nabla f\|^2 / (2L)$. This is the foundation of GD convergence proofs.

---

## 3. Gradient descent convergence rates

For different problem classes:

| Problem | Step size | Convergence rate |
|---|---|---|
| Convex, $L$-smooth | $1/L$ | $f(x_k) - f^* = O(1/k)$ |
| Strongly convex ($\mu$), $L$-smooth | $1/L$ | $\|x_k - x^*\|^2 = O((1 - \mu/L)^k)$ — geometric |
| Non-convex, $L$-smooth | $1/L$ | $\|\nabla f\|^2 \to 0$ at rate $O(1/k)$ |

For strongly convex + smooth, $\kappa = L/\mu$ is the **condition number**. With step size $\eta = 1/L$, GD contracts as $(1 - \mu/L)^k$ per step. With optimal step $\eta = 2/(L + \mu)$, the rate becomes $((\kappa - 1)/(\kappa + 1))^k$. Bad conditioning → slow convergence either way.

### Nesterov acceleration

For convex + smooth, Nesterov's accelerated GD achieves $O(1/k^2)$ — faster than vanilla. Builds momentum from the gradient at a "look-ahead" point.

For strongly convex + smooth, achieves rate $\propto (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ — quadratic improvement in conditioning dependence.

---

## 4. Second-order methods

Use the Hessian $H = \nabla^2 f$ for better step direction.

### Newton's method

$$
x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)
$$

For a quadratic, converges in one step. For convex + smooth + strongly convex, achieves *quadratic* convergence near the optimum (number of correct digits doubles per iteration).

**Costs**: forming and inverting $H$ is $O(n^3)$ in dimension. Infeasible for $n > 10^4$.

### Quasi-Newton (BFGS, L-BFGS)

Approximate $H^{-1}$ from successive gradients. L-BFGS stores only $O(mn)$ history; popular for medium-scale convex optimization (logistic regression, GLMs).

### Why not for deep learning?

- Hessian is huge ($n \sim 10^9$ for big models).
- Hessian isn't PSD (loss is non-convex).
- Cost of forming/storing/inverting prohibitive.
- Second-order info noisy on stochastic batches.

Approximations (Shampoo, K-FAC, Sophia) try to use cheap diagonal/block-diagonal approximations of $H$ while keeping memory manageable.

### Gauss-Newton

For least-squares $\frac{1}{2}\|r(x)\|^2$ with residual $r$: approximate Hessian by $J^\top J$ (Jacobian product). Always PSD. Lev-Marq adds regularization $J^\top J + \lambda I$.

---

## 5. Stochastic gradient methods

For empirical risk $f(x) = \frac{1}{n}\sum_i f_i(x)$ where $n$ is huge:

**SGD**: $x_{k+1} = x_k - \eta \nabla f_{i_k}(x_k)$ for random index $i_k$. Unbiased estimate of gradient; high variance.

### SGD convergence
- Convex + smooth: $O(1/\sqrt{k})$ with diminishing step.
- Strongly convex + smooth: $O(1/k)$ with $1/k$ step.
- Non-convex + smooth: $\|\nabla f\|^2 \to 0$ at $O(1/\sqrt{k})$.

Slower than full GD, but each iteration is $1/n$ as expensive — usually a win for huge datasets.

### Variance reduction
- **Mini-batch**: average $b$ gradients; reduces variance $b\times$.
- **SVRG, SAGA**: explicit variance reduction methods using past gradients. Theoretical wins for finite-sum convex problems; rarely used in deep learning.
- **Larger batch + LR**: linear scaling rule (Goyal et al., 2017): scale LR with batch size up to a critical batch.

### Adaptive methods (Adam et al.)

Adapt per-parameter step size based on historical gradients. Not strictly needed for convex problems; often wins for deep learning due to varying gradient magnitudes across parameters.

(See `10_optimizers/` for full optimizer details.)

---

## 6. Constrained optimization

Minimize $f(x)$ subject to $g_i(x) \leq 0$, $h_j(x) = 0$.

### Lagrangian

$$
\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_i \lambda_i g_i(x) + \sum_j \nu_j h_j(x)
$$

with $\lambda_i \geq 0$.

### KKT conditions (necessary at optimum *under a constraint qualification* like LICQ or Slater's; sufficient for convex)

1. **Stationarity**: $\nabla_x \mathcal{L} = 0$.
2. **Primal feasibility**: $g_i(x^*) \leq 0$, $h_j(x^*) = 0$.
3. **Dual feasibility**: $\lambda_i^* \geq 0$.
4. **Complementary slackness**: $\lambda_i^* g_i(x^*) = 0$ for each $i$.

Complementary slackness says: for each $i$, at least one of $\lambda_i$ and $g_i$ is zero (their product is zero). So an inactive constraint ($g_i < 0$) must have zero multiplier; a non-zero multiplier signals an active constraint.

### Examples in ML

- **SVM dual**: derived via Lagrangian + KKT. Support vectors = points with non-zero $\lambda$.
- **Constrained capacity in MoE**: capacity factor caps tokens per expert — Lagrangian relaxation.
- **PCA**: $\arg\max_w w^\top \Sigma w$ s.t. $\|w\|^2 = 1$ → Lagrangian gives eigenvalue equation.

### Lagrangian duality

Define dual function $g(\lambda, \nu) = \inf_x \mathcal{L}$. **Weak duality**: $g \leq f^*$ always. **Strong duality**: $g = f^*$ for convex problems with constraint qualifications (Slater's condition).

---

## 7. The loss landscape in deep learning

Deep network losses are non-convex. Theoretical results that matter:

### Saddle points dominate

In high dimensions, *saddle points* (Hessian has both positive and negative eigenvalues) vastly outnumber local minima. Most "stuck" points are saddles, not minima. Negative-curvature directions can be exploited to escape.

### Most local minima are good

Empirical and theoretical evidence (Choromanska et al. 2015; Kawaguchi 2016) suggests that for over-parameterized networks, most local minima have similar (low) loss values. Large flat regions of low loss.

### Flat vs sharp minima

Flat minima (low Hessian eigenvalues) generalize better than sharp ones empirically (Hochreiter & Schmidhuber 1997; Keskar et al. 2017). SGD's noise drives it toward flat minima.

### Edge of stability (Cohen et al. 2021)

When training with full-batch GD, the largest Hessian eigenvalue grows until $\lambda_{\max} \approx 2/\eta$, then *oscillates*. Training continues despite violating classical stability. Surprising.

---

## 8. Conditioning revisited (in optimization context)

Condition number $\kappa = L/\mu$ for strongly convex problems. Affects:

- **GD convergence rate**: $(1 - \mu/L)^k$ — exponential, but slow when $\kappa$ large.
- **Number of iterations**: $O(\kappa \log(1/\epsilon))$ for accuracy $\epsilon$.
- **With Nesterov**: $O(\sqrt{\kappa} \log(1/\epsilon))$.

In ML, ill-conditioning shows up because:
- Different parameters have different scales.
- Different layers have different curvatures.
- Some directions in parameter space are "stiff" (small eigenvalues of Hessian = directions that change loss slowly).

### Mitigations
- **Standardize features**: reduces conditioning of input.
- **Normalization layers** (BN, LN): renormalize internal activations.
- **Adaptive optimizers** (Adam): per-parameter step ≈ diagonal preconditioning.
- **Second-order methods** (Shampoo, K-FAC): explicit preconditioning.
- **Architecture design**: residual connections, careful initialization.

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Is deep learning convex? | Some parts | No — non-convex, but most local minima are reasonable |
| Newton's method always works? | Yes | Only locally; may diverge far from optimum or hit non-PSD Hessian |
| Adam = preconditioned SGD? | Sort of | Approximately — diagonal preconditioning + momentum |
| KKT applies only to convex? | Yes | Necessary conditions hold for any smooth optimization; sufficient only if convex |
| Why scale LR with batch size? | Tradition | Linear scaling rule from Goyal et al. — gradients have lower variance with bigger batch |
| Edge of stability — what's stable? | Loss going down | Loss bounces but trends down; classical stability bound violated |
| Saddle points in deep nets? | Bad | Very common; Hessian eigenvalues are mixed; SGD escapes via noise |

---

## 10. Eight most-asked interview questions

1. **What's strong convexity and why does it matter?** ($\mu$-strongly convex; gives geometric GD convergence rate.)
2. **Derive the gradient descent convergence rate for smooth + strongly convex.** ($\|x_k - x^*\|^2 \leq (1 - \mu/L)^k \|x_0 - x^*\|^2$.)
3. **Why doesn't Newton's method work for deep learning?** (Hessian too big to form/invert; not PSD; noisy on batches.)
4. **What's KKT and when does it apply?** (Necessary at optimum; sufficient for convex; complementary slackness.)
5. **Derive the SVM dual using Lagrangian.** (Standard derivation; support vectors emerge from KKT.)
6. **Why does Adam help in deep learning?** (Diagonal preconditioning of varying gradient scales; momentum.)
7. **What's the edge of stability phenomenon?** (Hessian top eigenvalue oscillates around $2/\eta$; classical stability violated.)
8. **Why are flat minima better for generalization?** (Robustness to perturbation; effective Bayesian model averaging in their basin.)

---

## 11. Drill plan

- For convex, smooth, strongly convex — recite definitions + GD rate.
- Derive descent lemma from $L$-smoothness in 5 lines.
- For Newton's method, recite: update, convergence rate, why it fails for deep learning.
- For Lagrangian + KKT — recite all four conditions.
- Derive SVM dual on paper.
- Sketch the loss landscape: saddles dominate, flat = good, edge of stability.

---

## 12. Further reading

- Boyd & Vandenberghe, *Convex Optimization* — the canonical text. Chapters 1–5 are essential.
- Nocedal & Wright, *Numerical Optimization* — second-order methods, quasi-Newton.
- Bubeck, *Convex Optimization: Algorithms and Complexity* — modern, concise.
- Goodfellow, Bengio, Courville, *Deep Learning*, ch. 8 — optimization for neural networks.
- Cohen et al. (2021), *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.*
- Choromanska et al. (2015), *The Loss Surfaces of Multilayer Networks.*
