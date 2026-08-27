# Optimization and Matrix Calculus — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

Optimization is what training is. Senior interviews go beyond "use Adam" — they probe whether you understand convexity, second-order behavior, conditioning, and the trade-offs between methods. This deep dive complements the linear-algebra deep dive by focusing on what *changes* during training: gradients, Hessians, step sizes, and convergence behavior.

---

## 1. Convex sets and convex functions

> **In plain language.** A convex set is one with no dents — the straight line between any two points stays inside it. A convex function is one shaped like a bowl, where any chord you draw sits on or above the surface. The definitions below are those two sentences written in symbols, plus three equivalent ways of checking.

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

> **Saying it out loud.** Convex means bowl-shaped: pick any two points on the graph, draw a straight line between them, and the function never pokes above that line. There are three equivalent ways to check it — the chord definition, a non-negative Hessian, or the tangent plane lying below the graph everywhere — and which one you use depends on what you have. Strict convexity means no flat regions, so the minimum is unique. Strong convexity is stronger still: the function must curve at least as much as some fixed quadratic, and that's the property that upgrades convergence from slow to geometric.

### Why convexity matters

For convex $f$:
- Every local minimum is global.
- Gradient descent converges (with decreasing step size).
- Strong convexity gives geometric convergence rate.

Most ML problems are NOT convex (deep nets, GMMs, K-means), but convex theory still informs:
- Convex sub-problems (e.g., per-step Adam updates).
- Loss landscape locally behaves like quadratic near a minimum (Taylor) — convexity intuition transfers.

> **Saying it out loud.** The payoff is that every local minimum is global, so a zero gradient means you're done and there's nothing better hiding elsewhere. That turns optimization from a search into a computation, and it makes results reproducible across initializations. Deep learning is emphatically not convex, but the theory still earns its keep, because near any minimum a smooth loss looks quadratic under Taylor expansion, so convex intuitions about conditioning and step sizes transfer directly to the endgame of training.

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

> **Saying it out loud.** Smooth here isn't about being differentiable — it means the gradient can't change too fast, capped by a constant $L$, which is the same as saying curvature is bounded. That matters because it tells you how far you can trust the gradient you just computed, and therefore how big a step is safe. Plug the descent lemma in and you get both the optimal step, one over $L$, and a guarantee of how much progress each step makes. That single inequality is the foundation under essentially every convergence proof in first-order optimization.

---

## 3. Gradient descent convergence rates

For different problem classes:

| Problem | Step size | Convergence rate |
|---|---|---|
| Convex, $L$-smooth | $1/L$ | $f(x_k) - f^* = O(1/k)$ |
| Strongly convex ($\mu$), $L$-smooth | $1/L$ | $\|x_k - x^*\|^2 = O((1 - \mu/L)^k)$ — geometric |
| Non-convex, $L$-smooth | $1/L$ | $\|\nabla f\|^2 \to 0$ at rate $O(1/k)$ |

For strongly convex + smooth, $\kappa = L/\mu$ is the **condition number**. With step size $\eta = 1/L$, GD contracts as $(1 - \mu/L)^k$ per step. With optimal step $\eta = 2/(L + \mu)$, the rate becomes $((\kappa - 1)/(\kappa + 1))^k$. Bad conditioning → slow convergence either way.

> **Saying it out loud.** Three regimes, and the difference between them is entirely about how much structure you can assume. Smooth and convex gives you error falling like one over $k$, which is painfully slow — ten times more iterations per extra decimal place. Add strong convexity and it becomes geometric, error shrinking by a constant factor each step. Drop convexity and all you can promise is that you approach a stationary point, which might be a saddle. That last one is where deep learning actually lives, and being honest about it is better than pretending the guarantees are stronger.

### Nesterov acceleration

For convex + smooth, Nesterov's accelerated GD achieves $O(1/k^2)$ — faster than vanilla. Builds momentum from the gradient at a "look-ahead" point.

For strongly convex + smooth, achieves rate $\propto (\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$ — quadratic improvement in conditioning dependence.

> **Saying it out loud.** Nesterov's idea is to evaluate the gradient where momentum is about to take you rather than where you currently are, so you can correct before overshooting. That look-ahead turns a one-over-$k$ rate into one-over-$k$-squared, and in terms of conditioning it takes the iteration count from proportional to $\kappa$ down to $\sqrt{\kappa}$. At a condition number of 10,000, that's 100 steps instead of 10,000. It's also provably the best any first-order method can do. Deep learning still mostly uses plain heavy-ball momentum, because the theoretical edge doesn't reliably survive stochastic gradients.

---

## 4. Second-order methods

Use the Hessian $H = \nabla^2 f$ for better step direction.

### Newton's method

$$
x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)
$$

For a quadratic, converges in one step. For convex + smooth + strongly convex, achieves *quadratic* convergence near the optimum (number of correct digits doubles per iteration).

**Costs**: forming and inverting $H$ is $O(n^3)$ in dimension. Infeasible for $n > 10^4$.

> **Saying it out loud.** Newton fits a quadratic bowl to your local landscape and jumps straight to its bottom, which is why it converges quadratically — correct digits double each iteration. It's also scale-invariant, so rescaling your parameters doesn't change the path it takes, unlike gradient descent. The catch is all in the fine print: you need to start close, and the Hessian needs to be positive definite. Far from the optimum, the Newton direction can point toward a saddle, which is why every serious implementation wraps it in a line search or trust region.

### Quasi-Newton (BFGS, L-BFGS)

Approximate $H^{-1}$ from successive gradients. L-BFGS stores only $O(mn)$ history; popular for medium-scale convex optimization (logistic regression, GLMs).

> **Saying it out loud.** Quasi-Newton methods build up an approximation to the inverse Hessian from the gradients you've already computed, so you get curvature information without ever forming the matrix. BFGS still stores a full $n$-by-$n$ approximation; L-BFGS keeps only the last 5 to 20 gradient and position differences and reconstructs the matrix's action on the fly, dropping memory to about $10n$. It's superb for smooth deterministic problems up to millions of parameters. It falls apart with stochastic gradients, because curvature estimated from noisy differences is nonsense.

### Why not for deep learning?

- Hessian is huge ($n \sim 10^9$ for big models).
- Hessian isn't PSD (loss is non-convex).
- Cost of forming/storing/inverting prohibitive.
- Second-order info noisy on stochastic batches.

Approximations (Shampoo, K-FAC, Sophia) try to use cheap diagonal/block-diagonal approximations of $H$ while keeping memory manageable.

> **Saying it out loud.** Three independent showstoppers. Size — a Hessian for a billion parameters would need something like $10^{18}$ entries, which is more memory than exists. Non-convexity — the Hessian isn't positive definite, so the Newton step can point uphill and take you to a saddle rather than a minimum. And noise — you're estimating curvature from a minibatch, and inverting a noisy matrix amplifies the noise dramatically. Modern approximations like K-FAC and Shampoo attack all three with structural assumptions, and they're finally getting competitive.

### Gauss-Newton

For least-squares $\frac{1}{2}\|r(x)\|^2$ with residual $r$: approximate Hessian by $J^\top J$ (Jacobian product). Always PSD. Lev-Marq adds regularization $J^\top J + \lambda I$.

> **Saying it out loud.** For least-squares problems you can throw away the awkward second-derivative term and approximate the Hessian as $J^\top J$. That's cheap, and crucially it's always positive semidefinite, so you never get a step in the wrong direction — the main failure mode of Newton on non-convex problems just disappears. Levenberg-Marquardt is the version people actually run, blending between Gauss-Newton and gradient descent via a damping parameter that adapts to how well the step worked. It's the standard tool for nonlinear curve fitting and bundle adjustment.

---

## 5. Stochastic gradient methods

For empirical risk $f(x) = \frac{1}{n}\sum_i f_i(x)$ where $n$ is huge:

**SGD**: $x_{k+1} = x_k - \eta \nabla f_{i_k}(x_k)$ for random index $i_k$. Unbiased estimate of gradient; high variance.

> **Saying it out loud.** The whole idea is that you don't need an exact gradient, you need a cheap one that points roughly the right way. A minibatch gradient is unbiased and noisy, and while that looks like a downgrade per step, you get hundreds or thousands of steps in the time a full-batch pass takes one. Progress per unit of compute is what matters, and noisy-and-frequent beats exact-and-rare by a wide margin at scale. The noise even turns out to be useful, since it's what kicks you off saddle points and biases you toward flat minima.

### SGD convergence
- Convex + smooth: $O(1/\sqrt{k})$ with diminishing step.
- Strongly convex + smooth: $O(1/k)$ with $1/k$ step.
- Non-convex + smooth: $\|\nabla f\|^2 \to 0$ at $O(1/\sqrt{k})$.

Slower than full GD, but each iteration is $1/n$ as expensive — usually a win for huge datasets.

> **Saying it out loud.** On paper SGD's rates are worse than full gradient descent across every regime, and the reason is that the gradient noise doesn't shrink as you approach the optimum — you end up rattling around in a noise ball rather than converging, which is why the step size has to decay. But the comparison is per iteration, and SGD's iterations cost a fraction as much. So the asymptotic rate is the wrong thing to compare; what you want is error per FLOP, and there SGD wins decisively once the dataset is large.

### Variance reduction
- **Mini-batch**: average $b$ gradients; reduces variance $b\times$.
- **SVRG, SAGA**: explicit variance reduction methods using past gradients. Theoretical wins for finite-sum convex problems; rarely used in deep learning.
- **Larger batch + LR**: linear scaling rule (Goyal et al., 2017): scale LR with batch size up to a critical batch.

> **Saying it out loud.** Mini-batching is the variance reduction everyone actually uses: average $b$ gradients and the variance drops by a factor of $b$, which lets you take bigger steps. The linear scaling rule follows from that — double the batch, double the learning rate, keeping total movement per example constant — and it works up to a critical batch size, past which you're paying for compute that doesn't buy fewer steps. The sophisticated methods, SVRG and SAGA, have beautiful theory for finite-sum convex problems and basically never get used in deep learning, because they assume a static objective and deep learning has data augmentation, dropout and a moving target.

### Adaptive methods (Adam et al.)

Adapt per-parameter step size based on historical gradients. Not strictly needed for convex problems; often wins for deep learning due to varying gradient magnitudes across parameters.

(See `10_optimizers/` for full optimizer details.)

> **Saying it out loud.** Adam divides each parameter's step by the running root-mean-square of its own recent gradients, which is a diagonal preconditioner — parameters with consistently large gradients move proportionally less. That approximately equalizes the effective curvature across axes, so you're no longer forced to pick a learning rate small enough for the sharpest direction. It's a crude second-order approximation, only the diagonal, and it's essentially free. That's why Adam trains transformers comfortably where plain SGD needs enormous care, and why vision, with better-conditioned convolutional losses, still often prefers SGD with momentum.

---

## 6. Constrained optimization

Minimize $f(x)$ subject to $g_i(x) \leq 0$, $h_j(x) = 0$.

> **Saying it out loud.** The Lagrangian turns a constrained problem into an unconstrained one by attaching a price to each constraint. Instead of forbidding a region, you charge for entering it, and at the right prices the unconstrained solution coincides with the constrained one. Those prices — the multipliers — aren't just bookkeeping: each tells you how much your optimum would improve if you loosened that constraint slightly, which is a genuinely useful number. Inequality multipliers have to be non-negative, because a constraint can only push you back into the feasible region, never pull you out.

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

> **Saying it out loud.** Four conditions, and they're worth being able to recite. The gradient of the Lagrangian is zero, your solution actually satisfies the constraints, the inequality prices are non-negative, and each price times its constraint is zero. That last one, complementary slackness, is the interesting one: a constraint you aren't pressing against has zero price and therefore no influence. For convex problems these aren't just necessary, they're sufficient — anything satisfying all four is the optimum, which makes them a certificate you can check rather than a mere characterization.

### Examples in ML

- **SVM dual**: derived via Lagrangian + KKT. Support vectors = points with non-zero $\lambda$.
- **Constrained capacity in MoE**: capacity factor caps tokens per expert — Lagrangian relaxation.
- **PCA**: $\arg\max_w w^\top \Sigma w$ s.t. $\|w\|^2 = 1$ → Lagrangian gives eigenvalue equation.

> **Saying it out loud.** The SVM is the example to lead with, because complementary slackness is exactly what makes it sparse — points comfortably on the right side of the margin get zero multiplier and drop out completely, leaving only the support vectors, so model size scales with problem difficulty rather than dataset size. PCA is the other one worth having: maximize variance subject to a unit-norm constraint, write the Lagrangian, set the gradient to zero, and the eigenvalue equation falls straight out. That derivation is a nice two-minute whiteboard answer that shows you can actually use this machinery rather than just define it.

### Lagrangian duality

Define dual function $g(\lambda, \nu) = \inf_x \mathcal{L}$. **Weak duality**: $g \leq f^*$ always. **Strong duality**: $g = f^*$ for convex problems with constraint qualifications (Slater's condition).

> **Saying it out loud.** Weak duality — the dual is always a lower bound on the primal — holds no matter how horrible your problem is, which is why dual bounds are useful even for combinatorial problems you'll never solve exactly. Strong duality, where the gap closes entirely, needs convexity plus a mild condition like there existing a strictly feasible point. And here's the fact worth knowing: the dual is a concave maximization problem always, because it's an infimum of functions affine in the multipliers. So even a non-convex primal has a convex dual, which is exactly why people bother going there.

---

## 7. The loss landscape in deep learning

Deep network losses are non-convex. Theoretical results that matter:

> **Saying it out loud.** The honest summary is that the classical worry — getting trapped in a bad local minimum — was mostly wrong, and the real story is stranger. In a billion dimensions, being a local minimum requires every eigenvalue to be positive, which is an absurd coincidence, so almost everything with a near-zero gradient is a saddle. Meanwhile the minima you do reach turn out to be roughly equally good, so where you land matters less than theory predicted. What actually matters is the geometry you settle into — flat versus sharp — and that's decided by the algorithm and its noise rather than the landscape alone.

### Saddle points dominate

In high dimensions, *saddle points* (Hessian has both positive and negative eigenvalues) vastly outnumber local minima. Most "stuck" points are saddles, not minima. Negative-curvature directions can be exploited to escape.

> **Saying it out loud.** To be a local minimum in a billion dimensions, every single Hessian eigenvalue has to be positive; to be a saddle you just need a mix, which is overwhelmingly more likely. So the things that stall training are saddles and the long flat plateaus around them, not bad minima. The upside is that a saddle always has an escape route — a direction of negative curvature — and SGD's gradient noise finds it more or less automatically. Full-batch gradient descent, with no noise, can genuinely sit at a saddle indefinitely.

### Most local minima are good

Empirical and theoretical evidence (Choromanska et al. 2015; Kawaguchi 2016) suggests that for over-parameterized networks, most local minima have similar (low) loss values. Large flat regions of low loss.

> **Saying it out loud.** In over-parameterized networks the minima you can actually reach have very similar loss, so the choice among them barely matters for training error. Part of the explanation is symmetry — permuting hidden units gives you an identical function, so an enormous number of apparently distinct minima are literally the same solution relabeled. The practical consequence is that you shouldn't spend effort trying to find a better basin; you should spend it on which basin generalizes, which is a different question and the one flat-versus-sharp addresses.

### Flat vs sharp minima

Flat minima (low Hessian eigenvalues) generalize better than sharp ones empirically (Hochreiter & Schmidhuber 1997; Keskar et al. 2017). SGD's noise drives it toward flat minima.

> **Saying it out loud.** The empirical finding is that solutions sitting in wide flat basins generalize better than ones in narrow sharp ones. The intuition is robustness — if perturbing the weights barely changes the loss, then a small shift in the data distribution probably won't hurt much either, whereas a sharp minimum is finely tuned to the exact training set. It's also the standard explanation for the large-batch generalization gap, since less gradient noise means less pressure to settle somewhere flat. Worth flagging the caveat that sharpness isn't reparameterization-invariant, so the notion is slipperier than it sounds — which is why SAM optimizes it explicitly rather than relying on it emerging.

### Edge of stability (Cohen et al. 2021)

When training with full-batch GD, the largest Hessian eigenvalue grows until $\lambda_{\max} \approx 2/\eta$, then *oscillates*. Training continues despite violating classical stability. Surprising.

> **Saying it out loud.** Classical theory says gradient descent is stable only while the sharpest curvature stays below $2/\eta$. What actually happens is that sharpness rises during training until it hits that threshold, then parks right there, with the loss oscillating but still trending down. So training spends most of its life in a regime the theory declares divergent. The reading that matters practically is that your learning rate is implicitly choosing how sharp a solution you get — a slightly aggressive learning rate isn't just tolerable, it's doing regularization work.

---

## 8. Conditioning revisited (in optimization context)

Condition number $\kappa = L/\mu$ for strongly convex problems. Affects:

- **GD convergence rate**: $(1 - \mu/L)^k$ — exponential, but slow when $\kappa$ large.
- **Number of iterations**: $O(\kappa \log(1/\epsilon))$ for accuracy $\epsilon$.
- **With Nesterov**: $O(\sqrt{\kappa} \log(1/\epsilon))$.

In ML, ill-conditioning shows up because:
- Different parameters have different scales.
- Different layers have different curvatures.
- Some directions in parameter space are "stiff" (large eigenvalues of Hessian = loss changes sharply, capping the step size), while others are flat/"sloppy" (small eigenvalues = loss changes slowly, needing many steps).

> **Saying it out loud.** The condition number is the ratio of sharpest curvature to flattest, and it's the single number that decides how painful your optimization will be. At $\kappa$ near one you walk straight to the bottom; at 10,000 you zigzag across a long thin valley making almost no progress along it, because any step size safe in the sharp direction is uselessly small in the flat one. Everything practical in deep learning optimization — feature standardization, normalization layers, Adam, Shampoo — is an attempt to shrink that ratio. And the payoff is concrete: Nesterov turns a $\kappa$ dependence into $\sqrt{\kappa}$, which is a hundredfold at $\kappa$ equal to 10,000.

### Mitigations
- **Standardize features**: reduces conditioning of input.
- **Normalization layers** (BN, LN): renormalize internal activations.
- **Adaptive optimizers** (Adam): per-parameter step ≈ diagonal preconditioning.
- **Second-order methods** (Shampoo, K-FAC): explicit preconditioning.
- **Architecture design**: residual connections, careful initialization.

> **Saying it out loud.** They're all the same idea applied at different points: make the curvature more uniform across directions. Standardizing features fixes it at the input, normalization layers fix it between layers, Adam fixes it per parameter with a diagonal preconditioner, and Shampoo or K-FAC do it with structured full-matrix approximations. Residual connections and careful initialization attack it architecturally, by keeping gradient magnitudes comparable across depth. If you can only do one, standardize your inputs — it's free and it's the most commonly skipped.

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
