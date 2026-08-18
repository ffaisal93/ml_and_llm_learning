# Optimization & Matrix Calculus — Interview Grill

> 50 questions on convexity, gradient descent, second-order methods, constrained optimization, deep learning loss landscapes. Drill until you can answer 35+ cold.

---

## A. Convexity

**1. Define a convex function.**
$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ for $\lambda \in [0,1]$ on a convex domain.

> **Saying it out loud.** A function is convex if the straight line between any two points on its graph never dips below the graph itself — picture a bowl, and any chord you draw sits on or above the surface. That's the whole definition, and the inequality is just that sentence in symbols. The reason anyone cares is what it rules out: no hidden valleys, so there's nowhere for an optimizer to get stuck that isn't the global answer.

**2. Hessian condition for convexity?**
$\nabla^2 f \succeq 0$ everywhere. Strict: $\succ 0$.

> **Saying it out loud.** For a twice-differentiable function, convex means the Hessian is positive semidefinite everywhere — curvature is non-negative in every direction you could travel. In one dimension that's just the second derivative being non-negative, and the matrix version is the same statement generalized. Positive definite rather than semidefinite gives you strict convexity, which rules out flat regions and guarantees the minimum is unique.

**3. First-order characterization?**
$f(y) \geq f(x) + \nabla f(x)^\top (y - x)$. Tangent line below graph.

> **Saying it out loud.** It says the tangent plane at any point lies entirely below the function — so the linear approximation is always an underestimate, never an overestimate. That's actually the most useful form in practice, because it means the gradient at your current point gives you a global lower bound on the function. Almost every convergence proof in convex optimization leans on exactly that fact.

**4. Strong convexity?**
$f(y) \geq f(x) + \nabla f(x)^\top (y-x) + \frac{\mu}{2}\|y-x\|^2$. Hessian $\succeq \mu I$.

> **Saying it out loud.** Strong convexity says the function isn't just bowl-shaped, it's at least as curved as a specific quadratic — there's a minimum amount of upward bend, $\mu$, in every direction. Convex alone allows arbitrarily flat regions where you can wander a long way without the value changing; strong convexity forbids that. That's why it upgrades convergence from slow polynomial to geometric, and adding an L2 penalty makes any convex problem strongly convex, which is one underappreciated reason regularization speeds up training.

**5. Why is convexity nice?**
Every local min is global. GD converges. Strong convexity → geometric convergence.

> **Saying it out loud.** Because every local minimum is the global minimum, so once your gradient is zero you're done — there's no question of whether a better solution is hiding somewhere else. That turns optimization from a search problem into a mechanical one, and it means results are reproducible: different initializations land in the same place. Add strong convexity and you get geometric convergence, meaning error shrinks by a constant factor every step, so you can actually predict how long you'll need to run.

**6. Is deep learning convex?**
No — non-convex in parameters. But most local minima are reasonable in over-parameterized regime.

> **Saying it out loud.** Not remotely — a neural network loss is wildly non-convex, with countless local minima and saddle points, and none of the classical guarantees apply. The interesting empirical fact is that it doesn't seem to matter much: in the over-parameterized regime most local minima reach similar loss, so where you land is less important than theory would suggest. There's also a permutation symmetry point worth making — you can swap two hidden units and get an identical function, so a huge number of those distinct minima are literally the same solution relabeled.

**7. $\ell_1$ norm — convex?**
Yes. Even non-smooth (kink at 0), but convex.

> **Saying it out loud.** Yes. The kink at zero means it's not differentiable there, but convexity and differentiability are separate properties — the absolute value is a perfectly good convex function with a corner. That's exactly why lasso is a tractable problem despite the non-smoothness, and why you handle it with subgradients or a proximal step instead of plain gradient descent. Non-smooth but convex is a comfortable place to be; smooth but non-convex is the hard one.

**8. Composition of convex functions — always convex?**
No. Convex of convex isn't necessarily convex. Convex + non-decreasing of convex is convex (e.g., $\exp(\|x\|)$ is convex).

> **Saying it out loud.** No, and this is a favorite trap. The rule is that a convex, non-decreasing function of a convex function is convex — the non-decreasing part is what does the work. So $\exp$ of a norm is convex, because exp is increasing. But square a function that goes negative and the composition can fail, and negating breaks it immediately. Getting this right is one of those small things that signals whether you've actually used convex analysis or just read about it.

---

## B. Smoothness and gradient descent

**9. Define $L$-smooth.**
Gradient is $L$-Lipschitz: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$. Equivalently $\nabla^2 f \preceq L I$.

> **Saying it out loud.** Smooth here doesn't mean differentiable — it means the gradient can't change too fast, bounded by a constant $L$. Equivalently, the curvature is capped: no direction curves more sharply than $L$. The reason it matters is that it tells you how far you can trust your current gradient, and therefore how big a step is safe. Every step-size rule in optimization is fundamentally an answer to that question.

**10. Descent lemma?**
$f(y) \leq f(x) + \nabla f(x)^\top(y-x) + \frac{L}{2}\|y-x\|^2$.

> **Saying it out loud.** The descent lemma says the function is bounded above by its linear approximation plus a quadratic correction term scaled by $L$. So you can't be surprised by more than a known amount when you take a step. It's the workhorse of every convergence proof: minimize the right-hand side over the step size and you immediately get both the optimal step and a guaranteed amount of progress per iteration.

**11. Optimal step size for $L$-smooth GD?**
$\eta = 1/L$. Gives steepest decrease per step.

> **Saying it out loud.** One over $L$ — you get it by minimizing the descent lemma's upper bound over the step size. The intuition is that $L$ is the worst-case curvature, so you step conservatively enough that even in the sharpest direction you don't overshoot. Go past $2/L$ and you provably diverge on a quadratic. In deep learning nobody knows $L$, so this becomes a tuning exercise, but the shape of the rule survives — sharper curvature demands smaller steps.

**12. GD rate for smooth convex?**
$f(x_k) - f^* = O(1/k)$.

> **Saying it out loud.** Error falls like one over the number of steps, which is slow — to get another decimal place of accuracy you need ten times as many iterations. That's the price of having no lower bound on curvature: the function can be nearly flat near the optimum, so progress crawls. It's also the baseline that makes acceleration impressive, since Nesterov turns it into one over $k$ squared for free.

**13. GD rate for smooth strongly convex?**
$\|x_k - x^*\|^2 \leq (1 - \mu/L)^k \|x_0 - x^*\|^2$. Geometric.

> **Saying it out loud.** Now you get geometric convergence — every step multiplies your error by a constant factor less than one, so accuracy improves exponentially with iterations. The factor is one minus $\mu$ over $L$, which is one minus the inverse condition number. That's the key connection: if the condition number is 1,000, your factor is 0.999 and you'll need thousands of steps. Strong convexity buys you the exponential rate; conditioning determines whether it's fast enough to matter.

**14. GD rate for non-convex smooth?**
$\min_{k' \leq k}\|\nabla f(x_{k'})\|^2 = O(1/k)$. Convergence to stationary point only.

> **Saying it out loud.** All you can promise is that the smallest gradient norm you've seen shrinks like one over $k$ — that is, you approach a stationary point. Not a minimum, a stationary point, which could be a saddle. That's the honest theoretical position for deep learning: nobody can prove you'll find a good solution, and the reason it works anyway is empirical. Saying that clearly is better than pretending the guarantees are stronger than they are.

**15. What's Nesterov acceleration?**
Modified momentum that achieves $O(1/k^2)$ for convex (vs $O(1/k)$ for GD). Quadratic improvement.

> **Saying it out loud.** Nesterov's trick is to look ahead — compute the gradient at where momentum is about to carry you, rather than where you currently are, so you can correct before overshooting. The result is a genuine improvement in the rate, from one over $k$ to one over $k$ squared for convex problems, and it's provably the best any first-order method can do. It's not just heuristic momentum; it's optimal in a formal sense. Deep learning still uses plain heavy-ball momentum most of the time because the theoretical advantage doesn't reliably survive stochastic gradients.

**16. Condition number $\kappa$?**
$L/\mu$ for strongly convex + smooth. Bigger $\kappa$ = worse conditioning = slower convergence.

> **Saying it out loud.** The condition number is the ratio of the sharpest curvature to the flattest, and it measures how stretched your bowl is. If it's 1 the level sets are circles and gradient descent walks straight to the bottom; if it's 1,000 they're long thin ellipses and you zigzag across the valley making almost no progress along it. Every conditioning technique in deep learning — normalization layers, Adam, feature standardization — is an attempt to make that number smaller.

**17. Why is Nesterov better for big $\kappa$?**
GD: $O(\kappa \log 1/\epsilon)$ iterations. Nesterov: $O(\sqrt{\kappa} \log 1/\epsilon)$. Quadratic improvement in conditioning dependence.

> **Saying it out loud.** Plain gradient descent needs a number of iterations proportional to the condition number; Nesterov needs the square root of it. So at $\kappa$ equal to 10,000, that's 10,000 steps versus 100 — a hundredfold difference, not a constant-factor tweak. And the worse your conditioning, the bigger the win, which is exactly when you need it. That square-root dependence is provably optimal for first-order methods, so nothing does asymptotically better without using second-order information.

---

## C. Second-order methods

**18. Newton's method update?**
$x_{k+1} = x_k - H_k^{-1} \nabla f(x_k)$.

> **Saying it out loud.** Newton multiplies the gradient by the inverse Hessian, which amounts to fitting a quadratic bowl to your local landscape and jumping straight to its bottom. That's why it's so fast — it uses curvature to figure out not just which way to go but exactly how far. It also makes it scale-invariant: rescale your parameters and Newton takes the same path, whereas gradient descent's behavior changes completely.

**19. Convergence rate of Newton's near optimum?**
Quadratic — number of correct digits doubles per iteration. Requires: starting close enough to the optimum, $H$ Lipschitz-continuous, and $H \succ 0$ at the optimum (strong convexity locally). Far from the optimum, Newton can diverge or step in the wrong direction.

> **Saying it out loud.** Quadratic, meaning the number of correct digits doubles every iteration — you go from two digits to four to eight, so a handful of steps is often all you need. The caveats are load-bearing though: you have to start close enough, the Hessian has to be well-behaved, and it has to be positive definite at the optimum. Far from the solution Newton can happily step toward a saddle or diverge, which is why practical implementations always add a line search or a trust region.

**20. Cost of Newton per step?**
$O(n^3)$ to invert Hessian (or $O(n^2)$ to solve).

> **Saying it out loud.** Cubic in the number of parameters if you invert the Hessian, quadratic if you just solve the linear system, plus quadratic memory to store it. For a thousand parameters that's fine. For a billion, storing the Hessian alone would need something like $10^{18}$ numbers, which is more memory than exists. So the barrier isn't theoretical elegance, it's that the object physically doesn't fit.

**21. Why doesn't Newton work for deep learning?**
$n \sim 10^9$ → can't form/invert. Loss non-convex → Hessian not PSD → can step in wrong direction. Stochastic batches → noisy Hessian.

> **Saying it out loud.** Three separate reasons, each independently fatal. Size — you cannot store or invert a matrix with $10^9$ rows. Non-convexity — the Hessian isn't positive definite, so the Newton direction can point uphill and take you straight to a saddle. And stochasticity — you're estimating the Hessian from a minibatch, so it's noisy, and inverting a noisy matrix amplifies that noise catastrophically. Methods like K-FAC and Shampoo attack all three with structured approximations, which is why they're getting traction now.

**22. BFGS vs L-BFGS?**
BFGS: stores full $n \times n$ Hessian approximation. L-BFGS: stores last $m$ gradient differences ($O(mn)$ memory). L-BFGS standard for medium-scale convex problems.

> **Saying it out loud.** BFGS builds an approximation to the inverse Hessian from the sequence of gradients it has seen, without ever forming the Hessian itself — but it still stores a full $n$-by-$n$ matrix. L-BFGS keeps only the last $m$ gradient and position differences, usually 5 to 20 of them, and reconstructs the action of that matrix on the fly. So memory drops from $n^2$ to about $10n$. It's excellent for smooth deterministic problems with up to millions of parameters, and it falls apart with stochastic gradients, because the curvature estimates it builds from noisy differences are garbage.

**23. Gauss-Newton — when?**
Least-squares with residual $r$: approximate Hessian by $J^\top J$. Always PSD. Cheap when $J$ is reasonable size.

> **Saying it out loud.** For least-squares problems you can drop the awkward part of the Hessian and approximate it as $J^\top J$, the Jacobian times its transpose. That's cheap and, crucially, always positive semidefinite, so you never get a step in the wrong direction. Levenberg-Marquardt is the version everyone actually uses, adding a damping term that interpolates between Gauss-Newton and gradient descent depending on how well it's working. It's the standard tool for nonlinear curve fitting and bundle adjustment.

**24. K-FAC, Shampoo — what are they?**
Block-diagonal / Kronecker-factored approximations to the Hessian. Cheap second-order methods for deep networks.

> **Saying it out loud.** They're attempts to get second-order behavior at a price you can afford, by assuming the Hessian has structure. K-FAC approximates each layer's curvature as a Kronecker product of two much smaller matrices, which turns an impossible inverse into two small ones. Shampoo does something similar with per-dimension preconditioners. They cost more per step than Adam but can converge in meaningfully fewer steps, and the recent AlgoPerf results made them credible rather than academic. The open question is always whether the overhead pays for itself at your scale.

---

## D. Stochastic methods

**25. SGD update?**
$x_{k+1} = x_k - \eta \nabla f_{i_k}(x_k)$ for random index $i_k$.

> **Saying it out loud.** Instead of computing the gradient over the entire dataset, compute it on one example or one minibatch and step immediately. The estimate is noisy but unbiased, so on average you're heading the right way. The reason it wins isn't accuracy per step, it's that you get thousands of updates in the time full-batch gradient descent takes to make one. Progress per unit compute is what matters, and noisy-and-frequent beats exact-and-rare by a wide margin.

**26. Why is SGD faster than full GD per epoch?**
Each step costs $O(1)$ instead of $O(n)$. With $n$ huge, SGD is way more iterations per dataset pass.

> **Saying it out loud.** Because a full-batch gradient over a million examples costs a million times more than a minibatch gradient, and it does not give you a million times better direction — after a few hundred samples the estimate is already pointing roughly the right way. So you're paying enormously for precision you don't need. On a million-example dataset, one epoch of SGD gives you thousands of parameter updates while full-batch gives you exactly one.

**27. SGD vs GD convergence rate?**
GD strongly convex: $O(\kappa \log 1/\epsilon)$. SGD strongly convex: $O(1/\epsilon)$. SGD has worse asymptotic rate but each step cheaper.

> **Saying it out loud.** On paper SGD looks worse — one over epsilon versus a logarithmic dependence — because the gradient noise doesn't vanish as you approach the optimum, so you rattle around in a noise ball instead of converging cleanly. That's why you decay the learning rate. But the comparison is per-iteration, and SGD's iterations are hundreds or thousands of times cheaper, so per unit of compute it wins decisively at scale. It's the classic case where the asymptotic rate is the wrong thing to compare.

**28. Linear scaling rule (Goyal et al.)?**
When you scale batch size by $k$, scale LR by $k$ — keeps the same effective update.

> **Saying it out loud.** Double your batch size, double your learning rate. The logic is that a batch twice as large has a gradient estimate with less noise, so you can afford a proportionally bigger step and end up with the same total movement per example seen. Goyal and colleagues used it to train ImageNet at batch size 8,192 in an hour without losing accuracy. Two practical caveats: you need a warmup period, because the rule breaks in the chaotic first few hundred steps, and it stops working past the critical batch size.

**29. Up to what batch size?**
Critical batch size — beyond that, returns diminish (McCandlish et al., 2018). Different per task.

> **Saying it out loud.** Up to the critical batch size, past which doubling the batch stops buying you proportionally fewer steps — you're paying twice the compute for a marginal gain. Below it you're noise-limited and bigger batches genuinely help; above it you're curvature-limited and they don't. McCandlish and colleagues showed it's predictable from the gradient noise scale, and that it grows during training as the gradient signal shrinks. It's a genuinely useful number because it tells you when adding GPUs stops making your run finish sooner.

---

## E. Constrained optimization

**30. State the Lagrangian.**
$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum \lambda_i g_i(x) + \sum \nu_j h_j(x)$, $\lambda \geq 0$.

> **Saying it out loud.** The Lagrangian folds the constraints into the objective, each multiplied by its own price — that's what the multipliers are, prices for violating a constraint. Instead of a constrained problem you now have an unconstrained one whose stationary points encode the constrained solution. The multipliers for inequality constraints have to be non-negative, because they can only push you back into the feasible region, never pull you out. Their values have a genuine economic reading: how much the optimum would improve if you relaxed that constraint slightly.

**31. Four KKT conditions?**
Stationarity ($\nabla_x \mathcal{L} = 0$), primal feasibility, dual feasibility ($\lambda \geq 0$), complementary slackness ($\lambda_i g_i = 0$).

> **Saying it out loud.** Stationarity — the gradient of the Lagrangian vanishes. Primal feasibility — your solution satisfies the original constraints. Dual feasibility — the inequality multipliers are non-negative. And complementary slackness — each multiplier times its constraint is zero. Together they're necessary at any optimum under mild conditions, and for convex problems they're sufficient too, meaning anything satisfying all four is the answer. That's what makes them a practical certificate rather than just theory.

**32. What's complementary slackness?**
Either constraint is active ($g_i = 0$) or its multiplier is zero. Encodes: only active constraints have non-trivial influence.

> **Saying it out loud.** It says for each constraint either the constraint is tight or its multiplier is zero — you can't have both slack and a nonzero price. Intuitively, a constraint you aren't pressing against isn't influencing your solution, so its price is zero. That's exactly what makes SVMs sparse: every training point that's comfortably on the right side of the margin has a zero multiplier and drops out entirely, leaving only the support vectors.

**33. SVM support vectors via KKT?**
Hard-margin: $\lambda_i = 0$ for non-support vectors; $\lambda_i > 0$ only for points exactly on the margin. Soft-margin (with box constraint $0 \leq \lambda_i \leq C$): $\lambda_i = 0$ off-margin; $0 < \lambda_i < C$ exactly on margin; $\lambda_i = C$ for margin violators.

> **Saying it out loud.** Complementary slackness partitions your training data. Points comfortably classified have multiplier zero and contribute nothing to the final model — you could delete them and get the same boundary. Points exactly on the margin have a multiplier strictly between zero and the cap. Points violating the margin sit pinned at the cap $C$. That three-way split is exactly what makes SVMs sparse and interpretable, and it's why the model size depends on how hard the problem is rather than how much data you have.

**34. Strong duality — when?**
For convex problems satisfying constraint qualifications (e.g., Slater's condition: strictly feasible point exists). $g(\lambda^*) = f(x^*)$.

> **Saying it out loud.** Strong duality means the dual problem's optimum exactly equals the primal's, with no gap, so you can solve whichever is easier. For convex problems it holds under mild conditions, and Slater's is the usual one: there just has to exist a point strictly inside the constraints. Weak duality — the dual being a lower bound — always holds, even for non-convex problems, which is why dual bounds are useful in integer programming even when the gap doesn't close.

**35. Why is SVM dual easier than primal?**
Many fewer variables (one $\lambda$ per training example, but most are zero). Plus kernel trick fits naturally into the dual.

> **Saying it out loud.** Two reasons. The dual has one variable per training example rather than one per feature, which is a huge win when features outnumber examples, and most of those variables come out exactly zero. More importantly, the data appears in the dual only as inner products between pairs of points — which means you can swap those inner products for a kernel and get a nonlinear classifier without ever computing the high-dimensional feature map. That's the kernel trick, and it only works because the dual has that structure.

---

## F. Deep learning loss landscape

**36. Saddle points vs local minima in high dim?**
Saddle points dominate. With many dimensions, the chance all eigenvalues of a random Hessian are positive is low.

> **Saying it out loud.** In high dimensions, being a local minimum means every one of a billion Hessian eigenvalues is positive, and that's an extraordinarily unlikely coincidence. A saddle only needs a mix, which is overwhelmingly the typical case. So the classic worry about getting trapped in bad local minima was largely misplaced — the real obstacle is saddle points and the long flat plateaus around them, where the gradient is nearly zero and progress stalls.

**37. How does SGD escape saddle points?**
Stochastic noise in gradient provides random kicks; one of them usually has component in negative-curvature direction.

> **Saying it out loud.** The gradient noise from minibatch sampling acts like a random kick, and near a saddle there's at least one direction of negative curvature, so any random perturbation with a component along it gets amplified — you slide off. Full-batch gradient descent has no such noise and can sit at a saddle essentially forever. It's a good example of a quirk that turns out to be a feature: the noise you'd naively want to eliminate is doing useful work.

**38. Flat vs sharp minima — generalization?**
Flat → better generalization empirically. Hessian eigenvalues correlate with train-test gap.

> **Saying it out loud.** The empirical observation is that solutions in flat basins generalize better than ones in sharp ones. The intuition is robustness — if the loss barely changes when you perturb the weights, it probably also barely changes when the data distribution shifts a little, whereas a sharp minimum is finely tuned to the exact training set. Large-batch training tends to find sharper minima, which is one explanation for the large-batch generalization gap. The caveat worth stating is that sharpness isn't reparameterization-invariant, so the definition is slipperier than it sounds.

**39. Why does SGD prefer flat minima?**
Stochastic noise can't keep you in a sharp basin (small fluctuations push you out). Flat basins are more "stable" under noise.

> **Saying it out loud.** Because gradient noise makes a sharp minimum unstable — you're constantly getting kicked, and in a narrow basin a small kick puts you over the wall, while a wide flat basin absorbs it. So the noise acts like a filter that only lets you settle where the landscape is forgiving. This is implicit regularization: nobody wrote it into the loss, it falls out of the algorithm. And it's the direct motivation for SAM, which makes the preference explicit by optimizing worst-case loss in a neighborhood.

**40. Edge of stability?**
With full-batch GD, top Hessian eigenvalue grows until $\lambda_{\max} \approx 2/\eta$, then oscillates. Loss bounces but decreases. Classical stability bound violated.

> **Saying it out loud.** Classical theory says gradient descent is stable only while the sharpest curvature stays below $2/\eta$. What actually happens in deep learning is that the sharpness rises during training until it hits that threshold and then hovers right at it, with the loss oscillating but still trending down. So training spends most of its time in a regime the theory says should diverge. It's one of the more interesting recent empirical findings, and the practical reading is that the learning rate is implicitly selecting how sharp a solution you'll end up in.

**41. Why does loss decrease at edge of stability despite oscillation?**
Implicit regularization toward flat regions. Even with oscillation, the average trajectory progresses.

> **Saying it out loud.** Because the oscillation happens along the sharpest direction while genuine progress happens along all the others. You're bouncing across a narrow valley and still moving down it. Better than that, the bouncing actively pushes you toward regions of lower sharpness, so it's self-correcting — it's implicit regularization toward flatness rather than a failure mode. Which means a slightly-too-large learning rate is often better than a safe one, and that's not something classical optimization would predict.

**42. Lottery ticket hypothesis (Frankle & Carbin)?**
Dense networks contain sparse subnetworks ("winning tickets") that match performance when trained from same init. Implies optimization finds good solutions specific to init.

> **Saying it out loud.** The claim is that a big dense network contains a small subnetwork which, if you rewind it to its original initialization and train it alone, matches the full network's performance. The striking part is the rewinding — reinitialize those same weights randomly and it doesn't work, so what matters is the specific combination of structure and initial values. The implication is that training is partly a search for a lucky subnetwork that was there from the start. The practical caveat is that you have to train the full network first to find the ticket, so it hasn't delivered cheap training, mostly insight.

---

## G. Conditioning in deep learning

**43. Why does normalization help conditioning?**
Renormalizes activations → reduces variance in Hessian eigenvalues across layers → better conditioned.

> **Saying it out loud.** Without normalization the scale of activations drifts wildly across layers, and that shows up as Hessian eigenvalues spread over orders of magnitude — a badly stretched bowl. Normalizing pins every layer's activations to a comparable scale, which compresses that eigenvalue spread and lowers the condition number. Better conditioning means you can use a much larger learning rate safely, which is the actual observed benefit — around ten times larger for BatchNorm in ResNets.

**44. Why does Adam help conditioning?**
Per-parameter step ≈ diagonal preconditioning. Approximately rescales each axis by historical gradient magnitude.

> **Saying it out loud.** Adam divides each parameter's step by the running root-mean-square of its own gradients, which is a diagonal preconditioner — parameters with consistently large gradients take proportionally smaller steps. That approximately equalizes the effective curvature across axes, so you're not forced to pick a learning rate small enough for the sharpest direction. It's a crude approximation to second-order information, only the diagonal, but it's essentially free and it's why Adam trains transformers where plain SGD struggles.

**45. Standardize input features — why?**
Inputs of different scales cause Hessian to have very different eigenvalues across input directions. Standardize → balanced.

> **Saying it out loud.** If one feature is measured in dollars ranging to millions and another is a 0-to-1 flag, the loss surface is enormously stretched — the weight on the dollar feature has huge curvature and the other has almost none. Gradient descent then zigzags, since any step size safe for the first is uselessly small for the second. Standardizing puts them on comparable footing and the condition number collapses. It's the cheapest optimization improvement available and the most commonly skipped.

**46. What's a "stiff" direction in parameter space?**
Direction with large Hessian eigenvalue — function curves sharply along it, so it caps the usable step size. The complementary flat ("sloppy") direction, with small eigenvalue, is the one that needs many steps. Having both at once is exactly a large condition number.

> **Saying it out loud.** A stiff direction is one with large curvature — the loss shoots up fast if you move along it, so it's what caps your learning rate. Its opposite is the flat or sloppy direction, where the loss barely changes and you need enormous numbers of steps to make progress. The pain is having both at once: the stiff direction forces a tiny step size, and with that step size the flat direction takes forever. That's exactly what a large condition number means, and why preconditioning — Adam, normalization, feature scaling — is the fix.

---

## H. Subtleties

**47. Subgradient — what is it?**
Generalization of gradient for non-smooth convex functions. For $\ell_1$: $\partial |x| = \mathrm{sign}(x)$ if $x \neq 0$; $\partial |0| = [-1, 1]$.

> **Saying it out loud.** At a kink there's no unique tangent, so instead of one gradient you get a whole set of slopes that all stay below the function. For the absolute value at zero, that's every slope between $-1$ and $1$. Any element of that set gives you a valid descent-ish direction, so subgradient descent works — just slowly, at one over the square root of $k$ rather than one over $k$. The reason you'd tolerate it is that lasso and hinge loss are non-smooth by design, and their kinks are where the useful behavior lives.

**48. Proximal gradient — when?**
Composite objectives like $f + g$ where $f$ smooth, $g$ non-smooth (e.g., $\ell_1$). Step: $x_{k+1} = \mathrm{prox}_{\eta g}(x_k - \eta \nabla f(x_k))$. Used for lasso (ISTA).

> **Saying it out loud.** When your objective splits into a smooth part and a nasty-but-simple part, like a smooth loss plus an L1 penalty. You take an ordinary gradient step on the smooth piece and then apply the proximal operator of the other piece, which for L1 is just soft thresholding — shrink everything toward zero and clamp small values to exactly zero. That's how you get genuine sparsity instead of the near-zero-but-not-zero weights subgradient descent gives you. ISTA and its accelerated version FISTA are the standard implementations.

**49. ADMM?**
Alternating Direction Method of Multipliers. Splits an objective into easier subproblems via auxiliary variables. Used for distributed convex optimization.

> **Saying it out loud.** ADMM splits a hard problem into two easy ones by introducing a copy of the variable and a constraint that the copies agree, then alternates: solve one subproblem, solve the other, update the multiplier enforcing agreement. The point is that each subproblem can be something you have a closed form for, even when the combined problem has none. It's especially good for distributed settings, since each machine can solve its own piece and you only exchange the consensus variable. The tradeoff is slow tail convergence — it gets you to a decent answer quickly and to a precise one slowly.

**50. Lagrangian dual is concave — true?**
Yes. The dual function $g(\lambda) = \inf_x \mathcal{L}(x, \lambda)$ is concave (infimum of affine functions is concave). So dual problem is convex regardless of primal.

> **Saying it out loud.** Yes, always, and the proof is a one-liner worth knowing: the dual function is an infimum over $x$ of functions that are affine in the multipliers, and a pointwise infimum of affine functions is concave. Notice nothing there assumed the primal was convex. So the dual problem is a convex optimization problem even when the primal is a horrible non-convex mess — which is why dual bounds are useful for combinatorial problems where you'll never solve the primal exactly.

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
