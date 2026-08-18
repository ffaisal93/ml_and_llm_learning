# Whiteboard Derivations — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This deep dive is the catalog of derivations you should be able to do on a whiteboard cold. Frontier-lab interviews routinely ask "derive X" — backprop, attention, OLS gradient, KL, EM, DPO. Knowing the *shape* of these derivations beats memorizing the answer.

This is a meta-document that points to the relevant deep dive for each derivation while listing the key steps you need to recite.

---

## 1. Backpropagation for a 2-layer MLP

**Setup**: $z_1 = W_1 x + b_1$, $h_1 = \sigma(z_1)$, $z_2 = W_2 h_1 + b_2$, $\hat{y} = \mathrm{softmax}(z_2)$, $\mathcal{L} = -\sum y \log \hat{y}$.

**Steps**:
1. **Cross-entropy + softmax simplification** (the magic step — derive it, don't just assert):

   Softmax Jacobian: $\partial \hat y_i / \partial z_{2,j} = \hat y_i (\delta_{ij} - \hat y_j)$.

   $\partial \mathcal{L}/\partial \hat y_i = -y_i / \hat y_i$.

   $\partial \mathcal{L}/\partial z_{2,j} = \sum_i \frac{\partial \mathcal{L}}{\partial \hat y_i} \frac{\partial \hat y_i}{\partial z_{2,j}} = -\sum_i \frac{y_i}{\hat y_i} \hat y_i (\delta_{ij} - \hat y_j) = -y_j + \hat y_j \sum_i y_i = \hat y_j - y_j$ (using $\sum y_i = 1$).

   So $\delta_2 = \hat y - y$.
2. $\nabla_{W_2} \mathcal{L} = \delta_2 h_1^\top$.
3. $\nabla_{b_2} \mathcal{L} = \delta_2$.
4. $\delta_1 = W_2^\top \delta_2 \odot \sigma'(z_1)$.
5. $\nabla_{W_1} \mathcal{L} = \delta_1 x^\top$.
6. $\nabla_{b_1} \mathcal{L} = \delta_1$.

**Key insights**:
- Cross-entropy + softmax simplifies dramatically: gradient is just $\hat y - y$. The mess from softmax's Jacobian and CE's $1/\hat y$ cancel.
- Chain rule: each layer multiplies by $W^\top$ (transpose) and $\sigma'$.

See `31_neural_networks/`.

> **Saying it out loud (narrate while you write).** "I'll write the forward pass across the top first, so I always have something to point at. Now backwards. The magic step is the last layer, so let me actually derive it rather than assert it: the softmax Jacobian is `y-hat_i` times `delta_ij minus y-hat_j`, and cross-entropy contributes `minus y_i over y-hat_i`. Multiply them and the `y-hat_i` cancels — I'm left with `minus y_j plus y-hat_j` times the sum of `y`, which is one because `y` is a distribution. So delta-2 is just `y-hat minus y`." Then the mechanical part, and say the pattern out loud: every layer does the same three things — the weight gradient is the incoming delta times the layer's input transposed, the bias gradient is just the delta, and to push further back you multiply by `W` transpose and gate by the activation derivative. Land it on shapes: if `dW` doesn't come out the same shape as `W`, you've transposed something, and that's the single most common whiteboard slip here.

---

## 2. Scaled dot-product attention

**Setup**: $Q, K, V \in \mathbb{R}^{L \times d}$.

**Steps**:
1. $\mathrm{scores} = QK^\top / \sqrt{d}$.
2. $\mathrm{attn} = \mathrm{softmax}(\mathrm{scores})$.
3. $\mathrm{output} = \mathrm{attn} \cdot V$.

**Why $\sqrt{d}$**: variance of $QK^\top$ entries scales with $d$ if $Q, K$ have unit-variance entries. Divide by $\sqrt{d}$ to keep variance at 1 → softmax doesn't saturate.

**Multi-head**: project to $h$ heads of dim $d/h$; do attention per head; concatenate; project back.

See `04_transformers/`, `05_attention_mechanisms/`.

> **Saying it out loud (narrate while you write).** "Three lines and then I'll justify the scaling. Scores are `Q K` transpose over root `d` — I'll draw the box, `L` by `L`, and say what it is: token `i`'s row tells me how much it attends to every other token. Softmax along that row, over keys, so each row sums to one. Then times `V`, which contracts the sequence dimension and hands me back `L` by `d_v`." Now the question they always ask: "Why root `d`? Because a dot product of two `d`-dimensional unit-variance vectors has variance `d`, so the raw scores grow like root `d`. Feed those into a softmax and it saturates into a near one-hot, where the gradient is essentially zero. Dividing by root `d` puts the variance back at 1." Close with the cost: that `L` by `L` box is quadratic in time and memory, which is the reason every long-context trick exists.

---

## 3. OLS closed form

**Setup**: $\mathcal{L}(w) = \frac{1}{2}\|y - Xw\|^2$.

**Steps**:
1. $\nabla_w \mathcal{L} = -X^\top (y - Xw) = X^\top Xw - X^\top y$.
2. Set to zero: $X^\top Xw = X^\top y$.
3. Solve: $\hat{w} = (X^\top X)^{-1} X^\top y$ (assuming $X^\top X$ invertible).

**Hessian**: $\nabla^2 \mathcal{L} = X^\top X$ — PSD always; PD if $X$ has full column rank.

**Geometric**: $\hat{y} = Py$ where $P = X(X^\top X)^{-1}X^\top$ is the projection onto $\mathrm{Col}(X)$.

See `24_linear_algebra_qa/`, `48_optimization_and_matrix_calculus/`.

> **Saying it out loud (narrate while you write).** "Half the squared residual norm, and I want the gradient, so I'll expand it: `y` transpose `y` minus `2 w` transpose `X` transpose `y` plus `w` transpose `X` transpose `X w`, all over two. Differentiating, the linear term gives `minus X` transpose `y` and the quadratic gives `X` transpose `X w` — so the gradient is `X` transpose times the residual, negated. Set it to zero and you get the normal equations." Then say the thing that scores: "the Hessian is `X` transpose `X`, which is positive semi-definite for any `X`, so this problem is convex and the stationary point is the global minimum — and it's strictly positive definite, hence uniquely solvable, exactly when `X` has full column rank." Finish geometrically: the fitted values are an orthogonal projection of `y` onto the column space of `X`, which is why the residual is perpendicular to every feature. And name the failure mode: collinear features make `X` transpose `X` singular, which is why you solve with QR or add ridge rather than literally inverting.

---

## 4. Logistic regression gradient

**Setup**: $p = \sigma(w^\top x)$, $\mathcal{L} = -[y \log p + (1-y)\log(1-p)]$.

**Steps**:
1. $\partial \mathcal{L}/\partial p = -y/p + (1-y)/(1-p) = (p - y)/(p(1-p))$ (combine fractions).
2. $\partial p/\partial z = \sigma(z)(1-\sigma(z)) = p(1-p)$ (sigmoid derivative).
3. **Chain rule — the magic cancellation**: $\partial \mathcal{L}/\partial z = \frac{p - y}{p(1-p)} \cdot p(1-p) = p - y$. The $p(1-p)$ from sigmoid derivative kills the $p(1-p)$ in the denominator from CE — that's the GLM canonical-link beauty.
4. $\nabla_w \mathcal{L} = (p - y) \, x$ (since $z = w^\top x$, $\partial z/\partial w = x$).

**Key insight**: same gradient form as linear regression (residual times input) — that's why these models feel the same. Hessian is $\sum p(1-p) x x^\top$, always PSD → loss convex.

See `01_classical_ml/`, `37_mle_map_estimation/`.

> **Saying it out loud (narrate while you write).** "Chain rule through the sigmoid, and I'll show the cancellation because that's the whole point. Differentiating binary cross-entropy with respect to `p` gives `minus y over p plus one minus y over one minus p`; over a common denominator that's `p minus y` on top and `p times one minus p` underneath. Now the sigmoid derivative is `p times one minus p`" — write it directly below — "so multiplying, those cancel" — strike them out — "and `dL/dz` is `p minus y`. Times `dz/dw`, which is `x`, gives residual times input." Then land the two payoffs: it's the same form as linear regression, which is the canonical-link property of generalized linear models, and the Hessian is a sum of `p(1-p) x x` transpose, which is PSD, so the loss is convex and there are no local minima to worry about. Named failure mode to mention: on separable data that convex loss has no finite minimizer, so the weights diverge unless you regularize.

---

## 5. KL divergence

*In plain language:* KL divergence is a number that says how badly one probability distribution stands in for another. If you built a code assuming distribution `q` but reality is `p`, KL is the extra bits you waste per symbol. It's zero only when the two match, and it is not symmetric, so the order of the arguments genuinely matters.

**Definition**: $\mathrm{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$.

**Properties**:
- $\geq 0$, with equality iff $p = q$ (Gibbs' inequality). **Proof via Jensen** (memorize this — most-asked):

  $-\mathrm{KL}(p\|q) = \sum_x p(x) \log \frac{q(x)}{p(x)}$. Since $\log$ is concave, **Jensen's inequality** gives $\sum p(x) \log \frac{q}{p} \leq \log \sum p(x) \cdot \frac{q(x)}{p(x)} = \log \sum q(x) = \log 1 = 0$. So $-\mathrm{KL} \leq 0$, i.e. $\mathrm{KL} \geq 0$. Equality iff $q/p$ is constant, i.e. $p = q$ (since both are distributions).

- Asymmetric: $\mathrm{KL}(p \| q) \neq \mathrm{KL}(q \| p)$.
- Forward KL ($\mathrm{KL}(p^* \| q)$): mean-seeking. MLE.
- Reverse KL ($\mathrm{KL}(q \| p^*)$): mode-seeking. Variational inference.

**MLE = forward KL minimization**:
$\arg\max_\theta \mathbb{E}_{p^*}[\log q_\theta(x)] = \arg\min_\theta \mathrm{KL}(p^* \| q_\theta) + H(p^*)$ — the entropy term is constant.

See `33_information_theory/`, `37_mle_map_estimation/`.

> **Saying it out loud (narrate while you write).** "KL is the expected log-likelihood ratio under `p` — the average number of extra bits I pay for using the wrong distribution. Two facts to prove, both quick. It's non-negative: I'll write minus KL, which is the expectation of `log q over p`, and since `log` is concave, Jensen lets me pull the expectation inside the log" — write it — "giving `log` of the sum of `q`, which is `log 1`, which is zero. So minus KL is at most zero, hence KL is at least zero, with equality only when the ratio is constant, meaning the distributions are identical." Then the asymmetry, said with the consequence: "forward KL is the MLE direction and it's mass-covering — `p` is out there where `q` is near zero, the log ratio explodes, so `q` is forced to smear over everything. Reverse KL is what variational inference minimizes and it's mode-seeking — `q` is penalized for putting mass where `p` has none, so it collapses onto one mode." That's the named tradeoff: blurry-but-inclusive versus sharp-but-incomplete, and it's exactly why VAEs blur and reverse-KL variational fits drop modes.

---

## 6. EM for GMM

*In plain language:* EM is what you do when your model has a hidden label you never observe — like which cluster each point came from. You alternate between guessing the hidden labels using your current parameters, and refitting your parameters as if those guesses were true. This section is mostly about why that loop can't make things worse.

**Setup**: $p(x) = \sum_k \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$.

**E-step**: posterior responsibilities

$$
\gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(x_i | \mu_j, \Sigma_j)}
$$

**M-step**: weighted MLE updates

$$
\mu_k = \frac{\sum_i \gamma_{ik} x_i}{\sum_i \gamma_{ik}}
$$

$$
\Sigma_k = \frac{\sum_i \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{\sum_i \gamma_{ik}}
$$

$$
\pi_k = \frac{\sum_i \gamma_{ik}}{N}
$$

**Why EM converges** (the key identity to memorize):

For any distribution $q(z)$:
$$
\log p_\theta(x) = \underbrace{\mathbb{E}_q[\log \tfrac{p_\theta(x, z)}{q(z)}]}_{\mathcal{L}(q, \theta) \text{ — ELBO}} + \underbrace{\mathrm{KL}(q(z) \,\|\, p_\theta(z|x))}_{\geq 0}
$$

So $\log p_\theta(x) \geq \mathcal{L}(q, \theta)$ always, with equality iff $q = p_\theta(z|x)$.

- **E-step**: set $q = p_\theta(z|x)$ (the posterior responsibilities $\gamma_{ik}$). KL = 0 → bound is tight: $\log p_\theta(x) = \mathcal{L}(q, \theta_t)$.
- **M-step**: maximize $\mathcal{L}(q, \theta)$ over $\theta$ (since $q$ is fixed, this is just weighted MLE). $\theta_{t+1}$ raises the bound.
- Net: $\log p_\theta(x_{t+1}) \geq \mathcal{L}(q, \theta_{t+1}) \geq \mathcal{L}(q, \theta_t) = \log p_\theta(x_t)$. Likelihood non-decreasing → bounded above → converges.

See `19_advanced_clustering/`.

> **Saying it out loud (narrate while you write).** "The problem is circular: if I knew which Gaussian each point came from I could fit the Gaussians, and if I knew the Gaussians I could assign the points. EM just alternates. The E-step computes soft assignments — the responsibility is the prior times the density, normalized over components, which is Bayes' rule. The M-step is ordinary maximum likelihood with those responsibilities as weights: each mean is a weighted average of the points, each covariance a weighted scatter, each mixing weight the share of total responsibility." Then the part interviewers are actually testing — why it converges: "write the log-likelihood as the ELBO plus a KL term. The E-step sets `q` to the exact posterior, which drives that KL to zero and makes the bound tight. The M-step then raises the bound. Since the bound was touching the likelihood before the step and can only have gone up, the likelihood is non-decreasing, and it's bounded above, so it converges." Land on the failure modes: convergence is to a local optimum, so initialization matters, and a component can collapse onto a single point, sending its variance to zero and the likelihood to infinity — which is why you floor the covariance.

---

## 7. PCA via SVD

**Setup**: centered $X \in \mathbb{R}^{n \times d}$.

**Steps**:
1. Center the data, compute covariance: $\Sigma = X^\top X / n$.
2. SVD of centered $X$: $X = U S V^\top$ with $U^\top U = I$, $V^\top V = I$.
3. **Substitute and simplify**: $X^\top X = (USV^\top)^\top (USV^\top) = V S U^\top U S V^\top = V S^2 V^\top$ (using $U^\top U = I$ — that's the load-bearing step). So $\Sigma = V (S^2/n) V^\top$ — this is the eigendecomposition of $\Sigma$.
4. Top-$k$ principal directions: columns of $V$. Variances along them: $S^2/n$.
5. Reduced data: $X V_k = U_k S_k$ (project data onto top-$k$ directions).

**Eckart-Young**: truncated SVD $X_k = U_k S_k V_k^\top$ minimizes $\|X - \tilde{X}\|_F^2$ over rank-$k$ $\tilde{X}$.

See `21_dimensionality_reduction/`.

> **Saying it out loud (narrate while you write).** "PCA asks for the directions of maximum variance, which are the top eigenvectors of the covariance matrix — but I'd never actually form the covariance matrix, and here's why. Take the SVD of the centered data, `X` equals `U S V` transpose. Now compute `X` transpose `X`" — write it out — "the `V S U` transpose times `U S V` transpose, and because `U` has orthonormal columns, `U` transpose `U` is the identity and vanishes. What's left is `V S squared V` transpose, which *is* the eigendecomposition, read straight off." So the principal directions are the columns of `V`, the variances are the singular values squared over `n`, and the projected data is `U_k S_k`. Then the reason to prefer this route: forming `X` transpose `X` squares the condition number and loses precision, while the SVD works on `X` directly. Close on Eckart-Young — truncating the SVD gives the provably best rank-`k` approximation in Frobenius norm, which is why this one decomposition underlies PCA, LSA, and low-rank compression alike.

---

## 8. SVM dual

*In plain language:* the SVM dual is a rewrite. Instead of solving for a weight vector directly, you solve for one weight per training point, and the answer turns out to depend on the data only through pairwise dot products. That rewrite is what makes kernels possible, and it's the whole reason anyone bothers.

**Primal**: $\min_w \frac{1}{2}\|w\|^2$ s.t. $y_i(w^\top x_i + b) \geq 1$.

**Lagrangian**: $\mathcal{L} = \frac{1}{2}\|w\|^2 - \sum_i \alpha_i [y_i(w^\top x_i + b) - 1]$.

**Steps**:
1. $\partial \mathcal{L}/\partial w = w - \sum_i \alpha_i y_i x_i = 0 \implies w^* = \sum_i \alpha_i y_i x_i$.
2. $\partial \mathcal{L}/\partial b = -\sum_i \alpha_i y_i = 0 \implies \sum_i \alpha_i y_i = 0$ (constraint on $\alpha$).
3. **Substitute $w^*$ back into $\mathcal{L}$** — this is the load-bearing step:
   - $\frac{1}{2}\|w^*\|^2 = \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^\top x_j$.
   - $\sum_i \alpha_i y_i (w^{*\top} x_i) = \sum_i \alpha_i y_i \sum_j \alpha_j y_j x_j^\top x_i = \sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^\top x_j$ (the *full* quadratic).
   - $\sum_i \alpha_i y_i b = b \cdot 0 = 0$ (using $\sum \alpha_i y_i = 0$).
   - $\sum_i \alpha_i$ stays.
   - Combining: $\mathcal{L}(w^*, b, \alpha) = \frac{1}{2}\sum_{ij}\alpha_i\alpha_j y_i y_j x_i^\top x_j - \sum_{ij}\alpha_i\alpha_j y_i y_j x_i^\top x_j + \sum_i\alpha_i = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^\top x_j$.

**Dual**: $\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j$ s.t. $\alpha \geq 0, \sum_i \alpha_i y_i = 0$.

**Kernel trick**: replace $x_i^\top x_j$ with $K(x_i, x_j)$. The dual is the *only* place data enters as inner products — perfect for kernels.

**KKT — support vectors**: complementary slackness gives $\alpha_i > 0$ only for points where $y_i(w^\top x_i + b) = 1$ (on margin); for soft-margin with $0 \leq \alpha_i \leq C$, $\alpha_i = C$ for margin violators.

See `35_kernel_functions/`, `48_optimization_and_matrix_calculus/`.

> **Saying it out loud (narrate while you write).** "The primal is minimize half the squared norm of `w` subject to every point being on the right side of the margin. I'll form the Lagrangian with one multiplier alpha per constraint. Take the derivative with respect to `w` and set it to zero: `w` equals the sum of `alpha_i y_i x_i` — so the solution is a weighted combination of the training points, which is already the interesting part. Derivative with respect to `b` gives the constraint that the `alpha y` terms sum to zero." Then the load-bearing move: "substitute `w`-star back in. The quadratic term gives me half the double sum, the constraint term gives me the full double sum with the opposite sign, so they combine to minus a half, the `b` term dies because `alpha y` sums to zero, and the plus-ones survive as sum of alpha." Land the two payoffs: the data appears *only* as inner products `x_i` transpose `x_j`, so you swap in a kernel and get nonlinear boundaries for free, and complementary slackness means alpha is nonzero only for points sitting exactly on the margin — those are the support vectors, and everything else could be deleted without changing the answer.

---

## 9. RoPE rotation

*In plain language:* RoPE encodes a token's position by rotating its query and key vectors by an angle proportional to that position. The trick is that when you later take a dot product between two rotated vectors, the absolute angles cancel and only the difference survives — so the model sees relative position for free.

**Goal**: encode relative position via rotation in 2D subspaces.

**Setup**: pair up dimensions; for pair $(2i, 2i+1)$, apply rotation by $m \theta_i$ to position $m$:

$$
R_m = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
$$

with $\theta_i = 10000^{-2i/d}$.

**Property**: $\langle R_m q, R_n k \rangle = \langle q, R_{n-m} k \rangle$. Inner product depends only on the *relative* position $n - m$.

**Why this works** (the algebra to memorize):
- $\langle R_m q, R_n k \rangle = (R_m q)^\top (R_n k) = q^\top R_m^\top R_n k$.
- Rotations are orthogonal, so $R_m^\top = R_m^{-1} = R_{-m}$.
- Rotations also compose by *adding angles*: $R_{-m} R_n = R_{n-m}$.
- Therefore $q^\top R_{n-m} k = \langle q, R_{n-m} k \rangle$ — a function of $n - m$ only.

This is what makes attention self-positionally-aware in a *relative* way without any added position embeddings to the input.

See `14_advanced_positional_embeddings/`.

> **Saying it out loud (narrate while you write).** "Instead of adding a position vector, RoPE rotates. I pair up the dimensions, treat each pair as a point in a plane, and rotate the pair at position `m` by angle `m theta`, with a different frequency theta for each pair — fast rotations for early dimensions, slow ones for later, so together they encode position across many scales. Now here's why it gives relative position." Write the inner product: "`R_m q` dotted with `R_n k` is `q` transpose `R_m` transpose `R_n k`. Rotation matrices are orthogonal, so the transpose is the inverse, which is rotation by minus `m`. And rotations compose by adding angles, so `R_minus-m` times `R_n` is `R_(n minus m)`." Land it: the attention score depends only on `n` minus `m`, never on absolute position, which is why RoPE extrapolates far better than learned absolute embeddings — and the named failure mode is that it still degrades past the training context length, which is exactly what NTK scaling and YaRN interpolate around.

---

## 10. DPO (direct preference optimization)

*In plain language:* DPO is RLHF with the reinforcement learning taken out. The usual pipeline trains a reward model and then optimizes against it; DPO shows that if you write down the optimal policy in closed form, the reward can be expressed in terms of the policy itself, so you can train straight from preference pairs with an ordinary classification loss.

**Starting point**: RLHF objective with KL regularization to a reference policy:

$$
\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x,y)] - \beta \, \mathrm{KL}(\pi(\cdot|x) \,\|\, \pi_{\mathrm{ref}}(\cdot|x))
$$

**Step 1 — derive the closed-form optimal policy.** Set up Lagrangian on the constrained max (with $\sum_y \pi(y|x) = 1$). Setting $\partial / \partial \pi(y|x) = 0$ gives $\log \pi(y|x) = \log \pi_{\mathrm{ref}}(y|x) + r(x,y)/\beta - \log Z(x) - 1$, where $Z$ is from the normalization Lagrange multiplier. Cleaning up:

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp(r(x, y)/\beta)
$$

with $Z(x) = \sum_y \pi_{\mathrm{ref}}(y|x) \exp(r(x,y)/\beta)$ — depends only on prompt $x$, not on $y$.

**Step 2 — invert for $r$**:

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)
$$

**Step 3 — substitute into Bradley-Terry**: $p(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$. Critically, $\beta \log Z(x)$ depends on $x$ only — it appears identically in both reward terms and **cancels in the subtraction**.

**Step 4 — final DPO loss** (NLL of preferences):

$$
\mathcal{L}_{\mathrm{DPO}} = -\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)
$$

**Key insight**: closed-form optimal policy + $Z(x)$ depending only on prompt = reward model eliminates itself. No RL loop, no rollouts, just a supervised classification loss on preferences.

See `08_training_techniques/`.

> **Saying it out loud (narrate while you write).** "The starting objective is standard RLHF: maximize reward, minus beta times the KL to a reference policy so you don't drift into gibberish. Step one — this constrained problem has a closed-form solution. Set up the Lagrangian with the normalization constraint, take the functional derivative, and you get that the optimal policy is the reference policy tilted by `exp` of reward over beta, divided by a partition function `Z of x`. Step two — and this is the trick — invert that: the reward equals beta times the log ratio of policy to reference, plus beta log `Z`. Step three, plug it into Bradley-Terry, which models the probability that one response is preferred as a sigmoid of the reward difference. And here `Z of x` depends only on the prompt, so it appears in both rewards identically and cancels in the subtraction" — strike it out. "What's left has no reward model in it at all: just a log-sigmoid of the difference of log-ratios on the chosen and rejected responses." Land the tradeoff: you gain enormous simplicity and stability by dropping the RL loop and the rollouts, and you lose online exploration — DPO only ever sees the fixed preference dataset, which is why on-policy methods still tend to win at the frontier.

---

## 11. Variational lower bound (ELBO)

*In plain language:* the ELBO is a workaround for an integral you can't compute. The quantity you actually want, the probability of the data with the hidden variable summed out, is intractable, so you build a lower bound on it that you *can* compute and maximize that instead. Pushing the bound up pushes the real thing up with it.

**Setup**: latent-variable model $p_\theta(x, z)$. Want to maximize $\log p_\theta(x)$.

**Trick**: introduce variational distribution $q(z|x)$ and use Jensen's:

$$
\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \mathbb{E}_{q(z|x)}\left[\frac{p_\theta(x, z)}{q(z|x)}\right]
$$

**Jensen's inequality** for concave $\log$: $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$. Apply it:

$$
\log p_\theta(x) = \log \mathbb{E}_q\!\left[\tfrac{p_\theta(x,z)}{q(z|x)}\right] \;\geq\; \mathbb{E}_q\!\left[\log \tfrac{p_\theta(x, z)}{q(z|x)}\right] = \mathbb{E}_q[\log p_\theta(x, z)] + H(q)
$$

This is the **ELBO**.

**Equivalent form** (split $\log p_\theta(x, z) = \log p_\theta(x|z) + \log p(z)$):

$$
\mathrm{ELBO} = \mathbb{E}_q[\log p_\theta(x|z)] + \mathbb{E}_q[\log p(z)] - \mathbb{E}_q[\log q(z|x)] = \mathbb{E}_q[\log p_\theta(x|z)] - \mathrm{KL}(q(z|x) \,\|\, p(z))
$$

The gap to true log-likelihood: $\log p_\theta(x) - \mathrm{ELBO} = \mathrm{KL}(q(z|x) \| p_\theta(z|x))$ — exactly the KL between approximate and true posterior. ELBO is tight when $q$ matches the true posterior.

Reconstruction term + KL-to-prior term. The VAE objective.

See `21_dimensionality_reduction/` (autoencoders), `33_information_theory/`.

> **Saying it out loud (narrate while you write).** "I want the log-likelihood, but it has an integral over the latent inside a log, which is hopeless. So I multiply and divide by a distribution `q` of my choosing — that turns the integral into an expectation under `q`. Now `log` is concave, so Jensen's inequality lets me swap the log and the expectation and only lose something: `log` of an expectation is at least the expectation of the log." Write the bound. "That's the ELBO, and rearranged it's a reconstruction term minus the KL from `q` to the prior — the VAE objective, exactly." Then the identity that scores: "the gap between the true log-likelihood and the ELBO is precisely the KL between my approximate posterior and the true one. So maximizing the bound does two jobs at once — it fits the model and it drags `q` toward the true posterior — and the bound is tight exactly when they coincide." Named failure mode: posterior collapse, where `q` just becomes the prior, the KL term goes to zero, and the latent carries no information at all.

---

## 12. Bias-variance decomposition

**Setup**: estimate $f^*(x)$ from random training set $D$. Evaluate at fixed $x$.

**Steps**:
1. Let $\bar{f}(x) = \mathbb{E}_D[\hat{f}_D(x)]$.
2. Add and subtract: $(y - \hat{f}_D)^2 = (y - \bar{f} + \bar{f} - \hat{f}_D)^2 = (y - \bar f)^2 + 2(y - \bar f)(\bar f - \hat f_D) + (\bar f - \hat f_D)^2$.
3. **Cross-term vanishes**: take $\mathbb{E}_D$. $y$ and $\bar f$ are constants w.r.t. $D$, so $\mathbb{E}_D[2(y - \bar f)(\bar f - \hat f_D)] = 2(y - \bar f) \mathbb{E}_D[\bar f - \hat f_D] = 2(y - \bar f) \cdot 0 = 0$ (by definition of $\bar f$).
4. $\mathbb{E}_D[(y - \hat{f}_D)^2] = (y - \bar f)^2 + \mathbb{E}_D[(\bar f - \hat f_D)^2]$.
5. Now take $\mathbb{E}$ over the noise in $y = f^*(x) + \epsilon$: first term becomes $(\bar f - f^*)^2 + \sigma^2 = \mathrm{Bias}^2 + \sigma^2$. Second term is $\mathrm{Var}$.

See `27_advanced_theory/`, `52_statistical_learning_theory/`.

> **Saying it out loud (narrate while you write).** "The whole derivation is one add-and-subtract. Imagine retraining on many different datasets and let `f-bar` be the average prediction at this point. Write the squared error and insert `minus f-bar plus f-bar` in the middle" — write it — "expand the square into three terms. Now take the expectation over datasets: the cross term has `f-bar minus f-hat` in it, and by definition the average of `f-hat` *is* `f-bar`, so that expectation is zero and the cross term vanishes." That's the trick, and it's the same trick as in the variance identity. "What's left is a squared gap between the average model and the truth — that's bias, the error you'd still have with infinite datasets — plus the spread of the models around their own average, which is variance. Then let `y` carry noise, and that contributes an irreducible sigma-squared." Land it with the intuition and the caveat: a straight line through a curve has high bias and low variance, a wiggly polynomial the reverse, and the classical U-curve says balance them — while modern over-parameterized networks descend a second time past the interpolation point, so treat the U as intuition, not law.

---

## 13. Information gain (decision tree split)

**Setup**: dataset $S$ with class labels.

**Entropy**: $H(S) = -\sum_c p_c \log p_c$.

**After split on feature $A$ into $\{S_v\}$**:

$$
H(S | A) = \sum_v \frac{|S_v|}{|S|} H(S_v)
$$

**Information gain**: $\mathrm{IG} = H(S) - H(S | A)$.

**Key identity**: $\mathrm{IG} = I(S; A)$ — IG is exactly the mutual information between class label and feature $A$. That makes it intuitive: pick the feature that's most informative about the label.

**Why $\mathrm{IG} \geq 0$**: conditioning never increases entropy (Jensen on concave $H$, applied to $H(S|A) \leq H(S)$). Equality iff $S \perp A$.

Tree picks the split that maximizes IG (or Gini decrease in CART).

**Gini**: $G(S) = 1 - \sum_c p_c^2$. Computationally cheaper (no log); similar selection.

See `26_tree_based_methods/`.

> **Saying it out loud (narrate while you write).** "A decision tree wants the split that most reduces uncertainty about the label. Entropy measures that uncertainty, so I write `H of S`, then the entropy after splitting on a feature — which is just the weighted average of the children's entropies, weighted by how many points fall into each. Information gain is the difference." Then the identity that makes it click: "that difference is exactly the mutual information between the label and the feature, so I'm literally picking the feature that tells me the most about the class." And why it's never negative: conditioning can't increase entropy on average, so gain is at least zero, with equality only when the feature is independent of the label. Close with the practical tradeoff: Gini does essentially the same job without computing logarithms, which is why CART uses it, and raw information gain is biased toward high-cardinality features — a unique ID column gets perfect gain and zero generalization, which is what gain *ratio* corrects.

---

## 14. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| What's $\sqrt{d}$ in attention? | Tradition | Variance scaling — keeps QK product unit-variance |
| Cross-entropy + softmax gradient? | Complicated | $p - y$. Beautifully simple. |
| Why does EM converge? | Gradient descent | Each E-step gives lower bound; M-step maximizes; likelihood monotone |
| What does ELBO bound? | Posterior | Log-marginal-likelihood from below |
| KL forward vs reverse? | Same | Forward mode-covering (MLE); reverse mode-seeking (VI) |
| SVM dual support vectors? | Random points | Points where $\alpha_i > 0$; on/violating margin |
| RoPE relative property? | Magic | $\langle R_m q, R_n k \rangle$ depends only on $n - m$ |

> **Saying it out loud.** The pattern across this table is that every one of these has a one-line reason, and interviewers are checking whether you know the reason or just the result. Root `d` is variance control, not tradition. The `p` minus `y` gradient is a cancellation you can derive in two lines. EM converges because each step tightens then raises a lower bound, not because it's doing gradient descent. ELBO bounds the log-marginal-likelihood from below, not the posterior. And forward versus reverse KL is the difference between blurring across all the modes and collapsing onto one. If you can say the *why* in a sentence for each row, you're ready for the follow-up rather than just the question.

---

## 15. Eight derivations to drill cold

1. **2-layer MLP backprop** with cross-entropy + softmax.
2. **Scaled dot-product attention** with multi-head + masking.
3. **OLS gradient + closed form** with PSD Hessian.
4. **Logistic regression gradient** showing convexity.
5. **EM for GMM**: E-step posterior, M-step updates.
6. **DPO loss** from RLHF + Bradley-Terry.
7. **ELBO derivation** via Jensen's inequality.
8. **Bias-variance decomposition**.

For each: 5 minutes on a whiteboard. Until automatic.

> **Saying it out loud.** When I'm at a whiteboard I say the plan before I write anything: what I'm deriving, what I'm allowed to assume, and roughly how many steps it'll take. Then I narrate every line while my hand moves, because silence reads as being stuck and an interviewer can't give you a nudge if they don't know where you are. Each of these eight has one load-bearing step — the softmax-cross-entropy cancellation, substituting `w`-star back into the Lagrangian, the vanishing cross term in bias-variance, the partition function cancelling in DPO — so I make sure I can name that step before I start, and if I blank on the algebra I say what the step is supposed to accomplish and keep going. Budget five minutes each. The failure mode this avoids is the real one: not getting it wrong, but freezing.

---

## 16. Drill plan

- 1 derivation per day for 8 days. Then cycle.
- Time yourself: 5 min per derivation cold; 3 min after a week of practice.
- Practice teaching each: explain to an imaginary interviewer.
- Pair the derivation with the relevant deep dive's "8 most-asked interview questions" to make sure you can recite both proof and intuition.

---

## 17. Further reading

This deep dive is a meta-collection. The full derivations live in:

- `31_neural_networks` for backprop.
- `04_transformers` and `05_attention_mechanisms` for attention.
- `01_classical_ml` for OLS and logistic.
- `19_advanced_clustering` for EM.
- `08_training_techniques` for DPO.
- `21_dimensionality_reduction` for ELBO/VAE.
- `27_advanced_theory` for bias-variance.

Drill the derivations in those locations and you'll be ready for the whiteboard rounds.
