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

---

## 5. KL divergence

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

---

## 6. EM for GMM

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

---

## 8. SVM dual

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

---

## 9. RoPE rotation

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

---

## 10. DPO (direct preference optimization)

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

---

## 11. Variational lower bound (ELBO)

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
