# The formula sheet

Ten minutes, one page. Nothing here is explained — it is recalled.

---

## Losses

**MSE** — Gaussian noise / MLE for regression:
$$\mathcal{L} = \frac{1}{n}\sum_i (\hat y_i - y_i)^2$$

**MAE** — median-seeking, robust to outliers:
$$\mathcal{L} = \frac{1}{n}\sum_i |\hat y_i - y_i|$$

**Huber** — quadratic near zero, linear in the tail; $\delta$ is the switch:
$$\mathcal{L}_\delta(r) = \begin{cases} \tfrac12 r^2 & |r| \le \delta \\ \delta(|r| - \tfrac12\delta) & |r| > \delta \end{cases}, \quad r = \hat y - y$$

**Binary cross-entropy** — $y \in \{0,1\}$, $p = \sigma(z)$:
$$\mathcal{L} = -\frac{1}{n}\sum_i \big[ y_i \log p_i + (1-y_i)\log(1-p_i) \big]$$

**Categorical cross-entropy** — one-hot $y$, so only the true class survives:
$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log p_c = -\log p_{\text{true}}$$

**KL divergence** — extra bits from coding $p$ with $q$; asymmetric, $\ge 0$:
$$D_{KL}(p \| q) = \sum_x p(x)\log\frac{p(x)}{q(x)}$$

**Cross-entropy = entropy + KL** — the identity everything rests on:
$$H(p,q) = H(p) + D_{KL}(p\|q)$$
Minimizing CE over $q$ = minimizing KL, since $H(p)$ is fixed.

**Contrastive (pairwise)** — pull positives, push negatives past margin $m$:
$$\mathcal{L} = y\, d^2 + (1-y)\max(0, m - d)^2$$

**Triplet** — anchor $a$, positive $p$, negative $n$:
$$\mathcal{L} = \max(0,\; d(a,p) - d(a,n) + m)$$

**InfoNCE / NT-Xent** — softmax over one positive against $K$ negatives, temperature $\tau$:
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q,k^+)/\tau)}{\sum_{i=0}^{K}\exp(\text{sim}(q,k_i)/\tau)}$$
Lower bound on mutual information; small $\tau$ = harder on near-negatives.

---

## Gradients

**Linear regression**, $\hat y = Xw$, MSE (dropping the 2/n):
$$\nabla_w \mathcal{L} = X^\top(\hat y - y)$$

**Logistic regression**, $\hat y = \sigma(Xw)$, BCE — *the same expression*:
$$\nabla_w \mathcal{L} = X^\top(\hat y - y)$$

**Softmax + categorical CE**, same again per-example:
$$\frac{\partial \mathcal{L}}{\partial z} = p - y$$

**Why they coincide.** All three are GLMs with the canonical link. Write the likelihood in exponential-family form; the log-partition derivative $A'(\theta)$ is exactly the mean $\hat y$, so $\partial \mathcal{L}/\partial \theta = \hat y - y$, and the chain rule to $w$ contributes $X^\top$. The link's derivative cancels the loss's curvature. Say this in one line: *canonical link ⇒ the sigmoid/softmax Jacobian cancels against the cross-entropy denominator, leaving the residual.*

**Corollary worth knowing.** BCE on a sigmoid gives $(p-y)x$; MSE on a sigmoid gives $(p-y)\sigma'(z)x$ — the extra $\sigma'(z)\to 0$ term is why MSE + sigmoid trains badly (saturated units get no gradient).

**Normal equation** (closed form, when you can afford $O(d^3)$):
$$w = (X^\top X)^{-1}X^\top y, \qquad \text{ridge: } w = (X^\top X + \lambda I)^{-1}X^\top y$$

---

## Activations

**Sigmoid** — $\sigma(x) = \dfrac{1}{1+e^{-x}}$, range $(0,1)$
$$\sigma'(x) = \sigma(x)(1-\sigma(x)), \quad \max = 0.25 \text{ at } x=0$$

**Tanh** — zero-centered, range $(-1,1)$
$$\tanh'(x) = 1 - \tanh^2(x), \quad \max = 1$$

**ReLU** — $\max(0,x)$
$$\text{ReLU}'(x) = \mathbb{1}[x>0] \quad (\text{undefined at } 0; \text{ pick } 0)$$

**Leaky ReLU / PReLU** — $\max(\alpha x, x)$, gradient $\alpha$ on the left; fixes dead units.

**GELU** — $x \cdot \Phi(x)$, $\Phi$ = standard normal CDF; tanh approximation:
$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{2/\pi}\,(x + 0.044715x^3)\right]\right)$$

**SiLU / Swish** — $x\,\sigma(\beta x)$, $\beta=1$ default:
$$\text{SiLU}'(x) = \sigma(x)\big(1 + x(1-\sigma(x))\big)$$

**SwiGLU** — the FFN in essentially every current LLM; splits the projection in two:
$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_1)\odot xW_3\big)W_2$$
Three matrices, so hidden width is scaled by $\approx 2/3$ to keep the parameter count.

**Softmax** — $p_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$; subtract $\max_j z_j$ for stability. Jacobian:
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$
Shift-invariant: $\text{softmax}(z + c) = \text{softmax}(z)$ — hence $C$ logits have $C-1$ degrees of freedom.

**Temperature** — $p_i \propto \exp(z_i/T)$. $T\to 0$ = argmax, $T\to\infty$ = uniform.

---

## Normalization

Common core: $\hat x = \dfrac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$, then $y = \gamma\hat x + \beta$. Only the axis of $\mu, \sigma$ changes.

**BatchNorm** — statistics over the **batch** dimension, per feature/channel:
$$\mu_c = \frac{1}{B}\sum_{b} x_{b,c}, \quad \sigma_c^2 = \frac{1}{B}\sum_b (x_{b,c}-\mu_c)^2$$
Train uses batch stats; inference uses running averages. Batch-size dependent, awkward for RNNs/variable-length sequences.

**LayerNorm** — statistics over the **feature** dimension, per example:
$$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i, \quad \sigma^2 = \frac{1}{d}\sum_i (x_i-\mu)^2$$
Identical at train and test; no batch coupling. Default in transformers.

**RMSNorm** — drops the mean subtraction and $\beta$:
$$y = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}\odot\gamma$$
Cheaper, empirically equal; used by Llama, Qwen, Gemma, DeepSeek and effectively all current open-weight LLMs.

**One-line difference.** BatchNorm normalizes *across examples for one feature*; LayerNorm normalizes *across features for one example*; RMSNorm is LayerNorm without centering.

**Pre-norm vs post-norm** — pre-norm ($x + \text{Attn}(\text{Norm}(x))$) keeps a clean residual path and trains deep stacks without warmup; post-norm is the original and needs warmup. Pre-norm is standard (OLMo is a notable post-norm holdout).

**QK-Norm** — normalize $Q$ and $K$ before the dot product; now common for training stability at scale.

---

## Attention

**Scaled dot-product:**
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$
$M$ = mask ($-\infty$ on disallowed positions).

**Why $\sqrt{d_k}$.** If $q,k$ have i.i.d. zero-mean unit-variance entries, $q\cdot k$ has variance $d_k$. Unscaled logits grow like $\sqrt{d_k}$, the softmax saturates, and its Jacobian $p(\delta - p)$ vanishes. Dividing by $\sqrt{d_k}$ restores unit variance.

**Multi-head** — $h$ heads, each of width $d_k = d/h$:
$$\text{MHA}(X) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O, \quad \text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)$$

**Complexity** — sequence $n$, model width $d$:
- Scores $QK^\top$: $O(n^2 d)$ time, $O(n^2)$ memory (naive) or $O(n)$ with FlashAttention tiling
- Projections: $O(n d^2)$
- Crossover: attention dominates once $n \gtrsim d$

**MQA / GQA** — cut the number of KV heads, keep all query heads. MHA: $h$ KV heads. GQA: $g$ groups, $1 < g < h$. MQA: $g = 1$. Shrinks the KV cache by $h/g$ with near-zero quality loss. GQA is the current default (Llama-2 70B: 64 query heads, 8 KV heads = 8× cache reduction). **MLA** (DeepSeek) instead compresses KV into a low-rank latent.

**RoPE** — rotate $q,k$ by a position-dependent angle so the dot product depends on $m-n$:
$$\langle R_m q, R_n k\rangle = f(q,k,m-n)$$
Relative position for free; extend context by scaling the base frequency (NTK / YaRN).

---

## Regularization

**L2 / ridge** — $\mathcal{L} + \lambda\|w\|_2^2$; gradient adds $2\lambda w$; shrinks smoothly, never exactly zero; = Gaussian prior (MAP).

**L1 / lasso** — $\mathcal{L} + \lambda\|w\|_1$; subgradient $\lambda\,\text{sign}(w)$; corners of the $\ell_1$ ball ⇒ exact zeros ⇒ feature selection; = Laplace prior.

**Elastic net** — $\lambda_1\|w\|_1 + \lambda_2\|w\|_2^2$; sparsity plus grouping of correlated features (L1 alone picks one arbitrarily).

**Dropout** — keep probability $p$. Train: $\tilde h = h \odot m / p$ with $m \sim \text{Bernoulli}(p)$ (*inverted dropout* — scale at train). Inference: identity, no scaling. Original formulation instead multiplied by $p$ at test; every framework now does the inverted version.

**Label smoothing** — target $y_c \to (1-\epsilon)y_c + \epsilon/C$. Caps logit margins, improves calibration, hurts if you later distill from the model.

**Weight decay vs L2.** Identical for plain SGD (up to a factor of the LR). Not identical for adaptive optimizers: L2 goes into the gradient and gets divided by $\sqrt{v}$, so large-gradient weights get decayed *less*. AdamW decouples it — decay applied directly to $w$:
$$w \leftarrow w - \eta\left(\frac{\hat m}{\sqrt{\hat v}+\epsilon} + \lambda w\right)$$

**Early stopping** — approximately L2 for linear models; the effective $\lambda \sim 1/(\eta t)$.

---

## Optimizers

**SGD** — $w \leftarrow w - \eta g$

**Momentum** — $v \leftarrow \beta v + g$, then $w \leftarrow w - \eta v$; effective step $\approx \eta/(1-\beta)$.

**Nesterov** — evaluate the gradient at the look-ahead point $w - \eta\beta v$:
$$v \leftarrow \beta v + \nabla f(w - \eta\beta v), \quad w \leftarrow w - \eta v$$

**AdaGrad** — $G \leftarrow G + g^2$, $w \leftarrow w - \eta g/(\sqrt{G}+\epsilon)$; LR decays monotonically to zero.

**RMSProp** — fix AdaGrad's decay with an EMA:
$$v \leftarrow \beta v + (1-\beta)g^2, \quad w \leftarrow w - \frac{\eta g}{\sqrt{v}+\epsilon}$$

**Adam** — momentum on both moments, with bias correction:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$
$$w \leftarrow w - \eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
Defaults $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$. **Bias correction** exists because $m_0=v_0=0$ biases early estimates toward zero; without it the first steps are far too small (and $\beta_2 = 0.999$ means the bias lasts ~1000 steps).

**AdamW** — Adam with decoupled weight decay (above). Standard for transformers.

**LR schedules** — linear warmup then cosine decay to ~10% of peak is the transformer default; warmup exists because early Adam variance estimates are noisy.

**Gradient clipping** — $g \leftarrow g \cdot \min(1, c/\|g\|)$, typically $c = 1.0$.

---

## Metrics

$$\text{Precision} = \frac{TP}{TP+FP}, \qquad \text{Recall} = \frac{TP}{TP+FN}$$
$$F_1 = \frac{2PR}{P+R}, \qquad F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P + R}$$
$\beta > 1$ weights recall; $\beta = 2$ common when misses are costly.

$$\text{TPR} = \text{Recall}, \qquad \text{FPR} = \frac{FP}{FP+TN}$$

**ROC-AUC** — area under TPR-vs-FPR; equals $P(\text{score of random positive} > \text{score of random negative})$. Invariant to class balance — which is exactly why it **misleads at extreme imbalance**: FPR barely moves when TN is enormous, so a useless model still scores 0.9.

**PR-AUC / average precision** — $\sum_n (R_n - R_{n-1})P_n$. Baseline = positive class prevalence, not 0.5. Use this when positives are rare and you care about them.

**Accuracy** misleads at any imbalance (99% negatives ⇒ 99% by predicting nothing).

**Calibration** — Brier score $\frac{1}{n}\sum(p_i - y_i)^2$; ECE = weighted mean $|{\text{accuracy}} - {\text{confidence}}|$ over bins. A model can rank perfectly (AUC 1.0) and be badly calibrated.

**Regression** — $R^2 = 1 - \text{SS}_{res}/\text{SS}_{tot}$ (can go negative); RMSE punishes outliers, MAE does not; MAPE explodes near $y=0$.

**Ranking** — $\text{DCG@k} = \sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}$, NDCG = DCG / ideal DCG. MRR = mean of $1/\text{rank of first hit}$.

---

## Probability and statistics

**Bayes** — $P(\theta \mid D) = \dfrac{P(D\mid\theta)P(\theta)}{P(D)}$, i.e. posterior $\propto$ likelihood $\times$ prior.

**MLE** — $\hat\theta = \arg\max_\theta \log P(D\mid\theta)$. **MAP** — $\arg\max_\theta \big[\log P(D\mid\theta) + \log P(\theta)\big]$. MAP = MLE + regularizer; Gaussian prior ⇒ L2, Laplace prior ⇒ L1. As $n\to\infty$ the prior washes out and MAP → MLE.

**Bias–variance** (squared loss, expectation over training sets):
$$\mathbb{E}\big[(y - \hat f(x))^2\big] = \underbrace{\big(\mathbb{E}[\hat f(x)] - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\text{Var}(\hat f(x))}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}$$
$\sigma^2$ is label noise — no model reduces it. The clean decomposition holds for squared loss; for 0-1 loss it is only approximate.

**Entropy** — $H(X) = -\sum_x p(x)\log p(x)$; max at uniform, $\log C$.

**Cross-entropy** — $H(p,q) = -\sum p\log q = H(p) + D_{KL}(p\|q)$.

**Mutual information** — $I(X;Y) = H(X) - H(X\mid Y) = D_{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)$.

**Perplexity** — $\exp(\text{CE in nats})$; the effective branching factor.

**Jensen–Shannon** — symmetric, bounded: $\tfrac12 D_{KL}(p\|m) + \tfrac12 D_{KL}(q\|m)$, $m = \tfrac12(p+q)$.

**CLT** — $\bar X \approx \mathcal{N}(\mu, \sigma^2/n)$; standard error $\sigma/\sqrt{n}$.

**Covariance / correlation** — $\rho = \dfrac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y}$; zero correlation $\ne$ independence (except jointly Gaussian).

**Sigmoid ↔ logit** — $\text{logit}(p) = \log\frac{p}{1-p}$, inverse of $\sigma$. Logistic regression coefficients are log-odds ratios.

---

## Scaling and inference cost

**Chinchilla** — for fixed compute $C \approx 6ND$, loss is minimized when $N$ and $D$ scale together:
$$D^\ast \approx 20\,N \quad (\text{tokens} \approx 20 \times \text{parameters})$$
Epoch AI's 2024 replication showed the paper's fitted constants were off, but the ~20:1 policy survives. In practice deployed models are trained far past this (inference cost dominates lifetime cost, so you over-train a smaller model).

**Loss form** — $L(N,D) = L_\infty + A/N^\alpha + B/D^\beta$.

**Parameter count**, decoder-only transformer, $L$ layers, width $d$ (dense, ignoring embeddings):
$$N \approx 12 L d^2$$
($4d^2$ for QKVO, $8d^2$ for a $4d$ MLP; SwiGLU with $\tfrac{2}{3}\cdot 4d$ hidden lands in the same place.)
Embeddings add $V d$ (and another $Vd$ if untied) — non-negligible for small models.

**FLOPs**
- Forward: $\approx 2N$ per token (one multiply-add per parameter)
- Backward: $\approx 4N$ per token
- **Training: $\approx 6N$ FLOPs per token**, so $C \approx 6ND$
- Attention adds $\approx 12 L d n$ per token — ignorable until $n$ is comparable to $d$
- MoE: use *active* parameters for FLOPs, *total* parameters for memory

**Memory, training** — fp16 weights $2N$ + fp16 grads $2N$ + Adam states fp32 $8N$ (+ fp32 master copy $4N$) $\approx 16$–$20$ bytes per parameter, before activations.

**KV cache** (per sequence):
$$\text{bytes} = 2 \times L \times n_{kv} \times d_{head} \times n_{\text{tokens}} \times \text{bytes/elt}$$
The leading 2 is K and V. Multiply by batch size. fp16 = 2 bytes, fp8 = 1, int4 = 0.5. With GQA, $n_{kv} \ll n_{heads}$ — that ratio *is* the savings.

**Inference regimes** — prefill is compute-bound ($O(n^2)$ attention, parallel); decode is memory-bandwidth-bound (one token at a time, must re-read all weights + KV cache). Hence batching, paged KV (vLLM), and speculative decoding.

**Rules of thumb** — serving a model in fp16 needs $\approx 2N$ bytes of weights; int8 $\approx N$; int4 $\approx N/2$, plus KV cache, plus ~20% overhead.
