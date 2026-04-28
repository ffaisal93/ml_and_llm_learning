# Topic 10: Optimizers — Algorithmic Comparison

> **Read first:** `02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md` for the LR-side of the story (when LR works/fails, schedules, scaling rules). This file focuses on the **algorithms themselves** — what each optimizer is doing mathematically, what problem it fixes, and when each one is the right choice.

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | Per-optimizer comparison — this file. |
| `INTERVIEW_GRILL.md` | 40 interview questions on optimizer algorithms specifically (different angle from `02_gradient_descent/INTERVIEW_GRILL.md`). |
| `optimizers.py` | From-scratch implementations of SGD, Momentum, Nesterov, RMSProp, Adam, AdamW, Lion. |

---

## The mental model: every optimizer is some approximation of Newton

Newton's method is $\theta_{t+1} = \theta_t - H_t^{-1} g_t$. Use the inverse Hessian as a preconditioner; you get quadratic convergence near a minimum. The problem: $H_t^{-1}$ is $O(d^2)$ to store and $O(d^3)$ to invert for $d$ parameters. For a 70B-parameter model, that's $\approx 5 \times 10^{21}$ Hessian entries — impossible.

Every modern optimizer is some computationally tractable approximation to Newton:

| Optimizer | Preconditioner | Storage | Approximation |
|---|---|---|---|
| SGD | $\eta \cdot I$ | 0 extra | "All directions are the same" |
| Momentum | $\eta \cdot I$ (with velocity) | $O(d)$ | Accumulate past gradients |
| RMSProp | $\eta \cdot \mathrm{diag}(1/\sqrt{\hat v})$ | $O(d)$ | $\mathrm{diag}(H) \approx \sqrt{g \cdot g}$ |
| Adam | $\eta \cdot \mathrm{diag}(1/\sqrt{\hat v})$ (+ momentum) | $O(d)$ | Momentum + RMSProp |
| AdamW | Same as Adam, decoupled WD | $O(d)$ | Adam with cleaner regularization |
| Lion | $\eta \cdot \mathrm{sign}(m)$ | $O(d)$ | Sign-based; no second moment |
| Sophia | $\eta \cdot \mathrm{diag}(1/\hat h)$ | $O(d)$ | Stochastic Hutchinson Hessian estimate |
| Shampoo | Block-diagonal of Hessian | $O(d^{1.5})$ | Per-layer Kronecker factors |
| K-FAC | Fisher block approx | $O(d^{1.5})$ | Per-layer Kronecker Fisher |

This single table is the conceptual spine of the topic. If an interviewer asks why Adam works, the answer starts here: it's a cheap diagonal approximation to second-order optimization. If an interviewer asks how to do better than Adam, the answer is on the lower rows.

---

## SGD: the baseline everything else compares to

$$
\theta_{t+1} = \theta_t - \eta \cdot g_t
$$

**Pros:** trivial to implement, well-understood theory, often best generalization on well-conditioned vision tasks. The implicit-regularization noise from mini-batching biases SGD toward flat minima.

**Cons:** sensitive to LR; struggles on ill-conditioned problems; needs hand-tuned schedule. For transformers and embedding-heavy models, SGD without modification is essentially unusable.

**When to use:** image classification with well-tuned schedule (ResNet-style); when you have compute to do thorough LR sweeps; when generalization > convergence speed.

---

## SGD with momentum

Two flavors — practically similar, conceptually distinct.

### Polyak (classical) momentum

$$
v_{t+1} = \beta\, v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

Accumulates a velocity that persists across steps. $\beta = 0.9$ is standard. Equivalent to a low-pass filter on gradients with effective horizon $1/(1-\beta) \approx 10$ steps.

### Nesterov momentum

$$
v_{t+1} = \beta\, v_t + \nabla L\!\big(\theta_t - \eta\, \beta\, v_t\big), \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

Computes the gradient at the *lookahead* position (where momentum would take you anyway) before the update. Often slightly better empirically; the convergence rate on convex problems improves from $O(1/T)$ to $O(1/T^2)$ for smooth strongly-convex cases.

**Why momentum helps:** in ill-conditioned valleys, plain SGD oscillates back and forth across the narrow direction while making slow progress along the long direction. Momentum averages out the oscillations and reinforces consistent directions, effectively re-conditioning the problem.

---

## RMSProp

$$
v_t = \beta\, v_{t-1} + (1-\beta)\, g_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta\, \frac{g_t}{\sqrt{v_t} + \varepsilon}
$$

Per-parameter rescaling by the running RMS of gradients. $\beta = 0.99$ typical. **Adaptive** — parameters with consistently large gradients get smaller updates; rare parameters get larger updates. Originally proposed by Hinton in lecture slides; never published as a paper.

**Why it works:** approximates the diagonal of the Hessian via $\mathbb{E}[g^2] \approx \mathrm{diag}(H)$ (true under specific assumptions — Fisher information for likelihood-based losses). Removes most of the LR-tuning sensitivity that plain SGD has.

**Modern usage:** rarely used standalone; lives on as a component of Adam.

---

## Adam: RMSProp + Momentum + Bias correction

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t \quad &\text{(1st moment, momentum)} \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2 \quad &\text{(2nd moment, RMS)} \\
\hat m_t &= m_t / (1 - \beta_1^t) \quad &\text{(bias correction)} \\
\hat v_t &= v_t / (1 - \beta_2^t) \quad &\text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
\end{aligned}
$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.999, \varepsilon = 10^{-8}, \eta = 10^{-3}$.

The defining features:

1. **Per-parameter LR** via $1/\sqrt{\hat v}$ (from RMSProp).
2. **Momentum** via $\hat m$ (smooths gradient direction).
3. **Bias correction** so that the first ~1000 steps aren't biased toward small moments.

**Why bias correction matters.** $m_t$ and $v_t$ are initialized at zero. With $\beta_2 = 0.999$, $v_t$ is biased low for the first ~1000 steps because there's not enough mass yet. Without correction, the early effective LR is wrong. The $1/(1-\beta^t)$ factor exactly inverts the geometric-series bias.

**Why people sometimes hate Adam.** It often achieves equal or lower training loss than SGD but worse generalization on some tasks. Three explanations: (1) aggressive per-parameter rescaling biases toward sharper minima, (2) less SGD-style implicit regularization, (3) different effective trajectory through parameter space. Mitigations: AdamW, longer training, switch to SGD for last epochs.

---

## AdamW: the modern default

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta\, \lambda\, \theta_t \qquad \text{(decoupled weight decay)}
$$

$m_t, v_t, \hat m_t, \hat v_t$ are computed the same as Adam. The change from Adam is purely the decay term. Why it matters:

In **plain Adam with L2 regularization**, you'd add $\lambda \cdot \theta$ to the gradient before the optimizer sees it: $g_t \leftarrow g_t + \lambda \cdot \theta_t$. Then $v_t$ accumulates $(g_t + \lambda \theta_t)^2$, the regularization term gets divided by $\sqrt{\hat v}$, and parameters with high gradient variance see a weakened L2 effect. Net result: regularization strength is non-uniform across parameters in a way nobody intends.

In **AdamW**, weight decay is applied directly to $\theta$ after the Adam update. Every parameter shrinks by exactly $\eta \cdot \lambda$ per step regardless of gradient statistics. This is what people mean when they say "weight decay" semantically — and it's what Adam-with-L2 fails to deliver.

**Why this is the LLM default:** language models have heterogeneous gradient statistics across embeddings, attention, and FFN parameters. Plain Adam+L2 leaves weight decay vestigial on the parameters that need it most. AdamW fixes it.

**Default $\lambda$ for LLM pretraining: $0.1$. For other domains: $0.01$–$0.05$ is more common.**

---

## Lion (Chen et al. 2023, *Symbolic Discovery of Optimization Algorithms*)

$$
\begin{aligned}
c_t &= \beta_1\, m_{t-1} + (1-\beta_1)\, g_t \quad &\text{(interpolation, no bias correction)} \\
\theta_{t+1} &= \theta_t - \eta\, \mathrm{sign}(c_t) - \eta\, \lambda\, \theta_t \quad &\text{(sign-based update)} \\
m_t &= \beta_2\, m_{t-1} + (1-\beta_2)\, g_t \quad &\text{(momentum stored separately)}
\end{aligned}
$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.99, \eta = 3 \times 10^{-5}$ (about 10x lower than Adam).

The trick: replace $\hat m / \sqrt{\hat v}$ with $\mathrm{sign}(\hat m)$. Updates are always $\pm \eta$ per parameter (modulo weight decay).

**Why it works:** the sign function is a kind of extreme normalization — every parameter gets the same step size, regardless of gradient magnitude. This is similar in spirit to Adam's per-parameter rescaling but more aggressive.

**Trade-offs:**

- **Pro:** Memory: only one extra state buffer ($m_t$) instead of Adam's two. For 70B models, that's tens of GB saved.
- **Pro:** Faster optimizer step (no division, no square root).
- **Con:** Higher sensitivity to LR and $\lambda$. Lion's optimal LR is typically 3–10x smaller than Adam's; weight decay should be 3x larger.
- **Con:** Less robust on small-data fine-tuning.

**Frontier-lab interest:** Several papers now report Lion matching or beating AdamW on language modeling at the 1B–10B scale. Whether it dominates at 70B+ is being actively investigated.

---

## Second-order and quasi-second-order methods

**Sophia (Liu et al. 2023):**

$$
\hat h_t = \mathrm{clip}(\text{stoch\_hessian\_estimate}, \rho)
$$

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat m_t}{\max(\gamma \cdot \hat h_t, \varepsilon)} - \eta\, \lambda\, \theta_t
$$

Uses Hutchinson's stochastic estimator: $\mathrm{diag}(H) \approx \mathbb{E}[v \odot Hv]$ for random $v$. Compute $Hv$ via Hessian-vector product (one extra backward pass). Reportedly converges in fewer steps than AdamW on LLM pretraining. The catch: extra compute per step. Whether the per-step efficiency offsets the cost is the empirical question.

**Shampoo (Anil et al. 2020):**

Tracks two-block-Kronecker preconditioners per layer: $L_t$ (left factor, size $m \times m$) and $R_t$ (right factor, size $n \times n$) for an $m \times n$ weight matrix. Update:

$$
W_{t+1} = W_t - \eta \cdot L_t^{-1/4}\, G_t\, R_t^{-1/4}
$$

Memory $O(m^2 + n^2)$ per layer instead of $O(d^2)$. Tractable. State-of-the-art empirical results on some tasks; rare in LLM pretraining mainstream because of system complexity.

**K-FAC (Martens & Grosse 2015):**

Approximates the Fisher information matrix as Kronecker-factored block-diagonal. Like Shampoo but for the Fisher (which equals the Hessian under specific conditions for likelihood losses). Used in some RL and Bayesian-deep-learning contexts; not common in modern LLM training.

**LBFGS:**

Deterministic, full-batch, classical quasi-Newton. Stores last $m$ gradient and parameter pairs to approximate $H^{-1}$. **Useless for stochastic deep learning** because it requires consistent full gradients. Sometimes used for fine-tuning small models or for debugging. Mention it once and move on.

---

## LARS / LAMB: layer-wise scaling

For very large batch training (>16K), even Adam's per-parameter scaling isn't enough. Updates can be heterogeneous across layers in ways that destabilize training.

**LARS (You et al. 2017):**

$$
\text{trust}_{\text{layer}} = \frac{\|\theta_{\text{layer}}\|}{\|\text{update}_{\text{layer}}\| + \varepsilon}, \qquad \eta_{\text{layer}}^{\text{eff}} = \eta_{\text{global}} \cdot \text{trust}_{\text{layer}}
$$

Forces the per-layer update-to-weight ratio to be constant.

**LAMB (You et al. 2019):**

LARS applied on top of Adam. Used for the famous "BERT in 76 minutes" run with batch size 65,536.

**Modern status:** LARS/LAMB are mostly superseded by **muP** (maximal update parameterization, Yang & Hu 2022), which achieves a similar effect by changing initialization and per-layer LR scaling at construction time, rather than enforcing trust ratios at runtime. muP also enables hyperparameter transfer across model widths.

---

## Comparison table (interview-ready)

| Optimizer | Memory overhead | Best for | Avoid for |
|---|---|---|---|
| SGD | 0 | Vision (well-conditioned) | Transformers without warmup |
| SGD+Momentum | $1\times$ params | Vision, well-tuned recipes | Ill-conditioned problems |
| Adam | $2\times$ params | General NLP, transformers | Cases where SGD generalization wins |
| AdamW | $2\times$ params | LLM pretraining, modern transformers | (Default; rarely the wrong answer) |
| Lion | $1\times$ params | Memory-constrained large-model training | Cases needing fine-grained LR adaptation |
| Sophia | $2\times$ params + HVP | Compute-efficient pretraining | Cases without Hessian-vector support |
| Shampoo | per-layer matrices | High-impact small-model runs | Memory-tight large-scale runs |
| LARS/LAMB | $2\times$ params | Very large batch (>16K) | Standard scale (use Adam/AdamW instead) |

---

## Common interview traps

1. **"Adam doesn't need LR tuning."** Wrong. Adam tolerates a wider LR range than SGD (maybe one decade vs. half a decade), but the optimal $\eta$ still varies by ~3x across reasonable choices, and the wrong $\eta$ still diverges.

2. **"L2 and weight decay are the same thing."** True for SGD. **False for Adam**. This is a high-frequency interview question.

3. **"Bigger $\beta_2$ is more stable."** No. Larger $\beta_2$ makes $\hat v$ slower to react to gradient outliers — that's stability against spikes. But it also makes $\hat v$ slower to track the actual gradient magnitude, which can destabilize when gradient statistics shift mid-training. Default $0.999$ is a reasonable compromise.

4. **"Adam's bias correction is just a numerical detail."** No. Without it, the first ~1000 steps have effective LR much too high (because $\hat v$ is biased low). Bias correction is the difference between exploding and not exploding for many runs.

5. **"Lion is strictly better because it uses less memory."** Empirically promising at moderate scale, but its LR sensitivity is higher and several frontier-lab teams still default to AdamW. Memory savings matter most at >50B parameters; below that, AdamW is hard to beat for stability.

6. **"You can compare optimizers by holding LR fixed."** No. Each optimizer has its own optimal $\eta$. Fair comparison requires sweeping LR and ideally schedule for each.

---

## What the interviewer may ask next

(Each fully answered in `INTERVIEW_GRILL.md`.)

1. Walk me through Adam, including bias correction.
2. Why is AdamW different from Adam-with-L2?
3. Why does Lion sometimes match AdamW with lower memory?
4. What's the connection between Adam and second-order methods?
5. Why does Adam sometimes generalize worse than SGD?
6. When would you choose Shampoo over Adam?
7. Why doesn't $\beta_2$ get set very high (like 0.9999)?

---

## Recommended ordering for study

1. Master SGD + momentum derivations (whiteboard-ready).
2. Master Adam with bias correction (whiteboard-ready).
3. Master the AdamW vs. Adam+L2 distinction (verbal, with example).
4. Skim Lion, Sophia, Shampoo at concept level (one paragraph each).
5. Drill `INTERVIEW_GRILL.md`.

---

## Cross-references

- `02_gradient_descent/LEARNING_RATE_DEEP_DIVE.md` — the LR-centric companion to this file.
- `02_gradient_descent/INTERVIEW_GRILL.md` — LR-focused grill (60 questions).
- `48_optimization_and_matrix_calculus/` — Hessians, conditioning, second-order theory.
