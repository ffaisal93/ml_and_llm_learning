# Topic 2: Gradient Descent & Learning Rate

## Files in this folder

| File | Purpose |
|---|---|
| `README.md` | Conceptual overview (this file) — read this first. |
| `LEARNING_RATE_DEEP_DIVE.md` | The core interview deep-dive on learning rate: when it works, when it fails, schedules, scaling rules, edge of stability, AdamW. **Most important file in this folder.** |
| `INTERVIEW_GRILL.md` | 60 active-recall interview questions with strong answers. Drill before interviews. |

---

## What you'll learn

- The mathematics that decides whether gradient descent converges, oscillates, or diverges.
- Why mini-batch dominates batch and stochastic GD in practice — and how to defend that answer rigorously.
- How learning rate, batch size, gradient noise, and generalization are linked through a single quantity ($\eta/B$).
- The standard schedules (warmup, cosine, linear) and *why* each phase exists.
- How to read training curves and gradient norms to debug an unstable run.
- The frontier-lab vocabulary: edge of stability, critical batch size, gradient noise scale, muP.

If you can answer the **60 grill questions** in `INTERVIEW_GRILL.md` cleanly, you are above the bar for an applied scientist screen on this topic.

---

## Why this topic matters in interviews

Almost every modern training recipe — for vision, NLP, RL, diffusion, and LLMs — is some variant of mini-batch gradient descent with an adaptive optimizer and a learning-rate schedule. Interviewers use questions in this area to probe:

1. **Do you understand optimization or recite slogans?** "Adam works better" is a slogan. Knowing that Adam approximates the diagonal of the Hessian via the second moment of the gradient and that it can over-rescale dimensions whose $\hat v$ is dominated by noise — that's understanding.
2. **Can you debug?** Given a loss curve and a gradient norm, can you diagnose whether the LR is too high, too low, or whether the issue lives elsewhere?
3. **Do you know how it scales?** From a 1B-parameter run to a 70B-parameter run, what changes? If you don't know what muP is, you'll struggle.
4. **Do you know modern subtleties?** Edge of stability, AdamW vs. Adam+L2, critical batch size, linear scaling rule and its limits — these are the topics that separate top candidates.

The deep-dive file goes section by section through these topics with the right level of math and the right honesty about what's settled and what isn't.

---

## Core intuition: gradient descent in one paragraph

You have a loss $L(\theta)$ and you want to find $\theta$ that makes it small. The gradient $\nabla L(\theta)$ points in the direction of steepest *increase*; subtract a small multiple of it from $\theta$ and you decrease the loss. Repeat. The "small multiple" is the learning rate $\eta$. The reason this isn't trivial: the loss surface in real deep learning is non-convex, ill-conditioned (curvature varies massively across directions), and stochastic (we use mini-batch estimates of $\nabla L$). Every interesting question in this folder follows from one of these three properties.

---

## The three regimes

### Batch gradient descent

Computes $\nabla L$ over the entire dataset before each update. **Stable**, **expensive**, **rarely used at scale**. Only an option for small datasets or when exact gradients are essential (rare).

### Stochastic gradient descent (SGD, single sample)

Computes $\nabla L$ from one sample per step. **Cheap per step**, **very noisy**, **fast to start learning**. The noise has an underappreciated benefit — it's a form of implicit regularization that biases SGD toward flat minima. But variance is too high for most practical use.

### Mini-batch gradient descent

The default. Batch size $B$ (typically 32–8192) trades stability for speed. Variance of the gradient estimate is $\sigma^2 / B$. The right $B$ depends on hardware (memory, parallelism) and on the gradient noise scale (after which doubling $B$ stops paying off). See `LEARNING_RATE_DEEP_DIVE.md` §6 for critical batch size.

---

## Why the learning rate is the master hyperparameter

For a quadratic loss with Hessian $H$, GD converges only if $0 < \eta < 2/\lambda_{\max}(H)$. Above that, you diverge in the sharpest direction. Below $1/\lambda_{\max}(H)$, you converge but waste steps in flatter directions. The optimal rate is $2/(\lambda_{\max} + \lambda_{\min})$, and convergence speed depends on the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$.

In real deep networks:

- $\lambda_{\max}(H)$ varies by orders of magnitude across layers.
- $H$ itself changes during training.
- We don't compute $H$; we approximate.

This is why a single global $\eta$ is fundamentally wrong, and why every modern optimizer is some attempt to recover per-direction step sizes. Adam approximates per-parameter step sizes via $1/\sqrt{\hat v_t}$. AdamW separates weight decay from preconditioning. LARS/LAMB and muP scale per layer. See `LEARNING_RATE_DEEP_DIVE.md` §1, §10, §14 for the full story.

---

## Common failure modes (with diagnostic signatures)

| What you see | Likely cause | First thing to try |
|---|---|---|
| NaN at step 1–5 | LR way too high, or fp16 overflow | Lower $\eta$ 10x; check forward-pass magnitudes |
| NaN at step 100–500 | Warmup too short / peak LR too high | Extend warmup; lower peak $\eta$ |
| Loss flat, gradients healthy | LR too low | LR finder; raise $\eta$ |
| Loss flat, gradients vanishing | Stuck at saddle/critical point | Warm restart, perturbation |
| Oscillation with growing amplitude | Past stability boundary | Lower $\eta$; clip gradients |
| Occasional spike, recovery | Edge of stability — often fine | Add gradient clipping at norm 1.0 |
| Fine-tuning destroys pretrained capability | LR too high for transfer | Reduce 10–100x |

The single most useful debugging quantity is the **per-layer update-to-weight ratio** $\|\eta \cdot \text{update}\| / \|\theta\|$. Healthy training has this around $10^{-3}$ per layer. See `LEARNING_RATE_DEEP_DIVE.md` §3.

---

## Reference implementations (from scratch)

The implementations below are minimal but correct. Use them as the code you would whiteboard in an interview when asked "implement SGD" or "implement Adam." For real training you'd use `torch.optim.SGD` or `torch.optim.AdamW`.

### Mini-batch SGD with momentum

```python
import numpy as np

class SGDMomentum:
    """
    Mini-batch SGD with classical momentum (Polyak).
    Update:
        v_{t+1} = beta * v_t + g_t
        theta_{t+1} = theta_t - eta * v_{t+1}
    Notes:
      - beta=0.9 is standard; higher beta = more inertia.
      - Nesterov variant uses gradient at theta - eta*beta*v_t (lookahead); often slightly better.
    """
    def __init__(self, params_shape, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros(params_shape)

    def step(self, params, grad):
        self.v = self.momentum * self.v + grad
        return params - self.lr * self.v
```

In math form:

$$
v_{t+1} = \beta\, v_t + g_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

### Adam (correct, with bias correction)

```python
import numpy as np

class Adam:
    """
    Adam optimizer (Kingma & Ba, 2014).
    See math below.
    Defaults: beta1=0.9, beta2=0.999, eps=1e-8.
    """
    def __init__(self, params_shape, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(params_shape)
        self.v = np.zeros(params_shape)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

In math form:

$$
\begin{aligned}
m_t &= \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1 - \beta_1^t} \quad \text{(bias correction)} \\
\hat v_t &= \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)} \\
\theta_{t+1} &= \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon}
\end{aligned}
$$

### AdamW (decoupled weight decay)

```python
class AdamW(Adam):
    """
    AdamW (Loshchilov & Hutter, 2019).
    Identical to Adam, plus a decoupled weight decay term added directly to theta.
    Why decoupled: in plain Adam, L2 regularization (lambda * theta added to gradient)
    is divided by sqrt(v_hat), weakening regularization where gradient variance is high.
    Decoupled decay applies a uniform fractional shrinkage, recovering the
    intended regularization behavior across all parameters.
    """
    def __init__(self, params_shape, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.01):
        super().__init__(params_shape, lr, beta1, beta2, eps)
        self.wd = weight_decay

    def step(self, params, grad):
        params = super().step(params, grad)
        return params - self.lr * self.wd * params
```

In math form:

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \varepsilon} - \eta\, \lambda\, \theta_t
$$

### Linear warmup + cosine decay schedule

```python
import math

def warmup_cosine_lr(step, warmup_steps, total_steps, peak_lr, min_lr_frac=0.1):
    """
    Linear warmup over `warmup_steps`, then cosine decay to `min_lr_frac * peak_lr`.
    Returns the LR at the given step.
    """
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_frac
    return min_lr + (peak_lr - min_lr) * cosine
```

In math form:

$$
\eta(t) = \begin{cases} \eta_{\max} \cdot \dfrac{t}{W} & t \leq W \\ \eta_{\min} + \tfrac{1}{2}(\eta_{\max} - \eta_{\min})\!\left(1 + \cos\!\left(\pi \cdot \dfrac{t-W}{T-W}\right)\right) & W < t \leq T \end{cases}
$$

---

## What to practice saying out loud

Before any interview involving training:

1. "Mini-batch GD is the practical default because batch size $B$ controls a tradeoff between gradient variance ($\sigma^2/B$) and per-step cost; below the gradient noise scale, larger batches help, above it they don't."
2. "The learning rate must satisfy $\eta < 2/\lambda_{\max}(H)$ for convergence on a quadratic. For deep networks, $\lambda_{\max}$ varies across layers and during training, which is why we need adaptive optimizers, schedules, and warmup."
3. "AdamW differs from Adam with L2 because Adam's preconditioning weakens L2 wherever $\hat v$ is large; AdamW decouples the decay so it's uniform across parameters."
4. "We use linear warmup because Adam's variance estimates are noisy and residual streams uncalibrated near initialization; we use cosine decay because it dominates step decay empirically and avoids sudden shocks to the optimizer."
5. "The implicit regularization scale is $\eta/B$; that's why scaling batch size requires scaling LR, and why very large batches lose the generalization benefit of SGD noise."

These five sentences, said cleanly, get you 70% of the way through any LR-related interview.

---

## What the interviewer may ask next

(Each is fully answered in `INTERVIEW_GRILL.md`.)

1. Walk me through Adam with bias correction.
2. Why does AdamW exist?
3. What's the linear scaling rule and when does it break?
4. What's edge of stability?
5. How would you transfer LR from a small to a large model? (muP)
6. Loss spikes occasionally — what do you do?
7. Why is fine-tuning LR much smaller than pretraining LR?
8. What's the gradient noise scale?

If any of these aren't crisp for you, that's the next thing to drill.

---

## Cross-references

- `10_optimizers/` — focused tour of optimizer algorithms (deeper SGD/Momentum/Adam/AdamW/Lion comparisons).
- `11_regularization/` — weight decay vs. L2, dropout, label smoothing.
- `48_optimization_and_matrix_calculus/` — gradients, Hessians, conditioning.
- `62_frontier_training_playbook/` — production training recipes.

---

## Next steps

1. Read `LEARNING_RATE_DEEP_DIVE.md` from start to finish.
2. Drill `INTERVIEW_GRILL.md` until you can answer 40+ of 60 cold.
3. Move on to `10_optimizers/` for the per-optimizer comparisons.
