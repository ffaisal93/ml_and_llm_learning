# Topic 2: Gradient Descent & Learning Rate

## Files in this folder

| File | Purpose |
|---|---|
| [`README.md`](README.md) | Conceptual overview (this file) — read this first. |
| [`LEARNING_RATE_DEEP_DIVE.md`](LEARNING_RATE_DEEP_DIVE.md) | The core interview deep-dive on learning rate: when it works, when it fails, schedules, scaling rules, edge of stability, AdamW. **Most important file in this folder.** |
| [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) | 60 active-recall interview questions with strong answers. Drill before interviews. |

---

## What you'll learn

- The mathematics that decides whether gradient descent converges, oscillates, or diverges.
- Why mini-batch dominates batch and stochastic GD in practice — and how to defend that answer rigorously.
- How learning rate, batch size, gradient noise, and generalization are linked through a single quantity ($\eta/B$).
- The standard schedules (warmup, cosine, linear) and *why* each phase exists.
- How to read training curves and gradient norms to debug an unstable run.
- The frontier-lab vocabulary: edge of stability, critical batch size, gradient noise scale, muP.

If you can answer the **60 grill questions** in [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) cleanly, you are above the bar for an applied scientist screen on this topic.

---

## Why this topic matters in interviews

Almost every modern training recipe — for vision, NLP, RL, diffusion, and LLMs — is some variant of mini-batch gradient descent with an adaptive optimizer and a learning-rate schedule. Interviewers use questions in this area to probe:

1. **Do you understand optimization or recite slogans?** "Adam works better" is a slogan. Knowing that Adam approximates the diagonal of the Hessian via the second moment of the gradient and that it can over-rescale dimensions whose $\hat v$ is dominated by noise — that's understanding.
2. **Can you debug?** Given a loss curve and a gradient norm, can you diagnose whether the LR is too high, too low, or whether the issue lives elsewhere?
3. **Do you know how it scales?** From a 1B-parameter run to a 70B-parameter run, what changes? If you don't know what muP is, you'll struggle.
4. **Do you know modern subtleties?** Edge of stability, AdamW vs. Adam+L2, critical batch size, linear scaling rule and its limits — these are the topics that separate top candidates.

The deep-dive file goes section by section through these topics with the right level of math and the right honesty about what's settled and what isn't.

---

## Start here: the whole idea, without the notation

Before any symbols, here is the entire concept.

You are standing somewhere on a hilly landscape, in thick fog, and you want to reach the lowest point.
You cannot see the valley. All you can do is feel the ground under your feet and tell which way is
downhill. So you take a step downhill, feel again, step again, and repeat. That is gradient descent.
Everything else in this folder is a detail about *how big a step to take* and *what goes wrong when the
landscape is a strange shape*.

Now the three pieces of vocabulary, each of which is simpler than it looks.

**The knobs.** A model is a big pile of numbers — weights — and training means finding good values for
them. Those numbers are written $\theta$ (theta). When you read "find $\theta$ that makes the loss
small," read it as "find knob settings that make the model less wrong."

**The loss.** A single number saying how wrong the model currently is on your data, written $L$. Big
number, bad model. The whole game is making it small. Your height on the landscape *is* the loss, and
your position *is* the knob settings.

**The slope.** The gradient, written $\nabla L$, is just the slope of the landscape where you are
standing. It has one number per knob, each saying "if you increase this knob a little, the loss changes
by this much." One mildly annoying convention: the gradient points *uphill*, toward steepest increase.
So to go down, you subtract it. That is why the update rule has a minus sign.

Put together, one step of gradient descent is:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla L
$$

In words: **new knobs = old knobs − (step size × slope)**. That is the entire algorithm. The rest of
this topic is about that little $\eta$.

### A worked example you can do in your head

Take the simplest possible landscape, $L(\theta) = \theta^2$ — a parabola whose lowest point is at
$\theta = 0$. Its slope is $\nabla L = 2\theta$.

Start at $\theta = 10$ with a step size of $\eta = 0.1$:

| Step | Position $\theta$ | Slope $2\theta$ | Update | New $\theta$ |
|---|---|---|---|---|
| 1 | 10.0 | 20.0 | $-0.1 \times 20$ | 8.0 |
| 2 | 8.0 | 16.0 | $-0.1 \times 16$ | 6.4 |
| 3 | 6.4 | 12.8 | $-0.1 \times 12.8$ | 5.12 |
| 4 | 5.12 | 10.24 | $-0.1 \times 10.24$ | 4.10 |

It marches steadily toward zero, and notice it slows down as it gets closer — because the slope
shrinks near the bottom. That self-braking behaviour is why gradient descent settles rather than
overshooting forever.

Now redo it with $\eta = 1.1$ instead:

| Step | Position $\theta$ | Slope $2\theta$ | New $\theta$ |
|---|---|---|---|
| 1 | 10.0 | 20.0 | $10 - 22 = -12$ |
| 2 | −12.0 | −24.0 | $-12 + 26.4 = 14.4$ |
| 3 | 14.4 | 28.8 | $14.4 - 31.7 = -17.3$ |

It bounces from side to side and the bounces get **bigger**. This is divergence, and it is the single
most common training failure. The step was so large that it leapt clean over the valley and landed
higher up the opposite wall.

That contrast is the heart of this whole topic. **Too small and you crawl; too large and you explode.**
Everything about schedules, warmup, and adaptive optimizers exists to manage that tension.

### Why this is hard in practice — three complications

If every loss looked like that parabola, gradient descent would be a solved, boring problem. Real
training is harder for exactly three reasons, and every question in this folder traces back to one of
them.

**One: the landscape has many valleys, not one.** The parabola has a single lowest point, so you cannot
fail to find it. This property is called *convex*. Real neural network landscapes are *non-convex* —
many valleys, ridges, and flat plateaus — so where you end up depends partly on where you started.

**Two: the valley is a long narrow canyon, not a round bowl.** This is the one that surprises people,
and it is worth picturing properly. Imagine a valley that is extremely steep from side to side but
almost flat along its length. You want to travel *along* it to reach the low end, but the steep walls
dominate what you feel underfoot. Take a step big enough to make progress along the gentle direction
and you slam into the steep wall and bounce. Take a step small enough to be safe on the steep wall and
you inch along the gentle direction forever.

That shape is called **ill-conditioning**, and the number describing how stretched the canyon is — the
ratio of steepest curvature to gentlest — is the **condition number**. A round bowl has a condition
number of 1 and is easy. A canyon stretched a thousand-to-one is miserable, because *one step size
cannot be right for both directions at once*. Hold on to that sentence: it is the reason Adam,
AdamW, per-layer scaling, and most of the rest of the optimizer zoo exist. They are all attempts to
use a different effective step size in each direction.

**Three: you cannot feel the ground exactly.** Computing the true slope means using every training
example, which is far too slow. So you estimate it from a small random sample — a mini-batch. Your
sense of "downhill" is therefore noisy, and it wobbles from step to step. That is what *stochastic*
means. Surprisingly, that noise turns out to be useful, and later sections explain why.

### The three regimes, in plain terms

Those complications explain the three ways people compute the slope.

Think of it as polling before deciding which way to walk. You can **survey every single person** — the
most accurate reading, and far too slow to do before every step. That is batch gradient descent. You can
**ask one person** — instant, but their opinion is noisy and might send you the wrong way. That is
stochastic gradient descent in its pure form. Or you can **ask a small group**, which is nearly as fast
and much steadier than one person. That is mini-batch, and it is what essentially everyone actually
uses.

The group size is the batch size $B$. Bigger groups give steadier readings — the noise falls in
proportion to $B$ — but each reading costs more, and past a certain size the extra steadiness stops
being worth the cost.

### What to carry into the rest of this topic

If you remember four things, the dense material below will make sense:

1. Gradient descent is: **new knobs = old knobs − step size × slope.**
2. Too small a step crawls; too large a step diverges. That tension never goes away.
3. The loss surface is a stretched canyon, so **no single step size is right for every direction** —
   which is what adaptive optimizers try to fix.
4. The slope is estimated from a sample, so it is noisy — and that noise is not purely a nuisance.

The sections that follow say all of this again, precisely, with the mathematics that lets you defend it
in an interview. Read them knowing that they are the formal version of the four points above, not new
material.

---

## The same idea, stated precisely

You have a loss $L(\theta)$ and you want to find $\theta$ that makes it small. The gradient $\nabla L(\theta)$ points in the direction of steepest *increase*; subtract a small multiple of it from $\theta$ and you decrease the loss. Repeat. The "small multiple" is the learning rate $\eta$. The reason this isn't trivial: the loss surface in real deep learning is **non-convex** (many valleys, not one bowl), **ill-conditioned** (a stretched canyon — curvature varies massively across directions, so no single step size suits them all), and **stochastic** (we use noisy mini-batch estimates of $\nabla L$ rather than the true gradient). Every interesting question in this folder follows from one of these three properties.

---

## The three regimes

### Batch gradient descent

Computes $\nabla L$ over the entire dataset before each update. **Stable**, **expensive**, **rarely used at scale**. Only an option for small datasets or when exact gradients are essential (rare).

### Stochastic gradient descent (SGD, single sample)

Computes $\nabla L$ from one sample per step. **Cheap per step**, **very noisy**, **fast to start learning**. The noise has an underappreciated benefit — it's a form of implicit regularization that biases SGD toward flat minima. But variance is too high for most practical use.

### Mini-batch gradient descent

The default. Batch size $B$ (typically 32–8192) trades stability for speed. Variance of the gradient estimate is $\sigma^2 / B$. The right $B$ depends on hardware (memory, parallelism) and on the gradient noise scale (after which doubling $B$ stops paying off). See [`LEARNING_RATE_DEEP_DIVE.md`](LEARNING_RATE_DEEP_DIVE.md) §6 for critical batch size.

---

## Why the learning rate is the master hyperparameter

Here $\lambda_{\max}(H)$ is the curvature in the *steepest* direction of the canyon and $\lambda_{\min}(H)$ the curvature in the *gentlest*; the Hessian $H$ is what packages all those curvatures together. For a quadratic loss with Hessian $H$, GD converges only if $0 < \eta < 2/\lambda_{\max}(H)$. Above that, you diverge in the sharpest direction. Below $1/\lambda_{\max}(H)$, you converge but waste steps in flatter directions. The optimal rate is $2/(\lambda_{\max} + \lambda_{\min})$, and convergence speed depends on the **condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$.

In real deep networks:

- $\lambda_{\max}(H)$ varies by orders of magnitude across layers.
- $H$ itself changes during training.
- We don't compute $H$; we approximate.

This is why a single global $\eta$ is fundamentally wrong, and why every modern optimizer is some attempt to recover per-direction step sizes. Adam approximates per-parameter step sizes via $1/\sqrt{\hat v_t}$. AdamW separates weight decay from preconditioning. LARS/LAMB and muP scale per layer. See [`LEARNING_RATE_DEEP_DIVE.md`](LEARNING_RATE_DEEP_DIVE.md) §1, §10, §14 for the full story.

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

The single most useful debugging quantity is the **per-layer update-to-weight ratio** $\|\eta \cdot \text{update}\| / \|\theta\|$. Healthy training has this around $10^{-3}$ per layer. See [`LEARNING_RATE_DEEP_DIVE.md`](LEARNING_RATE_DEEP_DIVE.md) §3.

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

(Each is fully answered in [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).)

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

1. Read [`LEARNING_RATE_DEEP_DIVE.md`](LEARNING_RATE_DEEP_DIVE.md) from start to finish.
2. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) until you can answer 40+ of 60 cold.
3. Move on to `10_optimizers/` for the per-optimizer comparisons.
