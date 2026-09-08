# Optimizer evolution: from SGD to AdamW

The easiest way to remember optimizers is not as six unrelated formulas. Treat them as a sequence of
repairs. Each optimizer keeps the useful part of the previous idea, adds one piece of state, and fixes
one specific failure.

```text
Full gradient descent
  └─ problem: one exact update scans the whole dataset
     Mini-batch SGD
       └─ problem: noisy steps zigzag and forget useful direction
          Momentum
            └─ problem: every parameter still shares one step scale
               AdaGrad
                 └─ problem: squared-gradient history grows forever
                    RMSProp
                      └─ problem: adaptive scale still uses a noisy current direction
                         Adam
                           └─ problem: L2 penalty is distorted by adaptive scaling
                              AdamW
```

For every optimizer, ask the same three questions:

1. **Direction:** does it use the current gradient or a smoothed history?
2. **Scale:** is the step size global or adapted per parameter?
3. **Memory:** what history is stored, and does old history fade?

## Notation and the starting point

Let $	heta_t$ be the parameter vector, $g_t$ the gradient estimate at step $t$, and $eta$ the
learning rate. Full-batch gradient descent uses every training example before moving:

$$
g_t = \frac{1}{N}\sum_{i=1}^{N}\nabla_\theta L_i(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta g_t.
$$

The gradient gives the locally steepest uphill direction, so subtracting it moves downhill. The
learning rate says how far we trust that local slope.

The geometric problem appears in a narrow valley. A steep direction requires a small $eta$ to avoid
crossing the valley repeatedly, while a flat direction needs a large $eta$ to make progress. One
global learning rate cannot satisfy both.

## Fix 1 — mini-batch SGD makes updates affordable

**Problem in full gradient descent:** an exact gradient requires a pass over all $N$ examples before
every update.

Mini-batch SGD estimates it from $B \ll N$ examples:

$$
g_t = \frac{1}{B}\sum_{i\in\mathcal{B}_t}\nabla_\theta L_i(\theta_t),
\qquad
\theta_{t+1}=\theta_t-\eta g_t.
$$

The estimate is noisy but cheap. We get frequent updates, efficient matrix operations on accelerators,
and noise that can sometimes help exploration and generalization.

**What remains broken:** SGD has no memory. If one coordinate alternates $+10,-10,+10,-10$, SGD
zigzags. If another stays $+1,+1,+1,+1$, SGD moves consistently but does not accumulate speed. It also
applies the same global learning-rate scale to every coordinate.

> **Memory hook:** SGD looks at the current mini-batch and takes a step.

## Fix 2 — momentum remembers persistent direction

**Problem in SGD:** the latest noisy batch can reverse the update even when the longer-term direction
is stable.

Momentum keeps an exponential moving average of gradients:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
\theta_{t+1}=\theta_t-\eta m_t.
$$

Here $eta_1$ is commonly near $0.9$. Gradients with a consistent sign accumulate; alternating
gradients cancel. In the narrow-valley picture, momentum damps side-to-side oscillation and accelerates
motion along the valley floor.

Some texts use $m_t=\beta_1m_{t-1}+g_t$. That changes the scale of $m_t$, not the idea. State your
convention before comparing learning rates.

**What remains broken:** direction is smoothed, but every parameter still shares the same base scale.
A parameter with naturally large gradients can dominate one with rare or small gradients.

> **Memory hook:** momentum asks, “Which direction has stayed useful?”

## Fix 3 — AdaGrad adapts the scale of each parameter

**Problem in momentum:** it remembers direction but does not normalize different coordinate scales.

AdaGrad accumulates squared gradients element by element:

$$
G_t=G_{t-1}+g_t^2,
\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{G_t}+\epsilon}.
$$

Squaring removes sign and measures magnitude. A frequently updated coordinate develops a large
denominator and receives smaller later steps. A rare coordinate keeps a relatively small denominator
and receives a larger effective step. This is especially useful for sparse features, such as uncommon
tokens in a large vocabulary.

For coordinate $j$, the effective learning rate is

$$
\eta_{t,j}^{\text{eff}}=\frac{\eta}{\sqrt{G_{t,j}}+\epsilon}.
$$

**What remains broken:** $G_t$ contains the entire history and can only increase. Even gradients from
the distant past keep shrinking today's step. Eventually the optimizer may almost stop learning.

> **Memory hook:** AdaGrad asks, “How much gradient has this parameter received in its entire life?”

## Fix 4 — RMSProp lets old scale information fade

**Problem in AdaGrad:** its denominator grows forever.

RMSProp replaces the lifetime sum with an exponential moving average:

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
\qquad
\theta_{t+1}=\theta_t-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}.
$$

Recent squared gradients matter most; older ones decay geometrically. RMSProp keeps AdaGrad's
per-parameter normalization but can recover when the gradient scale changes during training.

**What remains broken:** the denominator has a useful history of magnitude, but the numerator is still
the current noisy gradient. RMSProp does not, in its basic form, preserve momentum's estimate of
persistent direction.

> **Memory hook:** RMSProp asks, “How large have this parameter's gradients been recently?”

## Fix 5 — Adam combines smoothed direction and adaptive scale

**Problem in RMSProp:** it adapts magnitude but still reacts to the current direction.

Adam keeps two exponential moving averages:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The first moment $m_t$ is momentum-like direction. The second moment $v_t$ is RMSProp-like scale.
Because both start at zero, their early values are biased toward zero. Adam corrects that:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

The update is

$$
\theta_{t+1}
=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

Read the fraction literally:

```text
                 smoothed direction
Adam step =  η × --------------------
                 recent gradient scale
```

Typical defaults are $eta_1=0.9$, $eta_2=0.999$, and $\epsilon=10^{-8}$, but Adam still depends on
the learning rate, warmup, schedule, batch size, and clipping.

**What remains broken:** Adam stores two state tensors per parameter, may generalize differently from
well-tuned SGD with momentum, and couples ordinary L2 regularization to its adaptive denominator.

> **Memory hook:** Adam asks both, “Where have I consistently been going?” and “How large are this
> parameter's gradients normally?”

## Why Adam needs bias correction

Suppose the gradient is constant and $m_0=0$. After the first update,

$$
m_1=(1-\beta_1)g.
$$

With $\beta_1=0.9$, this is only $0.1g$. Dividing by $1-\beta_1^1=0.1$ recovers $g$. More generally,
the missing mass after $t$ steps is exactly $\beta_1^t$, so dividing by $1-\beta_1^t$ corrects the
zero-initialization bias. The same reasoning applies to $v_t$.

Bias correction does not remove the need for warmup in large-model training. It fixes the expected
scale of the moments; warmup also protects against high variance in early estimates and sharp initial
curvature.

## Fix 6 — AdamW decouples weight decay

**Problem in Adam:** adding an L2 penalty to the loss adds $\lambda\theta_t$ to the gradient. Adam then
divides that penalty by each parameter's adaptive scale:

$$
\frac{g_t+\lambda\theta_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The resulting shrinkage depends on gradient history. Two equal weights can decay by different amounts
because their second moments differ.

AdamW performs optimization and shrinkage as separate operations:

$$
\theta_{t+1}
=(1-\eta\lambda)\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

The adaptive term learns from data; the first term shrinks weights predictably. Biases and normalization
parameters are commonly excluded from decay because shrinking them is usually not the intended
regularization.

**What remains broken:** AdamW still needs two moment buffers, a schedule, and often warmup. It is a
cleaner Adam, not a universally best optimizer.

> **Memory hook:** AdamW is Adam with weight shrinkage performed separately.

## The progression in one table

| Optimizer | Direction used | Per-parameter scale | History | Main unresolved weakness |
|---|---|---|---|---|
| Mini-batch SGD | Current gradient | No | None | Noisy zigzag; one global scale |
| Momentum | Smoothed gradient | No | Recent directions | Different coordinates still share a scale |
| AdaGrad | Current gradient | Yes | All squared gradients | Effective rates shrink forever |
| RMSProp | Current gradient | Yes | Recent squared gradients | Direction remains noisy |
| Adam | Smoothed gradient | Yes | Recent gradients and squares | Adaptive L2 is not clean weight decay |
| AdamW | Smoothed gradient | Yes | Adam state + separate decay | Memory cost and schedule sensitivity remain |

## One gradient sequence, six interpretations

Suppose one coordinate receives $10,-10,10,-10$ while another receives $1,1,1,1$.

- **SGD** follows every sign change, so the first coordinate zigzags while the second moves steadily.
- **Momentum** cancels much of the alternating direction and accumulates the consistent direction.
- **AdaGrad** sees much more lifetime squared gradient in the first coordinate, so it scales that
  coordinate down; both denominators continue growing forever.
- **RMSProp** also scales down the first coordinate, but it forgets old squares if the behavior changes.
- **Adam** combines momentum's direction estimate with RMSProp's coordinate scale.
- **AdamW** makes the same adaptive update, then applies independent parameter shrinkage.

This example separates two ideas that are often blurred together: momentum changes the **numerator**;
AdaGrad and RMSProp change the **denominator**. Adam changes both.

## Code from memory

The following trace exposes the state each optimizer remembers. All operations are elementwise.

```python
import numpy as np

grads = np.array([[10., 1.], [-10., 1.], [10., 1.], [-10., 1.]])
b1, b2, eps = 0.9, 0.9, 1e-8
m = np.zeros(2)          # momentum / Adam first moment
G = np.zeros(2)          # AdaGrad lifetime squares
v = np.zeros(2)          # RMSProp / Adam recent squares

for t, g in enumerate(grads, start=1):
    m = b1 * m + (1 - b1) * g
    G = G + g * g
    v = b2 * v + (1 - b2) * g * g

m_hat = m / (1 - b1 ** len(grads))
v_hat = v / (1 - b2 ** len(grads))

print("momentum direction:", np.round(m, 4))
print("AdaGrad denominator:", np.round(np.sqrt(G), 4))
print("RMSProp denominator:", np.round(np.sqrt(v), 4))
print("Adam normalized step:", np.round(m_hat / (np.sqrt(v_hat) + eps), 4))
```

Expected output:

```text
momentum direction: [-0.181  0.3439]
AdaGrad denominator: [20.  2.]
RMSProp denominator: [5.8643 0.5864]
Adam normalized step: [-0.0526  1.    ]
```

The alternating coordinate has a large scale estimate but almost no persistent direction, so Adam's
normalized step is small. The consistent coordinate has a smaller raw gradient but a strong normalized
step.

## Interview questions

### Q1. Give the optimizer progression in one minute.

Mini-batch SGD makes updates affordable but has no memory. Momentum smooths gradient direction, so
oscillation cancels and consistent motion accelerates. AdaGrad adds a separate scale per parameter using
all past squared gradients, but its effective learning rates shrink forever. RMSProp replaces that
lifetime sum with a moving average. Adam combines momentum's smoothed direction with RMSProp's adaptive
scale and corrects the moments' initialization bias. AdamW keeps Adam's update but moves weight decay
outside the adaptive gradient calculation.

### Q2. Why do adaptive optimizers square gradients?

They need magnitude, not direction. Ordinary gradients of alternating sign can sum to zero even when a
coordinate repeatedly has large gradients. Squares stay positive, so the denominator correctly records
the coordinate's typical scale.

### Q3. What is the exact difference between AdaGrad and RMSProp?

AdaGrad sums every squared gradient with equal permanent weight. RMSProp uses an exponential moving
average, so recent squares matter more and old history fades. RMSProp therefore retains per-coordinate
scaling without forcing the effective learning rate monotonically toward zero.

### Q4. Is Adam simply momentum plus RMSProp?

That is the right intuition, with two qualifications. Adam maintains its own first and second moments
using specified decay rates, and it bias-corrects both because they start at zero. Its numerator is a
smoothed direction and its denominator is a smoothed squared-gradient scale.

### Q5. Why is AdamW different from Adam with L2 regularization?

L2 adds $\lambda\theta$ to the gradient, so Adam adaptively rescales the penalty using each parameter's
second moment. AdamW applies shrinkage directly to the parameter, outside the adaptive update. That makes
weight decay independent of gradient history and easier to tune.

### Q6. When might SGD with momentum beat AdamW?

AdamW often reaches a useful solution faster and handles poorly scaled or sparse gradients well. SGD
with momentum can still match or beat its final generalization on some vision and well-conditioned
problems when the learning-rate schedule is tuned. Optimizer choice is an empirical comparison under an
equal compute budget, not a universal ranking.

### Q7. What is the optimizer-state memory cost?

For $P$ parameters, SGD stores no persistent state, momentum stores roughly $P$ values, and Adam/AdamW
store about $2P$ values for $m$ and $v$, often in fp32 even when model weights use lower precision. At
large scale those two buffers can exceed the memory used by the weights themselves, motivating sharding
and lower-memory optimizers.

## Done when

- You can reconstruct the chain SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW from the weakness at
  each arrow.
- You can explain which optimizer changes direction, which changes scale, and which changes both.
- You can derive Adam's bias correction from zero initialization.
- You can explain, algebraically, why L2 inside Adam is not equivalent to decoupled weight decay.
- You can state one reason to choose AdamW and one reason to benchmark SGD with momentum.

