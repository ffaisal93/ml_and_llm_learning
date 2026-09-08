# Activation-function evolution: from sigmoid to SwiGLU

Activation functions are easiest to remember as a history of gradient problems. Each step keeps a
useful property, fixes one failure, and introduces a new tradeoff. The progression is not a strict
replacement chain—sigmoid remains correct for binary outputs, for example—but it explains why hidden
layers moved from saturating functions to rectifiers, smooth self-gates, and learned gates.

```text
Sigmoid
  └─ weakness: positive-only output and saturation on both sides
     Tanh
       └─ fix: zero-centred output; weakness: still saturates
          ReLU
            └─ fix: cheap, non-saturating positive side; weakness: dead units
               Leaky ReLU / PReLU
                 └─ fix: negative-side gradient; weakness: fixed kink/slope
                    ELU / SELU
                      └─ fix: smooth negative branch and better mean/variance behavior
                         GELU / SiLU
                           └─ fix: smooth input-dependent gating; strong modern default
                              GEGLU / SwiGLU
                                └─ fix: learn what information passes through an FFN
```

For every activation, ask four questions:

1. **Range:** is the output bounded, zero-centred, or unbounded?
2. **Gradient:** where does the derivative vanish, stay constant, or become discontinuous?
3. **Computation:** is it a comparison, an exponential, or an extra learned projection?
4. **Role:** is it for a hidden layer, an output distribution, or a gate?

## Why a network needs an activation

Without a nonlinearity, depth collapses. For two linear layers,

$$
h=xW_1+b_1,
\qquad
y=hW_2+b_2,
$$

substitution gives another affine map:

$$
y=x(W_1W_2)+(b_1W_2+b_2).
$$

A hundred linear layers still describe one linear transformation. Applying a nonlinear function
$\phi$ between layers prevents that collapse:

$$
h=\phi(xW_1+b_1).
$$

The activation therefore controls both **expressivity** in the forward pass and **gradient flow** in
the backward pass.

## Stage 1 — sigmoid makes a smooth gate

The logistic sigmoid maps any real input to $(0,1)$:

$$
\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

**Strengths.** It is smooth, bounded, and interpretable as a probability or gate. It remains the
natural output activation for binary and multi-label classification and appears inside LSTM gates.

**Weaknesses.** Its derivative is at most $0.25$ and approaches zero for large $|x|$. Backpropagation
multiplies derivatives across layers, so even the best-case factor through ten sigmoid layers is
$0.25^{10}\approx9.54\times10^{-7}$. The output is also always positive, so hidden activations are not
zero-centred and weight updates can zigzag.

**What led next.** Keep smooth bounded nonlinear behavior, but centre the hidden signal around zero.

> **Memory hook:** sigmoid is a good probability and gate, but a poor deep hidden activation.

## Stage 2 — tanh centres the signal

The hyperbolic tangent maps to $(-1,1)$:

$$
\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}},
\qquad
\frac{d}{dx}\tanh(x)=1-\tanh^2(x).
$$

**Strengths.** Tanh is zero-centred and has derivative $1$ at the origin. It was a meaningful
improvement for hidden states and remains useful when a bounded signed state is required, including
inside recurrent cells.

**Weaknesses.** It still saturates near $-1$ and $1$, so units with large pre-activations pass almost
no gradient. Xavier-style initialization can keep early activations near the useful central region,
but it cannot guarantee that they stay there throughout training.

**What led next.** Stop squeezing the positive side into a bounded interval. Preserve a large gradient
where the unit is active.

> **Memory hook:** tanh fixes centring, not saturation.

## Stage 3 — ReLU makes deep networks practical

The rectified linear unit is

$$
\operatorname{ReLU}(x)=\max(0,x),
\qquad
\operatorname{ReLU}'(x)=
\begin{cases}
0,&x<0,\\
1,&x>0.
\end{cases}
$$

At $x=0$ the derivative is undefined; libraries choose a subgradient, usually zero.

**Strengths.** ReLU is extremely cheap. Its positive branch has derivative exactly one, so it does not
saturate there. Negative outputs are exactly zero, creating sparse activations. Combined with He
initialization and residual connections, it enabled much deeper networks than sigmoid or tanh.

**Weaknesses.** A unit whose pre-activation is negative for every example outputs zero and receives
zero gradient. A large update can push the unit into this state permanently: the **dying ReLU** problem.
Its output is unbounded and its derivative jumps at zero.

**What led next.** Retain ReLU's cheap positive branch but leave a recovery gradient on the negative
side.

> **Memory hook:** ReLU solves positive-side saturation but can permanently switch a unit off.

## Stage 4 — Leaky ReLU and PReLU keep dead units alive

Leaky ReLU gives negative inputs a small slope $\alpha$:

$$
\operatorname{LeakyReLU}(x)=\max(\alpha x,x),
\qquad \alpha\approx0.01.
$$

PReLU uses the same equation but learns $\alpha$ from data.

**Strengths.** The negative-side derivative is no longer zero, so a unit can move back into the active
region. Computation remains almost as cheap as ReLU. PReLU lets each layer or channel learn how much
negative signal to retain.

**Weaknesses.** Leaky ReLU introduces a manually chosen slope; PReLU introduces parameters and can
overfit in small data regimes. Both retain a sharp kink at zero, and a fixed linear negative branch
does not adapt its gate to the input.

**What led next.** Make the negative transition smooth and encourage activations to remain closer to
zero mean.

> **Memory hook:** Leaky ReLU is ReLU with an escape route.

## Stage 5 — ELU and SELU smooth the negative branch

The exponential linear unit is

$$
\operatorname{ELU}(x)=
\begin{cases}
x,&x>0,\\
\alpha(e^x-1),&x\le0.
\end{cases}
$$

**Strengths.** ELU keeps ReLU's linear positive side, provides a smooth negative gradient, and allows
negative outputs. Its negative saturation can pull the activation mean closer to zero.

**Weaknesses.** The exponential is more expensive than a maximum, and the negative branch still
saturates. Its practical gain over a well-configured ReLU is architecture-dependent.

SELU chooses fixed scale and slope constants so activations tend toward zero mean and unit variance:

$$
\operatorname{SELU}(x)=\lambda
\begin{cases}
x,&x>0,\\
\alpha(e^x-1),&x\le0.
\end{cases}
$$

That self-normalizing behavior depends on compatible assumptions: LeCun-normal initialization,
roughly independent activations, sufficiently wide layers, and alpha dropout. Batch normalization,
arbitrary initialization, or incompatible residual structure can remove the guarantee.

**What led next.** Rather than hand-designing a negative branch, use a smooth input-dependent gate that
gradually controls how much of $x$ passes.

> **Memory hook:** ELU smooths the ReLU repair; SELU adds a conditional self-normalization recipe.

## Stage 6 — GELU and SiLU use smooth self-gating

GELU weights the input by the probability that a standard normal variable is below it:

$$
\operatorname{GELU}(x)=x\Phi(x).
$$

A common approximation is

$$
\operatorname{GELU}(x)\approx
\frac{x}{2}\left(1+\tanh\left[\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)\right]\right).
$$

SiLU, also called Swish with coefficient one, is

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

**Strengths.** Both are smooth and preserve a small negative signal instead of discarding it. Their
gate changes continuously with the input: strongly positive values pass almost linearly, strongly
negative values are suppressed, and uncertain values near zero pass partially. GELU became common in
BERT/GPT-style transformers; SiLU is common in modern vision models and inside SwiGLU.

**Weaknesses.** They require more computation than ReLU, are slightly non-monotonic on the negative
side, and their advantage is empirical rather than a theorem. ReLU can remain the right choice when
latency, integer-friendly inference, or simplicity dominates a small quality difference.

**What led next.** A fixed activation can only gate a value using that same scalar. A learned gated
unit can use one projection to decide how much of another projection should pass.

> **Memory hook:** GELU and SiLU are smooth self-gates; ReLU is a hard gate.

## Stage 7 — GLU variants learn the gate

A gated linear unit creates two projections from the same input:

$$
u=xW_u,
\qquad
g=xW_g,
\qquad
\operatorname{GLU}(x)=u\odot\sigma(g).
$$

The gate branch decides which features of the value branch pass. ReGLU, GEGLU, and SwiGLU replace the
sigmoid gate with ReLU, GELU, and SiLU respectively. A transformer SwiGLU feed-forward block is

$$
\operatorname{FFN}(x)
=\left(\operatorname{SiLU}(xW_g)\odot xW_u\right)W_o.
$$

**Strengths.** The model learns conditional feature routing rather than applying one fixed scalar
curve. SwiGLU and GEGLU have shown consistent parameter-matched gains in transformer feed-forward
blocks and are common modern LLM choices.

**Weaknesses.** A vanilla feed-forward network has two large matrices; a gated block has three. To
compare fairly, reduce the gated hidden width. If a vanilla block expands from $d$ to $4d$, it has
roughly $8d^2$ parameters. A gated block at width $h$ has roughly $3dh$, so parameter matching gives
$h\approx\frac{8}{3}d$ rather than $4d$. Gating also adds memory traffic and elementwise work.

**What comes next.** There is no single inevitable successor. Current work explores squared ReLU,
learned mixtures, sparsely activated experts, and hardware-friendly approximations. The choice is now
an architecture-and-efficiency tradeoff, not merely a vanishing-gradient repair.

> **Memory hook:** SwiGLU is SiLU plus a learned second branch that controls what passes.

## Hidden activations and output activations are different choices

Do not describe softmax as the next hidden activation after SwiGLU. The output layer is determined by
the probabilistic task:

| Task | Output activation | Typical loss | Interpretation |
|---|---|---|---|
| Binary classification | Sigmoid | Binary cross-entropy | One Bernoulli probability |
| Multi-label classification | One sigmoid/class | Sum of binary cross-entropies | Independent labels |
| Single-label multiclass | Softmax | Cross-entropy | One categorical distribution |
| Regression | Identity, or task-specific constraint | MSE, MAE, likelihood loss | Continuous quantity |
| Positive scalar | Softplus or exponential | Likelihood/task loss | Positive support |

Hidden activations are chosen for optimization, capacity, compute, and inductive bias. Output
activations encode the support and dependence structure of the prediction.

## Initialization must match the activation

An activation changes signal variance, so weight initialization and activation choice are coupled.

- **Sigmoid/tanh:** Xavier or LeCun-style scaling keeps early pre-activations near the non-saturated
  region.
- **ReLU/Leaky ReLU:** He initialization compensates for roughly half the signal being removed:
  $\operatorname{Var}(W)\approx2/n_{\text{in}}$ for ReLU.
- **SELU:** LeCun-normal initialization and alpha dropout are part of the self-normalizing recipe.
- **Transformer GELU/SiLU/SwiGLU:** residual paths, normalization placement, depth scaling, and the
  feed-forward width matter alongside the activation.

A poor pairing can make a good activation look broken before training has meaningfully started.

## The progression in one table

| Activation | Main strength | Main weakness | Why the next idea appeared |
|---|---|---|---|
| Sigmoid | Smooth probability/gate in $(0,1)$ | Saturates; positive-only hidden signal | Centre the activation |
| Tanh | Zero-centred and bounded | Still saturates both sides | Keep a large active gradient |
| ReLU | Cheap; derivative one for $x>0$ | Dead units; hard zero branch | Preserve a negative gradient |
| Leaky ReLU/PReLU | Dead units can recover | Fixed/learned slope; non-smooth kink | Smooth the negative branch |
| ELU/SELU | Negative outputs; smoother mean behavior | Exponential cost; assumptions/saturation | Use smooth input-dependent gates |
| GELU/SiLU | Smooth self-gating; strong empirical quality | More compute; small gains are task-dependent | Learn a separate gate |
| GEGLU/SwiGLU | Conditional learned feature routing | Third matrix and memory cost | Current frontier is quality/efficiency co-design |

## Code from memory

This dependency-free trace prints each activation and a numerical derivative at three inputs.

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def relu(x):
    return max(0.0, x)

def leaky_relu(x, alpha=0.01):
    return x if x >= 0 else alpha * x

def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))

def silu(x):
    return x * sigmoid(x)

def derivative(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2 * h)

functions = [sigmoid, math.tanh, relu, leaky_relu, gelu, silu]
for fn in functions:
    values = [(round(fn(x), 4), round(derivative(fn, x), 4))
              for x in (-6.0, 0.0, 6.0)]
    print(f"{fn.__name__:10s} {values}")
```

Expected output:

```text
sigmoid    [(0.0025, 0.0025), (0.5, 0.25), (0.9975, 0.0025)]
tanh       [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)]
relu       [(0.0, 0.0), (0.0, 0.5), (6.0, 1.0)]
leaky_relu [(-0.06, 0.01), (0.0, 0.505), (6.0, 1.0)]
gelu       [(-0.0, -0.0), (0.0, 0.5), (6.0, 1.0)]
silu       [(-0.0148, -0.0123), (0.0, 0.5), (5.9852, 1.0123)]
```

The central difference returns the midpoint of the left and right slopes at ReLU's kink, so it prints
$0.5$ at zero; a framework is free to choose another subgradient there. Notice the two distinct
failures at $x=-6$: sigmoid/tanh are saturated, while ReLU is exactly dead. Leaky ReLU retains a small
recovery gradient, and GELU/SiLU transition smoothly.

## Interview questions

### Q1. Give the activation progression in one minute.

Sigmoid is smooth and probabilistic but saturates and is not zero-centred. Tanh centres the signal but
still saturates. ReLU removes positive-side saturation and is cheap, which made deep networks practical,
but its zero negative branch can kill units. Leaky ReLU and ELU restore a negative gradient; GELU and
SiLU turn the hard threshold into a smooth input-dependent gate. Modern transformer feed-forward blocks
go one step further with GEGLU or SwiGLU: one learned projection gates another.

### Q2. Why did ReLU train deep networks better than sigmoid?

On its active side ReLU's derivative is one, while sigmoid's derivative is at most $0.25$ and becomes
near zero in saturation. Repeated sigmoid derivatives shrink gradients exponentially with depth. ReLU
also costs only a comparison. Its improvement is conditional: dead units, bad initialization, or an
excessive learning rate can still destroy gradient flow.

### Q3. What is the dying-ReLU problem, and how do you diagnose it?

A neuron is dead when its pre-activation is negative for every input, making both its output and gradient
zero. Track the fraction of exactly zero activations per layer and the fraction of units that are never
positive over a representative batch. Fix the cause with He initialization or a lower learning rate,
and use Leaky ReLU, GELU, or SiLU when a recovery gradient is valuable.

### Q4. GELU versus SiLU: what is the principled difference?

GELU is $x\Phi(x)$, so its gate is the standard normal CDF. SiLU is $x\sigma(x)$, so its gate is logistic.
Both are smooth, slightly non-monotonic self-gates with similar behavior. In practice the difference is
usually empirical and small; benchmark them inside the actual architecture and hardware path.

### Q5. Why does SwiGLU use three matrices, and how do you compare it fairly?

It needs a value projection, a gate projection, and an output projection. A vanilla $4d$ feed-forward
block has about $8d^2$ parameters; a gated block has about $3dh$. Set $h\approx8d/3$ to match parameters
before claiming a quality gain. Otherwise the gated model may simply be larger.

### Q6. When should sigmoid still be used?

Use it when the model needs an independent probability or differentiable gate: binary outputs,
multi-label outputs, LSTM gates, and mixture gates. Avoid stacking it as the ordinary hidden activation
of a deep feed-forward network because saturation makes gradient flow unnecessarily difficult.

### Q7. Is there a universally best hidden activation?

No. ReLU is cheap and remains strong for many convolutional and latency-sensitive models. GELU is a
common transformer baseline, SiLU is common in vision and gating, and SwiGLU/GEGLU are strong LLM
feed-forward choices when their extra projection is affordable. Compare quality at matched parameters,
training compute, and inference cost.

## Done when

- You can reconstruct sigmoid → tanh → ReLU → Leaky/ELU → GELU/SiLU → SwiGLU from the weakness at each
  arrow.
- You can distinguish saturation from a dead unit and explain how their gradients differ.
- You can pair sigmoid, softmax, and identity outputs with the appropriate prediction task.
- You can explain why He initialization matches ReLU and why SELU requires a complete recipe.
- You can derive the parameter-matched SwiGLU width of approximately $8d/3$.

