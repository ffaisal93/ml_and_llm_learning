# Neural Networks Fundamentals — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

Every modern model — transformers, CNNs, diffusion U-Nets, MoE — is built on the basics covered here. This deep dive nails MLPs, activations, initialization, backpropagation, and the gradient pathologies that motivate every later innovation (residual connections, normalization, modern optimizers).

---

## 1. The MLP — what's actually happening

A multilayer perceptron is a stack of affine transforms with non-linearities:

$$
h^{(\ell)} = \sigma\big(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\big)
$$

with $h^{(0)} = x$ and final layer producing logits.

Why non-linearity is essential: without $\sigma$, the whole network collapses to a single affine map $W_L \cdots W_1 x + \text{const}$. No expressive gain from depth.

**Universal approximation theorem (Cybenko 1989, Hornik 1991):** an MLP with one hidden layer of arbitrary width can approximate any continuous function on a compact domain to arbitrary precision, provided the activation is non-polynomial. *Width* is enough in principle. *Depth* is what makes it tractable — fewer parameters for the same expressivity.

**Why depth helps in practice:** depth gives compositional structure. Some functions need exponentially wide shallow networks but only polynomially deep ones (Telgarsky 2016). Hierarchical features (edges → parts → objects) benefit from depth.

> **Saying it out loud.** An MLP is just "multiply, bend, repeat." Each layer takes a weighted sum of everything from the previous layer, then pushes it through a non-linear function, and you stack that as many times as you need. The bending is the whole point — without it, ten layers of multiplication collapse into one matrix and depth buys you literally nothing. Now, in theory a single wide layer can approximate any continuous function, which is the universal approximation theorem, but the width you'd need can be exponential. So depth is what makes it practical: deep networks reuse features compositionally — edges into parts into objects — and get the same expressiveness with polynomially many parameters instead of exponentially many.

---

## 2. Activations — how to choose

| Activation | Formula | Range | Used in | Notes |
|---|---|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | $(0,1)$ | Old (1980s) | Saturates, vanishing gradient |
| Tanh | $(e^x - e^{-x})/(e^x + e^{-x})$ | $(-1, 1)$ | RNNs | Zero-centered but still saturates |
| ReLU | $\max(0, x)$ | $[0, \infty)$ | Default | Cheap, sparse, dying ReLU problem |
| Leaky ReLU | $\max(\alpha x, x)$, $\alpha \approx 0.01$ | $\mathbb{R}$ | Some CNNs | Fixes dying ReLU |
| GELU | $x \cdot \Phi(x)$ | $\mathbb{R}$ | Transformers | Smooth, standard in BERT/GPT-2 |
| SiLU/Swish | $x \cdot \sigma(x)$ | $\mathbb{R}$ | Modern LLMs (Llama) | Smooth, slightly better than GELU empirically |
| GLU/SwiGLU | $\sigma(W_g x) \odot (W x)$ | $\mathbb{R}$ | Llama, PaLM | Gating doubles param count for FFN; standard now |

**Sigmoid problems:**
- Saturates for $|x| > 5$: gradient $\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 1/4$ peaks at $x = 0$. Stacked sigmoids → gradient vanishes exponentially with depth.
- Not zero-centered: outputs in $(0,1)$ → all-positive activations push gradient updates of weights to alternate sign in odd patterns.

**ReLU benefits:**
- Cheap: just a max.
- Non-saturating for $x > 0$: gradient is exactly 1, no decay.
- Sparse: ~50% of activations are zero, induces implicit regularization.

**Dying ReLU:** if a neuron's pre-activation goes negative for all training data, gradient is 0 forever — neuron is dead. Mitigations: Leaky ReLU, GELU, careful initialization, lower learning rate.

**GELU:** $x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Smooth and stochastic interpretation: "stochastic regularizer" multiplies $x$ by a Bernoulli with parameter $\Phi(x)$. Standard in transformers since BERT/GPT-2.

**SwiGLU:** Used in Llama, PaLM. The FFN is $\text{SwiGLU}(x) = \text{Swish}(x W_g) \odot (x W) \cdot W_{\text{out}}$. Two parallel projections with element-wise gating. Costs ~50% more params than vanilla FFN but consistently better.

> **Saying it out loud.** The activation is where all the non-linearity lives, and the history of the field is basically the history of fixing its gradient. Sigmoid saturates — its derivative tops out at a quarter and collapses toward zero once the input is past about five — so stacking sigmoids multiplies your gradient into oblivion. ReLU fixed that by having a derivative of exactly one wherever it's active, which is what made deep networks trainable at all. Its downside is the dead neuron: once a unit outputs zero for everything, the gradient is zero and it never comes back. Modern nets use smooth variants — GELU, SiLU, and gated versions like SwiGLU — which keep a small gradient everywhere and buy a consistent point or so of quality.

---

## 3. Loss functions — pair with output activation

| Task | Output activation | Loss | Why |
|---|---|---|---|
| Regression | Identity | MSE: $\frac{1}{2}\|y - \hat{y}\|^2$ | Gaussian likelihood |
| Binary classification | Sigmoid | BCE: $-y \log p - (1-y) \log(1-p)$ | Bernoulli MLE |
| Multi-class | Softmax | Cross-entropy | Categorical MLE |
| Multi-label | Sigmoid (per class) | Sum of BCE | Independent Bernoullis |

The activation–loss pairings aren't accidents. They're the canonical link function for the corresponding GLM (sigmoid+BCE = logistic regression; softmax+CE = multinomial logistic regression). They make the gradient simple: $\nabla_z \mathcal{L} = \hat{y} - y$ in all three classification cases. Mismatched pairings (e.g., MSE on softmax outputs) cause flat loss surfaces and slow training.

> **Saying it out loud.** The pairings of output activation and loss aren't conventions, they're the same object seen twice. Sigmoid with binary cross-entropy is logistic regression; softmax with cross-entropy is its multi-class version; identity with squared error is a Gaussian likelihood. Each pairing is the maximum-likelihood loss for that output distribution, and the payoff is that the gradient at the output simplifies to predicted minus actual — no leftover derivative of the activation to shrink it. That's why mismatching them hurts: put squared error on a sigmoid and the gradient picks up the sigmoid's derivative, so the examples you're most confidently wrong about produce the smallest updates. Exactly backwards.

---

## 4. Forward pass — compute graph

For a single-layer network with input $x$, weights $W$, output $\hat{y}$, target $y$:

$$
z = W x + b, \quad \hat{y} = \sigma(z), \quad \mathcal{L} = \text{loss}(\hat{y}, y)
$$

This is a directed acyclic graph: nodes are tensors, edges are operations. Modern frameworks (PyTorch, JAX) build this graph dynamically and use it for automatic differentiation.

For an L-layer MLP:

$$
h^{(0)} = x, \quad z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}, \quad h^{(\ell)} = \sigma(z^{(\ell)}), \quad \hat{y} = h^{(L)}
$$

> **Saying it out loud.** The forward pass is a graph, not a formula, and that framing is what makes autodiff work. Each tensor is a node, each operation an edge, and the framework records the whole thing as you compute — which is why PyTorch can differentiate arbitrary Python control flow. The important consequence is memory: every intermediate value has to be kept alive because the backward pass will need it, so activation memory scales with batch size times sequence length times depth. That's usually what runs you out of GPU, not the parameters.

---

## 5. Backpropagation — derive it

**In plain terms.** Backprop answers one question: if I nudge this weight, how much does the loss change? Doing that naively for a billion weights would take a billion forward passes. The trick is to compute an "error signal" once per layer, going backward, and reuse it — which gets the whole gradient for roughly the cost of one extra forward pass.

Backprop is just the chain rule applied to a computational graph. For one layer:

$$
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} = \frac{\partial \mathcal{L}}{\partial z^{(\ell)}} \cdot \frac{\partial z^{(\ell)}}{\partial W^{(\ell)}} = \delta^{(\ell)} \big(h^{(\ell-1)}\big)^\top
$$

where $\delta^{(\ell)} = \partial \mathcal{L} / \partial z^{(\ell)}$ is the "error signal" at layer $\ell$.

**Recursive formula for $\delta$:**

$$
\delta^{(\ell)} = \big(W^{(\ell+1)}\big)^\top \delta^{(\ell+1)} \odot \sigma'(z^{(\ell)})
$$

Output layer (with cross-entropy + softmax, or BCE + sigmoid, or MSE + identity):

$$
\delta^{(L)} = \hat{y} - y
$$

**The full algorithm:**
1. Forward pass: compute $h^{(\ell)}, z^{(\ell)}$ for $\ell = 1, \ldots, L$, store them.
2. Compute $\delta^{(L)} = \hat{y} - y$.
3. For $\ell = L, L-1, \ldots, 1$:
   - $\nabla_{W^{(\ell)}} \mathcal{L} = \delta^{(\ell)} (h^{(\ell-1)})^\top$
   - $\nabla_{b^{(\ell)}} \mathcal{L} = \delta^{(\ell)}$
   - $\delta^{(\ell-1)} = (W^{(\ell)})^\top \delta^{(\ell)} \odot \sigma'(z^{(\ell-1)})$
4. Update parameters: $W \leftarrow W - \eta \nabla_W \mathcal{L}$.

**Why $O(\text{params})$ time?** Each layer's gradient is one matrix-multiply, the same cost as forward. Total is roughly 2× forward cost — a property called *reverse-mode autodiff*.

**Forward-mode autodiff** computes $J v$ for a fixed $v$ in input dim. Reverse-mode computes $u^\top J$ in output dim. We use reverse because outputs are 1-dim (scalar loss) and inputs are millions (params).

> **Saying it out loud.** Backprop is the chain rule with good bookkeeping. You define an error signal at each layer — how much the loss changes if that layer's pre-activation changes — and then there are exactly two rules. To get a layer's weight gradient, take its error signal and outer-product it with whatever went into that layer. To get the previous layer's error signal, push the current one back through the transposed weight matrix and multiply element-wise by the activation's derivative. At the output, with softmax and cross-entropy, the error signal is just predicted minus true, which is the cleanest starting point you could ask for. The reason it costs about the same as a forward pass is that every step is one matrix multiply, and the reason we use reverse mode rather than forward mode is that we have one scalar output and millions of parameters — reverse mode gets all of them in a single sweep.

---

## 6. Initialization — why it matters and what to use

**In plain terms.** Initialization is about picking the starting scale of the weights so the signal neither fades to nothing nor blows up as it passes through many layers. Each layer multiplies the size of the signal by some factor; you want that factor to be about one. Everything below is the algebra for choosing the weight variance that achieves it.

Bad init kills training. Two failure modes:
- **Vanishing**: activations shrink with depth → gradients vanish → no learning.
- **Exploding**: activations grow with depth → gradients explode → NaN.

**The principle:** preserve variance through the network. If $\text{Var}(h^{(\ell)}) = \text{Var}(h^{(\ell-1)})$, neither happens.

**For a layer with $n_{\text{in}}$ inputs and weights $w_{ij} \sim \mathcal{N}(0, \sigma^2)$:**

$$
\text{Var}(z) = n_{\text{in}} \sigma^2 \text{Var}(x) \implies \sigma^2 = 1/n_{\text{in}}
$$

That's **LeCun init** for tanh / sigmoid / SELU — preserves forward variance.

**Xavier (Glorot)** init balances forward and backward pass:

$$
\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}
$$

For ReLU, half the activations are zero, so we double the variance to compensate:

$$
\sigma^2 = 2/n_{\text{in}}
$$

That's **He (Kaiming)** init. Use this for ReLU/GELU/SiLU MLPs.

**Modern transformer scaling:** GPT-2 uses $\sigma = 0.02$ (a constant, regardless of fan-in) — this works because of LayerNorm, which re-normalizes activations. Plus a $1/\sqrt{2L}$ scaling for residual paths to keep variance growth controlled with depth.

**Empirical takeaway:** in modern architectures (transformers with LayerNorm + residuals), exact init scheme matters less than in old MLPs/CNNs. But it still matters — Megatron-LM, GPT-Neo, Llama all use specific schemes for stability at scale.

> **Saying it out loud.** The whole initialization story is one requirement: keep the variance roughly constant as signal passes through layers. If each layer multiplies variance by a factor slightly below one, after fifty layers you're at essentially zero; slightly above one and you're at NaN. Do the algebra and you get variance equal to one over fan-in — that's LeCun, for tanh-like activations. Xavier compromises between forward and backward flow with two over fan-in plus fan-out. And He doubles it to two over fan-in for ReLU, because ReLU zeroes half your activations and halves the variance on the way through. Modern transformers care less, because LayerNorm renormalizes anyway — GPT-2 just uses a flat 0.02 — but they add a one over root two-L scaling on residual projections so the residual stream doesn't grow as you stack blocks.

---

## 7. Vanishing and exploding gradients — pathologies

**Vanishing:** in deep networks with saturating activations (sigmoid/tanh), the gradient at each layer is $\sigma'(z) < 1$. After $L$ layers, gradient is multiplied by $\prod \sigma'(z^{(\ell)}) \to 0$. Lower layers learn nothing.

**Exploding:** if weights are too large, gradients can grow exponentially with depth. Symptom: NaN loss, training diverges.

**Five (modern) fixes:**
1. **Better activations**: ReLU/GELU don't saturate (gradient = 1 in active region).
2. **Better init**: Kaiming/Xavier preserve variance.
3. **Normalization**: BN/LN/RMSNorm renormalize each layer's activations, keeping gradient flow stable.
4. **Residual connections** (He et al. 2015): $h^{(\ell+1)} = h^{(\ell)} + F(h^{(\ell)})$. Gradient now has an additive identity path: $\partial h^{(\ell+1)}/\partial h^{(\ell)} = I + \partial F / \partial h^{(\ell)}$. The identity term ensures gradient never fully vanishes.
5. **Gradient clipping**: cap $\|\nabla\| \leq \tau$ to prevent explosions. Standard for training transformers and RNNs.

Pre-residual: 8-layer networks were hard. Post-residual: 1000+ layer networks (ResNet-1001) trained successfully.

> **Saying it out loud.** Gradients have to survive a trip through every layer, and each layer multiplies them by something. With sigmoids that something is at most a quarter, so after ten layers you've multiplied by a millionth and the early layers learn nothing — that's vanishing. Too-large weights give you the opposite, exponential growth and a NaN loss. Five things fix it, and modern networks use all of them: non-saturating activations, variance-preserving init, normalization layers, residual connections, and gradient clipping. Residuals are the big one, because adding the input back means the gradient gets an identity term — it's the difference between a product of small numbers and a sum with a one in it. Before residuals, eight layers was hard; after, people trained a thousand.

---

## 8. Training loop — what's actually happening

```python
for epoch in range(num_epochs):
    for x_batch, y_batch in loader:
        # Forward
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        # Backward
        optimizer.zero_grad()       # clear old gradients
        loss.backward()             # compute new gradients via autodiff
        # Optional: clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # Update
        optimizer.step()            # apply update rule
        scheduler.step()            # adjust learning rate
```

What `loss.backward()` does:
1. Walk the computational graph backward from `loss` to leaf tensors (parameters).
2. Apply chain rule using each op's saved backward formula.
3. Accumulate gradients in `param.grad` (note: accumulates, hence `zero_grad`).

Why `zero_grad()`? Gradients accumulate — useful for gradient accumulation across mini-batches when memory-constrained. Forgetting `zero_grad` is a classic bug.

> **Saying it out loud.** The training loop is five steps and the order is the whole game: zero the gradients, forward, compute the loss, backward, step. The one that catches people is zero-grad, because PyTorch *accumulates* into the gradient buffer rather than overwriting it — that's deliberate, since it's what makes gradient accumulation across micro-batches possible, but forget it and each step is contaminated by the last batch. Gradient clipping goes between backward and step, never after; clipping after the optimizer has already moved does nothing. And evaluation goes inside a no-grad block, or you'll build a graph you never free and slowly leak memory.

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Does deeper = always better? | Yes | No — without skip connections, deeper hurts past ~10 layers due to gradient pathologies |
| Why use ReLU over sigmoid? | Faster | Mainly: doesn't saturate → no vanishing gradient |
| What does `loss.backward()` do? | Computes gradients | Walks comp graph backward via chain rule, *accumulates* into `.grad` |
| Why does normalization help? | Faster training | Stabilizes activation/gradient magnitudes; allows higher LR |
| What does residual fix? | Optimizer | Gradient flow: identity path means gradient never fully vanishes |
| Why does dropout work? | Reduces overfitting | Forces redundancy; ensemble interpretation; ~$1/p$ implicit regularization |
| Best init for ReLU? | Xavier | He (Kaiming): $\sigma^2 = 2/n_{\text{in}}$ |

> **Saying it out loud.** Most of these gotchas share one root: naming the mechanism instead of the symptom. Deeper isn't automatically better — past about ten layers without skip connections it's actively worse, because of gradient flow, not because of capacity. ReLU beats sigmoid because it doesn't saturate, not because it's faster to compute, though it is. Normalization helps because it stabilizes activation and gradient magnitudes, which is what lets you raise the learning rate. And He init, not Xavier, is the answer for ReLU — two over fan-in — because ReLU throws away half the variance. In each case the shallow answer isn't wrong, it's just one level too high, and the interviewer is listening for the level below it.

---

## 10. Beyond MLPs — what to know about modern variants

**CNNs**: weight-sharing across spatial positions. Same backprop math, but convolutions instead of dense matmuls. Inductive bias: locality and translation equivariance.

**RNNs**: parameter-sharing across time. BPTT (backprop through time) unrolls the network and applies standard backprop. Suffers worst from vanishing gradients (long sequences). LSTMs/GRUs use gating to mitigate.

**Transformers**: stack of self-attention + MLP blocks with residual connections and LayerNorm. The MLP/FFN block is just a 2-layer MLP — everything in this deep dive applies. The attention block is a parameter-shared linear projection followed by a soft-mixing operation.

**Common pattern**: pre-LN block (LayerNorm before sub-layer, used in modern LLMs) is more stable than post-LN (original transformer paper). Pre-LN: $h \to h + F(\text{LN}(h))$. Post-LN: $h \to \text{LN}(h + F(h))$.

> **Saying it out loud.** Every modern architecture is the same backprop with a different weight-sharing pattern. CNNs share one kernel across all spatial positions, which encodes locality and translation equivariance and cuts the parameter count enormously. RNNs share one matrix across time, which is why they suffer worst from vanishing gradients — you're effectively raising a matrix to the power of the sequence length. Transformers drop recurrence entirely: attention mixes information between tokens, and a plain two-layer MLP does the computing within each token, all wrapped in residuals and LayerNorm. The one detail worth volunteering is pre-LN versus post-LN — putting the norm inside the residual branch keeps the identity path clean and is why modern LLMs train deep without heroic warmup schedules.

---

## 11. Eight most-asked interview questions

1. **Derive backpropagation for a 2-layer MLP from scratch.** (Lock down chain rule.)

   > **Saying it out loud.** Set it up as two matrix multiplies with an activation between, then softmax and cross-entropy on top. Going backward, the first step is the gift: the error at the output is just predicted minus true, because the softmax and cross-entropy derivatives cancel. From there, the second weight matrix's gradient is that error outer-producted with the hidden activations. Then push the error back through W2 transposed, multiply element-wise by the activation's derivative, and you have the hidden error — and W1's gradient is that outer-producted with the input. The pattern to say out loud is: every weight gradient is an error signal times the input to that layer.

2. **What is the dying ReLU problem and how do you fix it?** (Leaky ReLU, GELU, init, lower LR.)

   > **Saying it out loud.** A dead ReLU is a unit whose pre-activation is negative for every training example. Since ReLU's gradient is exactly zero on the negative side, it gets no gradient, never updates, and is dead permanently — it's not a slow learner, it's gone. The usual cause is a learning rate large enough to blast the weights or bias far negative in a single step. Fixes come in two flavors: change the activation so there's always some gradient — Leaky ReLU with a 0.01 slope, or GELU and SiLU which are smooth everywhere — or fix the optimization with He init and a lower learning rate. With a bad configuration you can lose ten to forty percent of your units, which looks like a model that trains but plateaus too high.

3. **Why does He initialization use $2/n$ and Xavier use $1/n$?** (ReLU drops half the activations.)

   > **Saying it out loud.** Both come from the same variance-preservation argument; they differ by a factor of two because of what ReLU does. The requirement is that the output variance of a layer equals its input variance, which gives you weight variance equal to one over fan-in. But ReLU zeroes out every negative pre-activation, which cuts the variance in half — so you double the initialization variance to compensate, giving two over fan-in. Xavier was derived for tanh, which is roughly linear and symmetric near zero and doesn't discard anything. Use Xavier with tanh, He with the ReLU family; get it backwards on a fifty-layer network and your signal decays by a factor of a million.

4. **What problem do residual connections solve?** (Vanishing gradients in deep networks; identity path in gradient.)

   > **Saying it out loud.** Residuals solve gradient flow. In a plain deep network the gradient is a long product of per-layer Jacobians, and if those are consistently a bit less than one, it vanishes before reaching the early layers. A residual connection adds the input back to the block's output, so when you differentiate you get identity *plus* the block's Jacobian — the identity term guarantees a direct path for the gradient no matter what the block does. The other way to say it is that each block only has to learn a correction, so a useless block is harmless rather than destructive. That's what took networks from about eight trainable layers to over a thousand.

5. **Why is sigmoid bad in hidden layers?** (Saturation → vanishing gradients; not zero-centered.)

   > **Saying it out loud.** Two problems, and the saturation one is the killer. Sigmoid's derivative peaks at 0.25 and collapses toward zero once the input is past about five in absolute value, so a ten-layer sigmoid network multiplies the gradient by at most one in a million before it reaches layer one — the early layers effectively never train. Second, sigmoid isn't zero-centered: all outputs are positive, so all the weight gradients into a given neuron share a sign and the optimizer zigzags instead of going straight. It's still the right choice at the output for binary classification, where you want a probability; it just doesn't belong in the middle.

6. **Compare forward-mode and reverse-mode autodiff.** (Reverse is efficient when outputs ≪ inputs.)

   > **Saying it out loud.** They're the same chain rule, traversed in opposite directions, and which one is efficient depends on the shape. Forward mode costs one pass per input; reverse mode costs one pass per output. Deep learning has millions of inputs — the parameters — and exactly one output, the scalar loss, so reverse mode gets the entire gradient in a single backward sweep while forward mode would need millions. Flip the shape — few inputs, many outputs — and forward mode wins, which is why it's used for Jacobian-vector products and sensitivity analysis. The cost of reverse mode is memory: it has to store every intermediate activation until the backward pass consumes it.

7. **Why pair softmax with cross-entropy?** (Gradient simplifies to $\hat{y} - y$; canonical link of multinomial GLM.)

   > **Saying it out loud.** Because the math cancels. Cross-entropy is the negative log-likelihood of a categorical distribution, and softmax is its canonical link, so when you differentiate the composition, everything collapses to predicted minus actual. There's no leftover derivative-of-the-activation factor to shrink the gradient — which means when the model is confidently wrong, the gradient is large, exactly as it should be. Pair them incorrectly, say squared error on a softmax, and you reintroduce that saturating factor and the worst examples produce the weakest updates. That's also why frameworks fuse the two into one op, for numerical stability with large logits.

8. **What is a universal approximator and what's the catch?** (One hidden layer can approximate anything; but width may be exponential — depth is more efficient.)

   > **Saying it out loud.** A universal approximator is a model class that can get arbitrarily close to any continuous function on a bounded domain, and a one-hidden-layer network with a non-polynomial activation qualifies. The catch is that it's an existence result and nothing more. It doesn't say how wide — the width can be exponential in the input dimension — and it doesn't say gradient descent will ever find those weights. That's why the theorem, despite sounding decisive, doesn't tell you to build shallow networks: depth achieves the same expressiveness with polynomially many parameters for compositional functions, and it's easier to optimize.


---

## 12. Drill plan

- Hand-derive backprop for a 2-layer MLP with ReLU + softmax + CE on paper. Repeat until 5 minutes.
- Implement an MLP from scratch in NumPy (no autodiff) — forward, backward, train on MNIST. Verify gradients with finite differences.
- For each activation in the table, recite: formula, derivative, range, when to use, failure mode.
- For each init scheme (Xavier, He, LeCun, GPT-2 0.02), recite: variance preservation argument and which activation it pairs with.
- Be able to draw the gradient flow through a residual block and explain why depth becomes trainable.

---

## 13. Further reading

- Rumelhart, Hinton, Williams (1986). *Learning representations by back-propagating errors.* — Original backprop paper.
- Glorot & Bengio (2010). *Understanding the difficulty of training deep feedforward neural networks.* — Xavier init.
- He et al. (2015). *Delving deep into rectifiers.* — He init.
- He et al. (2015). *Deep residual learning for image recognition.* — ResNets, residual connections.
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units.* — GELU.
- Shazeer (2020). *GLU Variants Improve Transformer.* — SwiGLU motivation.
- Goodfellow, Bengio, Courville. *Deep Learning* — chapters 6, 8 (optimization), 11 (practical).
