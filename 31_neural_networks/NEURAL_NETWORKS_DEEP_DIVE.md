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

---

## 3. Loss functions — pair with output activation

| Task | Output activation | Loss | Why |
|---|---|---|---|
| Regression | Identity | MSE: $\frac{1}{2}\|y - \hat{y}\|^2$ | Gaussian likelihood |
| Binary classification | Sigmoid | BCE: $-y \log p - (1-y) \log(1-p)$ | Bernoulli MLE |
| Multi-class | Softmax | Cross-entropy | Categorical MLE |
| Multi-label | Sigmoid (per class) | Sum of BCE | Independent Bernoullis |

The activation–loss pairings aren't accidents. They're the canonical link function for the corresponding GLM (sigmoid+BCE = logistic regression; softmax+CE = multinomial logistic regression). They make the gradient simple: $\nabla_z \mathcal{L} = \hat{y} - y$ in all three classification cases. Mismatched pairings (e.g., MSE on softmax outputs) cause flat loss surfaces and slow training.

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

---

## 5. Backpropagation — derive it

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

---

## 6. Initialization — why it matters and what to use

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

---

## 10. Beyond MLPs — what to know about modern variants

**CNNs**: weight-sharing across spatial positions. Same backprop math, but convolutions instead of dense matmuls. Inductive bias: locality and translation equivariance.

**RNNs**: parameter-sharing across time. BPTT (backprop through time) unrolls the network and applies standard backprop. Suffers worst from vanishing gradients (long sequences). LSTMs/GRUs use gating to mitigate.

**Transformers**: stack of self-attention + MLP blocks with residual connections and LayerNorm. The MLP/FFN block is just a 2-layer MLP — everything in this deep dive applies. The attention block is a parameter-shared linear projection followed by a soft-mixing operation.

**Common pattern**: pre-LN block (LayerNorm before sub-layer, used in modern LLMs) is more stable than post-LN (original transformer paper). Pre-LN: $h \to h + F(\text{LN}(h))$. Post-LN: $h \to \text{LN}(h + F(h))$.

---

## 11. Eight most-asked interview questions

1. **Derive backpropagation for a 2-layer MLP from scratch.** (Lock down chain rule.)
2. **What is the dying ReLU problem and how do you fix it?** (Leaky ReLU, GELU, init, lower LR.)
3. **Why does He initialization use $2/n$ and Xavier use $1/n$?** (ReLU drops half the activations.)
4. **What problem do residual connections solve?** (Vanishing gradients in deep networks; identity path in gradient.)
5. **Why is sigmoid bad in hidden layers?** (Saturation → vanishing gradients; not zero-centered.)
6. **Compare forward-mode and reverse-mode autodiff.** (Reverse is efficient when outputs ≪ inputs.)
7. **Why pair softmax with cross-entropy?** (Gradient simplifies to $\hat{y} - y$; canonical link of multinomial GLM.)
8. **What is a universal approximator and what's the catch?** (One hidden layer can approximate anything; but width may be exponential — depth is more efficient.)

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
