# Neural Networks Fundamentals — Interview Grill

> 50 questions on MLPs, activations, init, backprop, gradient pathologies. Drill until you can answer 35+ cold.

---

## A. MLP basics

**1. What does a single layer of an MLP compute?**
$h = \sigma(W x + b)$. Affine transform followed by element-wise non-linearity.

**2. Why does an MLP need non-linearity?**
Without it, stacked layers collapse to a single affine map $W_L \cdots W_1 x + c$. No expressive gain from depth.

**3. State the universal approximation theorem.**
A one-hidden-layer MLP with non-polynomial activation can approximate any continuous function on a compact set to arbitrary precision (Cybenko 1989, Hornik 1991).

**4. If one hidden layer is enough, why use depth?**
Width may need to be exponential. Depth is more parameter-efficient for hierarchical/compositional functions (Telgarsky 2016). Also, depth induces useful inductive biases.

**5. Why is depth alone hard to train?**
Vanishing/exploding gradients. Without residual connections + normalization, networks past ~10 layers struggle.

---

## B. Activations

**6. Why is sigmoid problematic in hidden layers?**
Two reasons. (1) Saturates — gradient $\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25$ everywhere, vanishes for $|x| > 5$. (2) Not zero-centered — outputs in $(0,1)$ cause weight gradients to all share sign.

**7. Why is ReLU the default?**
Cheap (just a max), non-saturating for $x > 0$ (gradient = 1 → no decay), induces sparsity (~50% activations zero). Solved the vanishing gradient problem for deep nets.

**8. What's the dying ReLU problem?**
If a neuron's pre-activation is negative for all training data, $\text{ReLU}'(x) = 0$ → no gradient → neuron is dead permanently. Caused by large negative bias or large LR pushing weights into dead region.

**9. How do you fix dying ReLU?**
Leaky ReLU ($\max(\alpha x, x)$), GELU/SiLU (smooth, non-zero gradient everywhere), better initialization, lower learning rate.

**10. What's GELU?**
$x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. Smooth, has stochastic regularizer interpretation. Standard in BERT, GPT-2, GPT-3.

**11. What's SwiGLU and why is it now standard in LLMs?**
$\text{SwiGLU}(x) = \text{Swish}(xW_g) \odot (xW)$. Two parallel projections with element-wise gating. Empirically beats vanilla FFN consistently (Shazeer 2020). Used in Llama, PaLM, Mistral.

**12. ReLU vs GELU empirically?**
GELU slightly better for transformers; ReLU still competitive and cheaper. SiLU/Swish often preferred over GELU in newer LLMs.

---

## C. Loss functions

**13. Why pair softmax with cross-entropy?**
Cross-entropy is the negative log-likelihood under a multinomial. Gradient simplifies to $\hat{y} - y$ — clean and easy. Canonical link function of the multinomial GLM.

**14. Why pair sigmoid with BCE?**
Same reason — gradient is $\hat{y} - y$. Mismatched pairings (e.g., MSE on sigmoid output) give vanishing gradients near saturation.

**15. When is MSE the right loss?**
Continuous regression with Gaussian noise assumption. Identity output activation. NOT for classification (gradient vanishes near saturation).

**16. Multi-class vs multi-label classification?**
Multi-class: one true class, softmax + CE. Multi-label: multiple true classes, sigmoid (per class) + sum of BCE.

---

## D. Backpropagation

**17. What is backpropagation?**
Reverse-mode automatic differentiation: chain rule applied backward through a computational graph to compute gradient of scalar loss w.r.t. all parameters in $O(\text{forward})$ time.

**18. Walk through backprop for a 2-layer MLP.**
Forward: $z_1 = W_1 x$, $h_1 = \sigma(z_1)$, $z_2 = W_2 h_1$, $\hat{y} = \mathrm{softmax}(z_2)$. Loss: $\mathcal{L} = -\sum y \log \hat{y}$.

Backward:
- $\delta_2 = \hat{y} - y$
- $\nabla_{W_2} \mathcal{L} = \delta_2 h_1^\top$
- $\delta_1 = W_2^\top \delta_2 \odot \sigma'(z_1)$
- $\nabla_{W_1} \mathcal{L} = \delta_1 x^\top$

**19. Why is reverse-mode used for ML?**
Loss is scalar (1 output), parameters are millions. Reverse-mode costs $O(\text{outputs}) = O(1)$ passes. Forward-mode costs $O(\text{inputs}) = O(\text{millions})$ — infeasible.

**20. When would forward-mode be preferred?**
When inputs are few and outputs are many (e.g., computing a Jacobian-vector product, sensitivity analysis with few parameters).

**21. What does `loss.backward()` actually do?**
Walks the computation graph backward from `loss` to leaf tensors, applies stored backward formulas via chain rule, *accumulates* gradients into `param.grad`. (Hence the need for `zero_grad`.)

**22. Why call `optimizer.zero_grad()` before backward?**
Gradients accumulate in `.grad` — useful for gradient accumulation across mini-batches, but if you forget to zero, gradients from previous batches contaminate current ones. Classic bug.

**23. What are activations stored for during forward pass?**
Backward pass needs them — gradient w.r.t. weights involves the input to that layer. Without storing, you'd have to recompute (gradient checkpointing trades memory for compute by doing exactly this).

---

## E. Initialization

**24. What's the goal of weight initialization?**
Preserve activation variance (and gradient variance) across layers — prevent vanishing or exploding signals.

**25. Derive LeCun and Xavier (Glorot) init.**
For $z = Wx$ with $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ and $x_i$ iid with variance $v$: $\text{Var}(z_j) = n_{\text{in}} \sigma^2 v$. To preserve forward variance: $\sigma^2 = 1/n_{\text{in}}$ — that's **LeCun init** (for tanh/sigmoid/SELU). To preserve *both* forward and backward variance: $\sigma^2 = 2/(n_{\text{in}} + n_{\text{out}})$ — that's **Xavier (Glorot) init**.

**26. Why does He init differ from Xavier?**
ReLU zeros out half the activations, halving the variance contribution. Compensate: $\sigma^2 = 2/n_{\text{in}}$. Xavier was derived for tanh, where this issue doesn't apply.

**27. What init does GPT-2 use?**
$\sigma = 0.02$ (fixed, not depending on fan-in), plus a $1/\sqrt{2L}$ scaling on residual outputs. Works because LayerNorm renormalizes activations regardless.

**28. What happens with all-zero init?**
All neurons compute the same thing → identical gradients → never break symmetry. Network never learns. Bias to zero is fine; weights need random init.

**29. What happens if weights are too large?**
Activations explode, gradients explode, NaN. Especially with deep networks — $\text{Var}(h^{(L)}) = (\sigma^2 n)^L$ blows up if $\sigma^2 n > 1$.

---

## F. Vanishing and exploding gradients

**30. Why do gradients vanish in deep sigmoid networks?**
Each layer multiplies gradient by $\sigma'(z) \leq 0.25$. After $L$ layers: gradient scaled by $\leq 0.25^L \to 0$.

**31. Five fixes for vanishing gradients?**
(1) Non-saturating activations (ReLU/GELU). (2) Better init (He/Xavier). (3) Normalization (BN/LN/RMSNorm). (4) Residual connections. (5) Architectures designed for long-range gradient flow (LSTM gates, transformer attention).

**32. How do residual connections help?**
$h^{(\ell+1)} = h^{(\ell)} + F(h^{(\ell)})$. Gradient: $\partial h^{(\ell+1)} / \partial h^{(\ell)} = I + \partial F / \partial h^{(\ell)}$. Identity term ensures gradient never fully vanishes — there's always a direct gradient path.

**33. What's gradient clipping?**
Cap $\|\nabla\| \leq \tau$ (clip by norm) or $|\nabla_i| \leq \tau$ (clip by value). Prevents loss spikes from exploding gradients. Standard for transformers (typically $\tau = 1.0$).

**34. Why are RNNs especially prone to vanishing gradients?**
BPTT unrolls a single weight matrix $W$ across $T$ time steps. Gradient is $W^T \cdot \prod \sigma'$. If $\|W\| < 1$, gradient vanishes; if $\|W\| > 1$, explodes. LSTMs use gates to maintain a roughly identity state path (similar idea to residual).

**35. Pre-LN vs Post-LN — which is more stable?**
Pre-LN ($h + F(\text{LN}(h))$) — standard in modern LLMs. Gradient flows through the residual path without going through LN first, which keeps it well-scaled. Post-LN (original transformer) is harder to train deep (requires careful warmup).

---

## G. Training loop

**36. What's a typical PyTorch training loop?**
For each batch: zero gradients, forward, compute loss, backward, optionally clip gradients, optimizer step, scheduler step.

**37. What's gradient checkpointing?**
Trade memory for compute — don't store activations during forward pass; recompute them during backward. Used to fit large models in memory at the cost of ~33% slowdown.

**38. What's gradient accumulation?**
Run forward+backward on multiple micro-batches without optimizer.step(), then step. Effective batch size = micro-batch × accumulation steps. Used when memory limits batch size.

**39. What does mixed-precision training do?**
Forward/backward in FP16 or BF16, weights and optimizer state in FP32. Faster, less memory. BF16 is preferred over FP16 for stability (no dynamic loss scaling needed).

**40. What's a learning rate scheduler typically doing?**
Warmup (linearly increase LR from 0) + decay (cosine, linear, or constant). Warmup prevents early instability; decay refines at the end. LLMs typically use cosine decay to ~10% of peak.

---

## H. Modern architectures

**41. How is a CNN different from an MLP for backprop?**
Same chain-rule math, but convolution instead of matmul → weight sharing across spatial positions. Backprop convolution is convolution with flipped kernel.

**42. What's a transformer FFN block?**
A 2-layer MLP applied position-wise: $\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$. Hidden dim typically 4× model dim. Modern variants use SwiGLU.

**43. Why do CNNs typically use BN and transformers use LN?**
BN normalizes across batch — works well with large batches and image data (translation-invariant statistics). LN normalizes across features per-token — independent of batch size, works for variable sequence lengths, more stable for transformers.

**44. What's the role of dropout in modern transformers?**
Less critical than in early MLPs. Used in BERT-style training (10–20%); often removed or reduced in large LLMs that have implicit regularization from massive data + weight decay.

**45. What's weight decay actually doing?**
$\ell_2$ penalty on parameters: $\mathcal{L} + \lambda \|w\|^2$. Pulls weights toward zero. In AdamW, decoupled from gradient (correct implementation); in vanilla Adam, it's coupled and weaker than intended.

---

## I. Subtleties

**46. Can the loss go up during training?**
Yes — with momentum-based optimizers, large LR, or when the LR scheduler resets. Long-term trend should be down. Short-term noise is normal.

**47. What does it mean if training loss plateaus at a non-zero value?**
Model has reached a local minimum or saddle point given current capacity/data/optimizer. Possible fixes: more capacity, better optimizer, lower LR, data augmentation, regularization.

**48. Why might validation loss go up while training loss goes down?**
Overfitting. Model memorizes training data. Fixes: regularization, early stopping, more data, smaller model.

**49. What's catastrophic forgetting?**
Sequential training on task A then B → model forgets A. Common in RL, transfer learning, continual learning. Fixes: replay, EWC (elastic weight consolidation), PEFT (LoRA).

**50. Lottery ticket hypothesis?**
Frankle & Carbin (2018): dense networks contain sparse subnetworks ("winning tickets") that, trained from scratch with the same init, match the dense network's performance. Suggests over-parameterization is mostly about init/optimization landscape.

---

## Quick fire

**51.** *Best init for ReLU?* He: $\sigma^2 = 2/n_{\text{in}}$.
**52.** *Best init for tanh?* LeCun: $\sigma^2 = 1/n_{\text{in}}$. (Xavier/Glorot is $2/(n_{\text{in}} + n_{\text{out}})$ — balances forward + backward; commonly used for tanh too.)
**53.** *Output activation for binary classification?* Sigmoid.
**54.** *Output activation for multi-class?* Softmax.
**55.** *Backprop time complexity?* $O(\text{forward})$ — about 2× forward.
**56.** *Why does sigmoid vanish?* $\sigma' \leq 0.25$, multiplies through depth.
**57.** *What does residual fix?* Vanishing gradients (identity path).
**58.** *Why pre-LN over post-LN?* Cleaner gradient flow through residual.
**59.** *FFN hidden dim ratio in transformers?* Typically $4d_{\text{model}}$.
**60.** *Standard gradient clip value?* 1.0 (clip by global norm).

---

## Self-grading

If you can't answer 1-15, you don't know neural networks. If you can't answer 16-35, you can't pass a deep-learning interview screen. If you can't answer 36-50, frontier-lab applied scientist interviews on training large models will go past you.

Aim for 40+/60 cold.
