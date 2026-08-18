# ML Coding Patterns — Interview Grill

> 40 questions on stable softmax, attention, sampling, and other coding patterns. Drill until you can answer / code 28+ cold.

---

## A. Numerical stability

**1. Why does naive softmax overflow?**
$e^{x_i}$ for large $x_i$ → infinity. Need to subtract max.

**2. The stable softmax formula?**
$e^{x_i - m} / \sum_j e^{x_j - m}$ where $m = \max x_j$.

**3. Why is subtracting max safe?**
Cancels in numerator and denominator → mathematically identical.

**4. Log-sum-exp formula?**
$\log\sum e^{x_j} = m + \log\sum e^{x_j - m}$.

**5. Cross-entropy + softmax in one step — why?**
Combine as logits $-$ logsumexp; avoids underflow on softmax output.

**6. PyTorch's `nn.CrossEntropyLoss` takes what input?**
Raw logits, not probabilities. Combines log-softmax + NLL internally.

> **Saying it out loud.** Everything in this group comes from one fact: `exp` overflows fast, and floating point has no mercy. So you never exponentiate a raw logit — you subtract the row max first, which changes nothing mathematically because the constant cancels top and bottom, and guarantees the largest exponent is zero. Log-sum-exp is the same move for log-space: pull the max out front, exponentiate the rest, add the max back. And you never chain softmax into log, you fuse them, which is why `nn.CrossEntropyLoss` takes raw logits. Failure mode to name: overflow to NaN, or an underflowed probability whose log is minus infinity.

---

## B. Attention

**7. Scaled dot-product attention scaling factor?**
$1/\sqrt{d_k}$. Prevents softmax saturation for large $d_k$.

**8. Why divide by $\sqrt{d_k}$ specifically?**
Variance of $Q K^\top$ scales with $d_k$ if Q, K have unit-variance entries. Divide to maintain unit variance → prevents softmax saturation.

**9. How is causal mask applied?**
Set masked positions to $-\infty$ before softmax. (NOT multiply by 0 after.)

**10. Why $-\infty$ before softmax?**
$\exp(-\infty) = 0$ → masked positions contribute 0 to numerator and denominator after softmax.

**11. Multi-head reshape order?**
$[B, L, D] \to [B, L, H, D/H] \to [B, H, L, D/H]$ — heads come before sequence.

**12. Common attention bug?**
Wrong axis for softmax. Should be over last dim (key dim).

> **Saying it out loud.** Attention scores every query against every key, softmaxes those scores, and returns the weighted average of the values — a soft lookup. The $1/\sqrt{d_k}$ is there because the variance of a dot product grows with the dimension, so unscaled scores get huge, the softmax saturates into a one-hot, and gradients die. Masking is additive and happens *before* the softmax: you push forbidden scores to $-\infty$ so they come out as exactly zero weight and the rest still renormalize. Multiply-by-zero afterwards leaves the denominator wrong. The two bugs interviewers watch for are softmaxing over the wrong axis and a multi-head reshape that mixes heads with positions.

---

## C. Sampling

**13. Greedy = top-k with k = ?**
1.

**14. Top-p strategy?**
Sort logits, take cumulative softmax, keep smallest set with cumprob $\geq p$, sample.

**15. Top-p with $p = 1$?**
Includes all tokens — equivalent to full sampling.

**16. Temperature does what to logits?**
Divides by $T$. $T \to 0$: greedy. $T \to \infty$: uniform.

**17. Frequency / presence penalty?**
Subtract from logits of recently-used tokens. Reduces repetition.

**18. Why does beam search produce repetitive text?**
Maximum-likelihood paths cluster in low-entropy regions. Sampling with top-p avoids this.

> **Saying it out loud.** Decoding is all about how much of the tail you let in. Temperature rescales the logits — toward zero it becomes greedy, toward infinity it becomes uniform. Top-k keeps a fixed count of candidates; top-p keeps the smallest set whose mass reaches $p$, so it widens when the model is unsure and narrows when it's confident, which is why top-p is the modern default. Repetition penalties just subtract from the logits of tokens you've already used. The tradeoff is coherence versus diversity, and the named failure is degenerate repetition when you decode too greedily.

---

## D. Beam search

**19. Beam search update?**
Maintain top-$B$ hypotheses; expand each by all next tokens; keep top-$B$ overall.

**20. Length normalization?**
Divide log-prob by $\mathrm{len}^\alpha$ (typically $\alpha = 0.6$ for translation).

**21. When to use beam search?**
Tasks with single-correct-answer flavor: translation, summarization with reference. Not for open-ended generation.

**22. Beam size trade-off?**
Larger: better likelihood but slower; sometimes worse output quality (repetition).

> **Saying it out loud.** Beam search keeps the $B$ best partial sequences at each step rather than committing to one, so it's a wider greedy search, not a real search of the whole space. Length normalization is mandatory because each extra token subtracts more log-probability — divide by length to the power $\alpha$, roughly 0.6, or the model always ends the sentence early. Bigger beams give better likelihood and cost linearly more compute. The tradeoff worth stating: for translation or summarization, where there's roughly one right answer, beam wins; for open-ended generation the highest-likelihood text is famously bland and repetitive, so you sample instead.

---

## E. K-means and clustering

**23. K-means update step?**
Assign each point to nearest centroid; update each centroid to mean of assigned.

**24. K-means complexity per iteration?**
$O(NKD)$ where $N$ data, $K$ clusters, $D$ dim.

**25. Empty cluster handling?**
Re-initialize centroid (random point, far from existing centroids, or split largest cluster).

**26. K-means++ initialization?**
First centroid random; subsequent centroids sampled with prob $\propto$ squared distance to nearest existing centroid. $O(\log K)$-approximation guarantee.

> **Saying it out loud.** K-means alternates assign-to-nearest and move-to-mean, and it's really coordinate descent on total within-cluster squared distance, so it converges monotonically but only to a local optimum. Each iteration costs $O(NKD)$, which is why it scales fine until $K$ gets big. Initialization is the whole ballgame: K-means++ picks seeds with probability proportional to squared distance from the nearest existing centroid and buys you an $O(\log K)$ approximation guarantee. The failure mode to name is the empty cluster — mean of nothing is NaN, so you re-seed it — plus the standing assumption of roughly spherical, equally sized clusters.

---

## F. Backprop and MLP

**27. Cross-entropy + softmax gradient?**
$dz = (p - y)/n$. Beautifully simple.

**28. ReLU derivative?**
1 if $z > 0$, else 0. (Zero at $z = 0$ technically but doesn't matter in practice.)

**29. Backprop for $h_2 = W_2 \sigma(W_1 x)$?**
$dW_2 = (h_1)^\top dz_2$, $dz_1 = (W_2^\top dz_2) \odot \sigma'(z_1)$, $dW_1 = x^\top dz_1$.

**30. Why store activations during forward?**
Backward needs them — gradient w.r.t. weights uses input to that layer.

> **Saying it out loud.** Backprop is the chain rule run backwards with caching: each layer receives the gradient of the loss with respect to its output and produces two things, a gradient for its own weights and one to hand to the layer below. You store the forward activations precisely because the weight gradient is the layer's input times the incoming gradient. The identity worth memorizing is that softmax plus cross-entropy gives $(p - y)/n$ — everything else cancels. ReLU's derivative is just an indicator of positive pre-activation. Named tradeoff: activation checkpointing throws those cached activations away to save memory and recomputes them, usually costing about 30 percent extra compute.

---

## G. Padding and masking

**31. Why pad sequences?**
Variable lengths can't be batched without padding to common length.

**32. Padding mask shape?**
$[B, L]$, 1 for valid, 0 for padding.

**33. Combined causal + padding mask?**
Lower-triangular AND padding-mask broadcasted. Bool AND.

**34. Common bug with padding?**
Forgetting to mask padding from loss (you're predicting on tokens that don't exist).

> **Saying it out loud.** Padding exists so variable-length sequences fit in one rectangular tensor, and masking exists so those fake positions never affect anything. The padding mask is $[B, L]$, one for real tokens, and you combine it with the lower-triangular causal mask by a boolean AND. Two places people forget it: in attention, where pad tokens otherwise soak up weight and make a sentence's embedding depend on its batchmates, and in the loss, where you'd be training the model to predict padding. The failure mode is eval numbers that shift when you change batch size — a real bug that looks like noise.

---

## H. Vectorization

**35. Vectorized cosine similarity?**
Normalize each row of Q and K independently; matmul $QK^\top$. Result is $[|Q|, |K|]$.

**36. Pairwise distance matrix?**
$\|x_i - x_j\|^2 = \|x_i\|^2 + \|x_j\|^2 - 2 x_i^\top x_j$. Compute via norm² + matmul.

**37. Why avoid Python loops?**
GIL + interpreter overhead. Vectorized NumPy/PyTorch is 10–100× faster.

**38. Broadcasting trick?**
Use `[:, None]` and `[None, :]` to get pairwise computation without explicit loop.

> **Saying it out loud.** Vectorizing is about replacing a Python loop with one big array operation, because the loop pays interpreter overhead per element while the matmul runs in optimized C or on the GPU — routinely 10 to 100 times faster. Cosine similarity is row-normalize both matrices then one matmul. Pairwise squared distance uses the expansion into norms plus a matmul rather than materializing every difference. Broadcasting with `[:, None]` and `[None, :]` is the general trick. The tradeoff is memory: the broadcast difference tensor for $N$ points in $D$ dimensions is $N^2 D$, so past a few thousand points you chunk it or you get an out-of-memory error.

---

## I. Production patterns

**39. Why use mixed precision?**
2× memory savings + faster matmul (BF16/FP16). Master weights in FP32 for stability.

**40. Why use gradient accumulation?**
Effective batch size = micro_batch × accum_steps. Use when memory limits batch size.

> **Saying it out loud.** These two are the standard answers to "my model doesn't fit." Mixed precision stores activations and does matmuls in BF16 or FP16, which roughly halves memory and lets you use tensor cores, while keeping a master copy of the weights in FP32 so tiny updates don't get rounded away. Gradient accumulation fakes a larger batch by summing gradients over several micro-batches before stepping, so effective batch equals micro-batch times accumulation steps. The tradeoff is time: accumulation buys batch size at a proportional cost in wall clock. FP16's named failure mode is gradient underflow, which is why you need loss scaling; BF16 has the range to avoid it and is the usual choice today.

---

## Quick fire

**41.** *Stable softmax — subtract?* Max.
**42.** *Attention scale?* $1/\sqrt{d_k}$.
**43.** *Mask method?* $-\infty$ before softmax.
**44.** *Top-p set?* Smallest summing to $\geq p$.
**45.** *Cross-entropy gradient?* $(p - y)/n$.
**46.** *ReLU derivative at 0?* 0 (or 1, conventionally).
**47.** *Beam size standard?* 5.
**48.** *K-means complexity?* $O(NKD)$.
**49.** *Padding mask shape?* $[B, L]$.
**50.** *Cross-entropy combines?* Log-softmax + NLL.

---

## Self-grading

If you can't code 8+ of the 8 main patterns from scratch in 10 min each, drill more. If you don't know the gotchas (top-p set semantics, mask via $-\infty$, log-sum-exp), interviewers will catch them.

Aim for 30+/50 cold + ability to code each top-8 pattern in $\leq$ 10 min.
