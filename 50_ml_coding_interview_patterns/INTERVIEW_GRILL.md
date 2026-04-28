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

---

## I. Production patterns

**39. Why use mixed precision?**
2× memory savings + faster matmul (BF16/FP16). Master weights in FP32 for stability.

**40. Why use gradient accumulation?**
Effective batch size = micro_batch × accum_steps. Use when memory limits batch size.

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
