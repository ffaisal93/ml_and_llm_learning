# Positional Embeddings — Interview Grill

> 40 questions on positional encoding. Drill until you can answer 30+ cold.

---

## A. Foundations

**1. Why do transformers need positional encoding?**
Pure attention is permutation-equivariant: shuffle tokens and the output shuffles the same way. Attention has no innate notion of order. Positional encoding is the only mechanism by which transformers know what comes first.

**2. State permutation equivariance formally.**
For any permutation matrix $P$: $\mathrm{Attention}(P \cdot X) = P \cdot \mathrm{Attention}(X)$. This means the attention output depends only on the multiset of input tokens, not their order. Adding positional information breaks this.

**3. What are the main families of positional encoding?**
Absolute (sinusoidal, learned), relative (T5 bias, Transformer-XL), rotary (RoPE), bias-based (ALiBi), and none (NoPE).

---

## B. Sinusoidal

**4. Walk me through sinusoidal positional encoding.**
For each position $t$ and dimension $2i$ (even) or $2i+1$ (odd):

$$
\mathrm{PE}(t, 2i) = \sin(t / 10000^{2i/d}), \qquad \mathrm{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})
$$

Add $\mathrm{PE}(t)$ to the token embedding. Different dimensions oscillate at exponentially different frequencies, giving each position a unique signature.

**5. Why exponentially-spaced frequencies?**
Spans many orders of magnitude (low frequencies for global structure, high frequencies for local). The base 10000 is empirical; not deeply principled. Could be 1000 or 100000 with similar results.

**6. Why does sinusoidal in theory enable extrapolation?**
The encoding is defined for any position $t$, including beyond training length. Plus the theoretical property: for any $\Delta t$, there exists a fixed linear transform $M_{\Delta t}$ such that $\mathrm{PE}(t + \Delta t) = M_{\Delta t} \cdot \mathrm{PE}(t)$, so relative positions can be computed by linear operations on absolute encodings.

**7. Why does sinusoidal extrapolation fail in practice?**
The encoding is well-defined at long range, but the **learned weights** that work with it are trained only on positions seen in training. The model's attention patterns at position 5000 (when trained at 1024) are unreliable.

**8. What replaced sinusoidal?**
Learned positional embeddings (BERT, GPT-2/3) for simplicity, then RoPE for relative-position handling and better extrapolation.

---

## C. Learned positional embeddings

**9. What's a learned positional embedding?**
A $\text{max\_position} \times d$ matrix; the $t$-th row is the position embedding for position $t$. Added to token embeddings: $\text{input}_t = \text{embedding}(\text{token}_t) + \text{position\_embedding}[t]$. Used in BERT, GPT-2, GPT-3.

**10. Pros of learned positional embeddings?**
Simple. Empirically strong within training range. No hand-designed function.

**11. Cons of learned positional embeddings?**
Hard cap at `max_position`. No extrapolation possible. Position embeddings near `max_position` are noisier than near 0 (less training data for those positions).

**12. Why did learned positional embeddings lose to RoPE?**
The hard cap on context length. Modern users want flexible context lengths, often longer than training. Learned PE cannot extend beyond training length without retraining.

---

## D. RoPE

**13. Walk me through RoPE.**
For each pair of dimensions $(d_{2k}, d_{2k+1})$, treat as a 2D vector and rotate by angle $t \cdot \theta_k$ where $t$ is position and $\theta_k = 10000^{-2k/d}$. Apply this rotation to $Q$ and $K$ (not $V$) before computing attention scores.

**14. Why does the dot product of rotated Q and K depend only on relative position?**
For $Q$ at position $m$ and $K$ at position $n$:

$$
[R(m\theta)\, q]^\top [R(n\theta)\, k] = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R((n-m)\theta)\, k
$$

The rotation matrices' product simplifies to $R((n-m)\theta)$ — a rotation by the difference. So the dot product depends only on $n - m$, not absolute $m$ or $n$.

**15. Why isn't V rotated in RoPE?**
$V$ carries content (the actual information being mixed via attention weights). Rotating $V$ would entangle position with content. Rotating only $Q$ and $K$ cleanly separates position (in attention scores) from content (in value mixing).

**16. What's the complex-number interpretation of RoPE?**
View each pair $(q_{2k}, q_{2k+1})$ as a complex number. Multiplication by $e^{i \cdot t \cdot \theta_k}$ rotates by $t \cdot \theta_k$. The attention dot product becomes the real part of $q^* \cdot k$, which depends on the relative angle.

**17. Why does RoPE outperform sinusoidal in practice?**
(a) Applied at every layer's attention, not just to inputs — stronger positional signal throughout. (b) Relative position by construction — the right inductive bias. (c) Better empirical extrapolation, especially with NTK/YaRN.

**18. Where is RoPE used in production?**
LLaMA, LLaMA-2, LLaMA-3, Mistral, Mixtral, Qwen, Gemma, Gemma 2, Falcon (some variants), GPT-J, GPT-NeoX. Effectively the modern standard for decoder-only LLMs.

---

## E. RoPE extension (NTK, YaRN)

**19. Why doesn't RoPE extrapolate naively?**
High-frequency components ($\theta_k$ for small $k$) cycle quickly, so positions beyond training have "phase configurations" the model never saw. The model can't generalize to those configurations.

**20. What's NTK-aware scaling?**
Scale RoPE's base frequency to compress frequencies into a wider range. Effectively interpolates between trained frequencies, allowing longer context. Free at inference time. Up to ~4× extension with mild quality loss.

**21. What's YaRN?**
Combines per-frequency interpolation (high frequencies fully interpolated, low frequencies untouched) with attention scaling adjustment. Extends context up to ~16× training length with minimal quality loss. Used in several recent open models.

**22. What's linear positional interpolation (Chen et al. 2023)?**
Rescale positions: divide by $L_{\text{test}} / L_{\text{train}}$ so the effective range matches training. Simple. Loses precision at high frequencies. Good for ~4× extension.

**23. Why does context extension matter for production?**
Training a 70B model from scratch at 128K context is infeasibly expensive. Extension methods let you train at 4K–32K and serve at 128K+ with mild quality degradation. Critical for cost-effective long-context serving.

**24. What's LongRoPE?**
Microsoft's search-based approach to RoPE frequency scaling for very long context (millions of tokens). More expensive to set up than YaRN but reportedly better quality at extreme lengths.

---

## F. ALiBi

**25. Walk me through ALiBi.**
Add a linear bias to attention scores penalizing distant positions:

$$
\text{scores}[i, j] \mathrel{+}= -m_h \cdot |i - j|
$$

where $m_h$ is a head-specific slope. No positional embeddings needed; the bias provides position information.

**26. How are ALiBi slopes chosen?**
Press et al. propose $m_h = 2^{-8h/H}$ for head $h$ of $H$. Geometric range from $2^{-8/H}$ (small slope, attends far) to $2^{-8}$ (large slope, attends close). Different heads naturally specialize for different ranges.

**27. ALiBi pros/cons vs RoPE?**
Pros: simpler (no rotations), extrapolates trivially (bias is well-defined at any distance), no need for context extension techniques. Cons: empirically slightly weaker than RoPE at large scales, less expressive (a single bias per relative offset vs RoPE's frequency decomposition).

**28. Where is ALiBi used?**
BLOOM, MPT, some Falcon variants. Its popularity declined as RoPE became dominant.

---

## G. T5 relative bias

**29. What's T5-style relative position bias?**
Add a learned bias to attention scores based on bucketed relative offset:

$$
\text{scores}[i, j] \mathrel{+}= b(\text{bucket}(i - j))
$$

The buckets are typically log-spaced: small offsets get individual buckets; large offsets get coarser bins.

**30. Pros/cons of T5 relative bias?**
Pros: Truly relative. Can extrapolate to longer lengths if bucketing is sensible. Cons: Adds parameters per (head, bucket). Less expressive than RoPE for certain pattern types.

**31. Why isn't it more popular?**
Mostly superseded by RoPE for decoder-only models. Still used in T5, Flan-T5, and some encoder-decoder variants.

---

## H. NoPE and edge cases

**32. What's NoPE?**
No positional encoding at all. Just rely on the causal mask to break permutation invariance.

**33. Why can NoPE work for causal LMs?**
The causal mask itself breaks permutation invariance: position $i$ can only see positions $\leq i$, so the *role* of each position differs (first token has no context; last has full context). This asymmetry provides some implicit position signal.

**34. Why doesn't NoPE work for encoder LMs?**
Encoder LMs (bidirectional) have no causal mask; tokens see each other in both directions. Without explicit position, true permutation invariance returns. NoPE is specifically a causal-LM phenomenon.

**35. NoPE vs RoPE in practice?**
NoPE works comparably at moderate scales for causal LMs. At large scales and long contexts, RoPE generally wins. NoPE is more of a research curiosity than a production technique.

---

## I. Conceptual gotchas

**36. What's the difference between absolute and relative positional encoding?**
Absolute: each position has a unique fixed encoding (sinusoidal, learned). Relative: only position differences matter (T5 bias, RoPE). Modern LLMs prefer relative because it generalizes better.

**37. Can you mix two types of positional encodings?**
You can, but rarely useful. Adding both sinusoidal and learned doubles the position information; mostly redundant. Some research mixes RoPE with global tokens that don't get rotated, but these are special cases.

**38. What's xPos?**
RoPE + exponential decay on long-range attention. Better extrapolation at slight quality cost. Used in some research models; not mainstream.

**39. How does positional encoding interact with sparse attention?**
For sliding window: position information must work within the window. RoPE works fine because relative offsets within a window are small. For global tokens (Longformer), you may need special position handling (no positional encoding for `[CLS]`, etc.).

---

## J. Quick fire

**40.** *Original positional encoding paper?* Vaswani et al. 2017.
**41.** *RoPE paper?* Su et al. 2021.
**42.** *ALiBi paper?* Press et al. 2021.
**43.** *YaRN paper?* Peng et al. 2023.
**44.** *RoPE base frequency?* $10000^{-2i/d}$.
**45.** *Default ALiBi slope?* $2^{-8h/H}$ for head $h$ of $H$.

---

## Self-grading

If you can't answer 1-10, you don't know positional encodings. If you can't answer 11-25, you'll struggle on architecture deep-dives. If you can't answer 26-40, frontier-lab interviews will go past you.

Aim for 30+/40 cold.
