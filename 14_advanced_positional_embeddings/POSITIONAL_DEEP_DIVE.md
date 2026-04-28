# Positional Embeddings: A Frontier-Lab Interview Deep Dive

> **Why this exists.** Position is the one thing transformers don't get for free — pure attention is permutation-equivariant. How position is encoded determines whether your model can extrapolate beyond training length, whether it can do precise position-dependent operations, and how easily you can extend the context window. Modern LLMs use RoPE; understanding its derivation is now a standard interview question.

---

## 1. Why position needs encoding at all

Pure scaled-dot-product attention has a striking property: **permutation equivariance**. If you permute the input tokens, the output is permuted the same way. Concretely, for any permutation matrix $P$:

$$
\mathrm{Attention}(P \cdot X) = P \cdot \mathrm{Attention}(X)
$$

This means attention has no innate notion of order. From the model's perspective, "the cat sat on the mat" is identical to "mat the on sat cat the." For language — where order is everything — this is the wrong inductive bias.

Positional encoding injects order information. It's the **only** mechanism by which transformers know what comes first.

---

## 2. The four eras of positional encoding

| Method | Year | Used by | Status |
|---|---|---|---|
| Sinusoidal | 2017 | Original transformer | Historical |
| Learned | 2018 | BERT, GPT-2, GPT-3 | Replaced for length flexibility |
| Relative | 2018-19 | T5, Transformer-XL | Niche; replaced by RoPE |
| RoPE | 2021 | LLaMA, Mistral, Gemma, Qwen, etc. | **Modern standard** |
| ALiBi | 2021 | BLOOM, MPT | Modern alternative |
| NoPE | 2023 | Some research | Surprising — works for some causal LMs |

We'll cover each.

---

## 3. Sinusoidal positional encoding (Vaswani 2017)

The original method. Add a deterministic positional vector to the input embedding:

$$
\mathrm{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad \mathrm{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

$$
\text{input}_t = \text{embedding}(\text{token}_t) + \mathrm{PE}(t)
$$

Each dimension oscillates at a different frequency, ranging from $2\pi$ (low dim) to $10000 \cdot 2\pi$ (high dim). Different positions get distinguishable signatures.

### Properties

**Position invariance under linear transform.** For any fixed $\Delta t$, there exists a linear transform $M_{\Delta t}$ such that $\mathrm{PE}(t + \Delta t) = M_{\Delta t} \cdot \mathrm{PE}(t)$. This means relative positions can in principle be computed by linear operations on absolute encodings — which the model can learn.

**Extrapolation in theory.** The encoding is defined for any $t$, including $t$ larger than training length. So extrapolation is mathematically possible.

**Extrapolation in practice.** Mediocre. The model learns position-specific patterns that don't generalize cleanly. The encoding is well-defined at long range, but the *learned weights* aren't.

### Why sinusoidal looks like a magic formula

The exponential frequency spacing $10000^{2i/d}$ was chosen so frequencies span many orders of magnitude. Why 10000? Empirical choice; not deeply principled. The factor would work approximately as well at 1000 or 100000.

### Why it lost

- Length extrapolation in practice was disappointing.
- Adding to input embeddings means position information must propagate through layers; later layers have weaker positional signal.
- RoPE provides the same relative-position property but cleanly mixed into attention.

---

## 4. Learned positional embeddings

Treat position as a categorical variable; learn a $d$-dim embedding per position:

$$
\text{input}_t = \text{embedding}(\text{token}_t) + \text{position\_embedding}[t]
$$

$\text{position\_embedding} \in \mathbb{R}^{\text{max\_position} \times d}$.

### Pros

- Simple. No hand-designed function.
- Empirically strong within training range.
- BERT, GPT-2, GPT-3 used this.

### Cons

- **Hard cap on context length** = `max_position`. Beyond it, you have no embedding.
- **No extrapolation possible.** Position 1025 is unseen if training maxed at 1024.
- Position embeddings near `max_position` trained on fewer examples → noisier than positions near 0.

### Why it lost

The hard cap is the killer. Modern users want long context windows; learned positional embeddings can't extend beyond training without retraining.

---

## 5. T5-style relative position bias

T5 (Raffel et al. 2020) replaced absolute positional encodings with a **bias term** added directly to attention scores:

$$
\text{scores}[i, j] = \frac{Q_i \cdot K_j}{\sqrt{d_k}} + b(i - j)
$$

$b(i - j)$ is a learned scalar per relative offset, with offsets bucketed (e.g., into ~32 buckets, log-spaced for distant offsets).

### Pros

- Truly relative. No absolute positions.
- Can extrapolate to new lengths if the bucketing is sensible.
- Elegant: position handled in attention, not added to embeddings.

### Cons

- Adds a learnable parameter per (head, bucket).
- Less expressive than RoPE for certain pattern types.
- The bucketing is ad-hoc.

### Status

Used in T5, Flan-T5. Mostly superseded by RoPE for decoder-only models.

---

## 6. RoPE — Rotary Position Embedding (Su et al. 2021)

The dominant method in modern LLMs. Worth fully understanding because interviewers love this one.

### The setup

Group dimensions into pairs. For each pair $(d_{2k}, d_{2k+1})$ of dimensions in $Q$ and $K$, **rotate** that pair by an angle proportional to position:

$$
q_t^{(k)} \leftarrow R(t \cdot \theta_k) \cdot q_t^{(k)}, \qquad k_t^{(k)} \leftarrow R(t \cdot \theta_k) \cdot k_t^{(k)}
$$

where $R(\alpha)$ is the standard 2D rotation matrix:

$$
R(\alpha) = \begin{bmatrix} \cos \alpha & -\sin \alpha \\ \sin \alpha & \cos \alpha \end{bmatrix}
$$

and $\theta_k = 10000^{-2k/d}$ — same frequency schedule as sinusoidal.

### Why this gives **relative** position information

Key insight. The dot product of rotated $Q$ and $K$ at positions $m$ and $n$:

$$
[R(m\theta)\, q]^\top [R(n\theta)\, k] = q^\top R(m\theta)^\top R(n\theta)\, k = q^\top R((n - m)\theta)\, k
$$

The $m$ and $n$ only show up as their **difference**. The dot product depends solely on relative position $n - m$, not absolute $m$ or $n$.

So even though we're encoding absolute positions on $Q$ and $K$, the attention scores end up depending only on relative position. This is the elegant trick.

### Implementation

In practice, RoPE is implemented as element-wise complex multiplication:

- View pairs $(q_{2k}, q_{2k+1})$ as complex numbers $\tilde q_k = q_{2k} + i \cdot q_{2k+1}$.
- Multiplication by $e^{i \cdot t \cdot \theta_k}$ rotates by $t \cdot \theta_k$.
- The dot product becomes the real part of $\tilde q^* \cdot \tilde k$, which depends on the relative angle.

### Why RoPE outperforms sinusoidal in practice

- **Mixed into attention, not embeddings.** Position information is applied at every layer's attention, not just to inputs. Stronger positional signal throughout the network.
- **Relative by construction.** The dot product depends only on relative position, which is the right inductive bias.
- **Better extrapolation.** Empirically much better than sinusoidal at extending beyond training length, especially with techniques like NTK-aware scaling.

### V is not rotated

Important detail: only $Q$ and $K$ are rotated. $V$ stays as-is. Reason: $V$ carries content; rotating it would entangle position with content. Keeping $V$ un-rotated cleanly separates position (in attention scores) from content (in value mixing).

---

## 7. RoPE length extrapolation: NTK and YaRN

RoPE works well at training length, but naive extrapolation beyond training length still breaks down — frequencies trained at short ranges don't generalize cleanly.

### The problem

The high-frequency components of RoPE ($\theta_k$ for small $k$) wrap around quickly. At positions beyond training, these frequencies have completed many full rotations; the model never trained on these phase configurations.

The low-frequency components ($\theta_k$ for large $k$) extrapolate cleanly — they haven't even completed one full rotation in training, so the model has plenty of room.

### NTK-aware scaling (bloc97 2023)

Scale RoPE's base frequency to compress frequencies, effectively interpolating between trained frequencies. For target context length $L_{\text{target}}$ and training length $L_{\text{train}}$:

$$
\text{scale} = \left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{d/(d-2)}
$$

$$
\theta_k^{\text{new}} = \theta_k^{\text{original}} \cdot \text{scale}^{-2k/d}
$$

This stretches RoPE's effective range without retraining. Free at inference time. Mostly preserves quality up to ~4× extension.

### YaRN (Peng et al. 2023)

Extends NTK scaling with:

- Per-frequency interpolation: high frequencies fully interpolated, low frequencies untouched.
- Attention scaling: $1/\sqrt{d_k}$ adjusted to compensate for stretched frequencies.

YaRN extends context to ~16× training length with minimal quality loss. Used by several recent open models.

### Linear positional interpolation (Chen et al. 2023)

Simply rescale positions: instead of training at length $L_{\text{train}}$ and testing at $L_{\text{test}}$, divide all positions by $L_{\text{test}} / L_{\text{train}}$ so the effective range is unchanged. Simple but loses precision at high frequencies. Good enough for ~4× extension.

### What "extension" means in practice

Training a 7B model from scratch at 32K context is expensive. RoPE extension methods let you train at 4K and serve at 32K with mild quality degradation. Used widely in production.

---

## 8. ALiBi (Press et al. 2021)

A radically simpler approach: add a linear bias to attention scores that penalizes attending to distant positions.

$$
\text{scores}[i, j] \mathrel{+}= -m_h \cdot |i - j|
$$

$m_h$ is a head-specific slope (a hyperparameter, not learned). The bias linearly increases with distance, so far-away positions are exponentially down-weighted in the softmax.

### Properties

- No positional embeddings at all.
- Linear bias is monotonic in distance — the model can attend to closer positions more.
- Different heads use different $m_h$, allowing different attention ranges.
- Extrapolates well: the linear bias is well-defined at any distance.

### Slope choice

Press et al. propose: head $h$ of $H$ heads gets slope $m_h = 2^{-8h/H}$. So slopes range geometrically from $2^{-8/H}$ (small, attends far) to $2^{-8}$ (large, attends close). Different heads naturally specialize for different ranges.

### Used in

BLOOM, MPT (MosaicML), some Falcon variants. Simpler than RoPE but slightly weaker empirically; lost popularity to RoPE.

---

## 9. NoPE (No Position Encoding)

Surprising recent finding (Kazemnejad et al. 2023): **causal language models can sometimes work without explicit position encodings**.

### Why this could work

The causal mask itself breaks permutation invariance. Position $i$ can only see positions $\leq i$, which means the *role* of each position differs (the first token has no context; the last has full context). This asymmetry alone provides some position signal.

### Empirical findings

Counterintuitively, Kazemnejad et al. (2023) found NoPE *generalizes better to lengths beyond training* than RoPE/sinusoidal/ALiBi on length-extrapolation tasks. The trade-off: NoPE's in-context-learning quality at fixed length is somewhat weaker, and the result hasn't transferred cleanly to flagship-scale pre-training. NoPE remains a research curiosity rather than a production default, but its existence — and the fact that the causal mask alone provides positional information — is interesting and might come up in interviews.

---

## 10. xPos and other RoPE variants

### xPos (Sun et al. 2022)

RoPE plus an exponential decay term that further dampens long-range attention. Better extrapolation at the cost of slight quality at training range. Used in some research models.

### Adaptive RoPE

Dynamically adjust frequencies based on sequence length. Active research area.

### LongRoPE (Microsoft, 2024)

Applies search-based frequency scaling to extend RoPE. Used in models with multi-million-token contexts.

### Relative position bias variants

T5 bias, Transformer-XL biases, AliBI — all variations on adding learned/heuristic biases to attention scores. Most replaced by RoPE in modern LLMs.

---

## 11. Practical interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why does the original transformer use sinusoids?" | Theoretical extrapolation property (linear transform between positions). Empirically mediocre; replaced by RoPE. |
| "What's the difference between absolute and relative position?" | Absolute: each position gets a unique encoding. Relative: only position differences matter. RoPE achieves relative via rotation. |
| "Why RoPE only on Q and K, not V?" | $V$ carries content; rotating $V$ would entangle position with content. Keeping $V$ un-rotated cleanly separates position (in attention scores) from content (in value mixing). |
| "Why does RoPE encode relative position from absolute rotations?" | Dot product of rotated $Q$ at position $m$ and rotated $K$ at position $n$ equals $q^\top R((n-m)\theta)\, k$ — depends only on $n - m$. |
| "How do you extend context with RoPE?" | NTK-aware scaling, YaRN, linear interpolation. All retrain-free. |
| "Why isn't ALiBi as popular as RoPE?" | Slightly weaker quality empirically, especially at large scales. RoPE's relative-position property is better aligned with what the model needs. |
| "What's the receptive field of RoPE?" | Unlimited in principle; in practice limited by training length and extrapolation. With YaRN, ~16× training length. |
| "Why do learned positional embeddings cap context length?" | They're a finite lookup table. No embedding for positions beyond `max_position` seen during training. |

---

## 12. The 8 most-asked positional encoding questions

1. **Why do transformers need positional encoding?** Pure attention is permutation-equivariant; without position, "cat sat" = "sat cat".
2. **Sinusoidal vs learned vs RoPE?** Sinusoidal: theoretical extrapolation, mediocre in practice. Learned: simple, hard cap. RoPE: relative position via rotation, modern standard.
3. **Walk me through RoPE.** Rotate $Q$ and $K$ by angle proportional to position; dot product depends on relative position.
4. **Why does V not get rotated?** $V$ is content, not query/key. Rotating would entangle position with content.
5. **What's ALiBi?** Linear bias on attention scores penalizing distant positions. Simpler than RoPE; weaker empirically.
6. **How do you extend RoPE context?** NTK-aware scaling, YaRN, position interpolation. Free at inference.
7. **What's the relative position story for sinusoidal?** Sinusoidal is fundamentally absolute; the linear-transform property between positions enables learning relative effects but doesn't enforce them.
8. **What's NoPE?** No position encoding. Surprisingly works for causal-only models at moderate scales because the causal mask breaks permutation invariance.

---

## 13. Drill plan

1. Whiteboard the RoPE rotation and the relative-position derivation.
2. Memorize the four-method comparison (sinusoidal/learned/RoPE/ALiBi).
3. Know how YaRN works at a sketchy level for context extension questions.
4. Drill `INTERVIEW_GRILL.md`.

---

## 14. Further reading

- Vaswani et al., "Attention is All You Need" (sinusoidal, 2017).
- Devlin et al., "BERT" (learned positional embeddings, 2018).
- Raffel et al., "T5" (relative position bias, 2020).
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE, 2021).
- Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (ALiBi, 2021).
- Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation" (2023).
- Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models" (2023).
- Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers" (NoPE analysis, 2023).
- Sun et al., "A Length-Extrapolatable Transformer" (xPos, 2022).
