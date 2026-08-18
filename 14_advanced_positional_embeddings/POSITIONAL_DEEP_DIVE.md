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

> **Saying it out loud.** So the thing to know is that attention on its own has no idea what order the words arrived in. Shuffle the tokens and you get the same outputs back, just shuffled — the math genuinely cannot tell "the cat sat on the mat" from "mat the on sat cat the." That's why we bolt on a positional encoding: it is the only place in the entire transformer where order enters the computation. And it's not academic — get it wrong and the model can't count, can't copy in order, and falls apart past the length it trained on.

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

> **Saying it out loud.** The short history is: a fixed sine-wave formula in 2017, then just learning one vector per position for BERT and GPT-3, then relative-position biases in T5, and then RoPE showed up in 2021 and basically won. Open LLaMA, Mistral, Qwen or Gemma today and you'll find RoPE; ALiBi is the main live alternative, in BLOOM and MPT. The one-line reason RoPE won is that it hands you relative position for free inside the attention dot product, and that's what makes retrain-free context extension possible.

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

> **Saying it out loud.** Sinusoidal encoding gives every position a fingerprint built from sine and cosine waves at many different frequencies — picture a rack of clocks whose hands tick at wildly different speeds, so no two moments look alike. The elegant property is that shifting by a fixed distance is a linear transform, so in principle the model can recover relative offsets from absolute codes. In practice it underdelivered: the formula is defined at any position, but the weights reading it only ever saw short ones. That's the line to say — it extrapolates on paper, not in reality.

---

## 4. Learned positional embeddings

Treat position as a categorical variable; learn a $d$-dim embedding per position:

$$
\text{input}_t = \text{embedding}(\text{token}_t) + \text{position-embedding}[t]
$$

$\text{position-embedding} \in \mathbb{R}^{\text{max-position} \times d}$.

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

> **Saying it out loud.** Learned positional embeddings are the dumbest thing that works: keep a lookup table with one vector per position and add it to the token embedding. Simple, and strong inside the training range, which is why BERT and GPT-3 used it. The killer is that a table is finite — if you trained to 1024, position 1025 simply doesn't exist, so context length is hard-capped and you can't extend without retraining. There's a subtler failure too: the last rows of the table appear in far fewer training examples, so they're noisier than the early ones.

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

> **Saying it out loud.** T5's move is to stop putting position in the embeddings at all and instead add a learned number directly onto the attention score, based on how far apart the two tokens are. Offsets get bucketed — nearby distances get their own bucket, far ones get lumped together on a log scale — so you learn maybe 32 numbers per head instead of one per distance. The upside is that it's genuinely relative, so it can stretch to lengths you didn't train on. The downside is the bucketing is ad hoc and it's less expressive than RoPE, which is why it stayed inside the T5 family instead of taking over.

---

## 6. RoPE — Rotary Position Embedding (Su et al. 2021)

*In plain language:* RoPE is the position scheme nearly every modern LLM uses. Before any notation: take each query and key vector, chop it into pairs of numbers, and spin each pair like a clock hand by an angle proportional to the token's position. Everything below is just that one idea written with a 2D rotation matrix.

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

> **Saying it out loud.** So the reason RoPE works is a rotation trick. You spin each query and key by an angle proportional to its position, and when you dot a query at position $m$ against a key at position $n$, the two rotations compose into a single rotation by $n - m$. The absolute positions cancel and only the distance survives — you encoded absolute position and got relative position out, inside attention, at every single layer. The detail interviewers listen for is that only Q and K get rotated, never V: V carries the content, and spinning it would smear position into the information you're actually moving.

---

## 7. RoPE length extrapolation: NTK and YaRN

*In plain language:* this section is about running a model on sequences longer than it was trained on without retraining it. NTK-aware scaling, YaRN, and position interpolation are three flavours of one move — slow the rotations down so a long sequence folds back into the range of angles the model already knows.

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

> **Saying it out loud.** The problem is that RoPE's fastest-spinning dimensions have wrapped around many times by the time you're past the training length, so the model is staring at angle combinations it has never seen. The fix is to slow the rotations down so a longer sequence maps back into the angle range it does know. Linear interpolation does that bluntly and buys about 4x; NTK-aware scaling does it per-frequency, also around 4x; YaRN leaves the slow frequencies alone, fully interpolates the fast ones, and rescales attention, and that gets you roughly 16x with very little quality loss. That's why you can train at 4K and serve at 32K instead of paying for long-context pretraining.

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

> **Saying it out loud.** ALiBi throws out positional vectors entirely and just subtracts a penalty from the attention score that grows linearly with distance — the further away a token is, the harder it is to attend to. Each head gets its own fixed slope, so some heads look nearby and some look far, and the slopes come from a formula rather than being learned. Because a straight line is defined at any distance, extrapolation is free; that was the whole pitch, train short and test long. The tradeoff is expressiveness — one scalar penalty per distance is less flexible than RoPE's per-frequency decomposition, and at scale it measures out slightly weaker, which is why RoPE won.

---

## 9. NoPE (No Position Encoding)

Surprising recent finding (Kazemnejad et al. 2023): **causal language models can sometimes work without explicit position encodings**.

### Why this could work

The causal mask itself breaks permutation invariance. Position $i$ can only see positions $\leq i$, which means the *role* of each position differs (the first token has no context; the last has full context). This asymmetry alone provides some position signal.

### Empirical findings

Counterintuitively, Kazemnejad et al. (2023) found NoPE *generalizes better to lengths beyond training* than RoPE/sinusoidal/ALiBi on length-extrapolation tasks. The trade-off: NoPE's in-context-learning quality at fixed length is somewhat weaker, and the result hasn't transferred cleanly to flagship-scale pre-training. NoPE remains a research curiosity rather than a production default, but its existence — and the fact that the causal mask alone provides positional information — is interesting and might come up in interviews.

> **Saying it out loud.** NoPE is the surprising result that a decoder-only model can sometimes just not have a positional encoding and still work. The reason is the causal mask: token 1 sees one thing and token 500 sees five hundred, so every position already has a distinguishable role even with no explicit signal. On pure length-generalization benchmarks it actually beat RoPE and ALiBi. The catch — say this so you don't sound naive — is that it's a causal-only phenomenon, since a bidirectional encoder without position codes is truly permutation-invariant, and it hasn't held up at flagship pretraining scale. Research curiosity, not a default.

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

> **Saying it out loud.** Past vanilla RoPE there's a family of tweaks, and mostly you need to recognise the names. xPos adds an exponential decay on top of the rotation so distant attention fades — better extrapolation, slightly worse quality at the training length. LongRoPE searches for per-dimension frequency scalings and is what gets you into million-token context territory. The older relative-bias methods — T5, Transformer-XL, ALiBi — are all the same idea of adding something to the attention score, and they've largely been displaced. The honest summary is that essentially everything since 2021 is a modification of RoPE's frequency schedule, not a new idea.

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
