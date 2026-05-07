# Modern LLM Architecture Choices — What Everyone Actually Does

> Distilled from Stanford CS336 (Tatsu Hashimoto's "Architecture and Hyperparameters" lecture, 2025) + cross-referenced with the recent open-model lineage (Llama 2/3/4, Qwen 2.5/3, Gemma 2/3/4, DeepSeek V2/V3, OLMo, Mistral, Cohere Command A, etc.).
>
> The point: the original 2017 transformer got most things right. Modern LLMs differ only on a small number of architectural axes. This chapter walks each axis, says **what the consensus is**, **what alternatives exist**, **why people changed**, and **how confidently to use each choice**.
>
> Pair with `TRANSFORMERS_DEEP_DIVE.md` (architecture mechanics), `05_attention_mechanisms/ATTENTION_DEEP_DIVE.md` (attention details), `14_advanced_positional_embeddings/POSITIONAL_DEEP_DIVE.md` (position embedding depth), `61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md` (training systems).

---

## 0. The mental model

Architecture design is not elegant. It's a multi-objective optimization over **representation quality**, **GPU utilization (arithmetic intensity, parallelism)**, and **stability** at scale. Every "modern" choice exists because all three have to be satisfied — sometimes at the cost of theoretical purity.

Three eras:

- **2017–2020 (Vaswani → GPT-3).** Lots of experimentation. No standard.
- **2023 (Llama 2 onward).** Llama 2 became the gold standard; everyone trained "Llama 2 likes" with minor variations.
- **2024–2026.** Re-experimentation focused on (a) stability at extreme scale, (b) long context.

The key insight: **most architecture hyperparameters live in wide forgiving basins** — getting them roughly right is enough; getting them very wrong is fatal. Senior engineering means knowing which knobs are wide-basin (most) and which are narrow (a few stability-critical ones).

---

## 1. Layer Norm Placement — the one thing everyone agrees on

### Consensus: move LayerNorm OUT of the residual stream

Original transformer (Vaswani 2017) put LayerNorm **inside** the residual path (post-norm in residual). This was the wrong call.

Three modern placements:

```
                 Pre-Norm                  Post-Norm-outside-residual           Double Norm
                (most models)               (Grok, Gemma 2, OLMo 2)              (some)
                
              x ──┐                            x ──┐                          x ──┐
                  │                                │                              │
              LayerNorm                            │                          LayerNorm
                  │                                │                              │
              Attention                       Attention                       Attention
                  │                                │                              │
                  │                            LayerNorm                      LayerNorm
                  │                                │                              │
              x + Δ ←───                       x + Δ ←───                     x + Δ ←───
```

**All modern models put LayerNorm OUTSIDE the residual stream.** The only exception was OPT-350M (and OPT was a mess in general).

### Why pre-norm works

The mantra: **"keep your residual stream clean."**

With pre-norm, the residual `x` flows uninterrupted from input to output. In the backward pass, gradients propagate **straight through** without normalization distortion at every layer. This:
- Removes the need for warmup (originally) — though warmup is still used in practice.
- Stabilizes gradient norms across depth.
- Reduces gradient-spike frequency (Salazar & Yen).
- Enables training very deep models.

Post-norm-inside-residual breaks all of this.

### Pre-norm vs post-norm-outside-residual

Both work. Pre-norm is more common (likely because Llama 2 used it). Post-norm-outside (apply LayerNorm to the output of attention/MLP, then add to residual) is used by Grok, Gemma 2, OLMo 2.

**Double norm** (some models): pre-norm + post-norm both, outside the residual. Wasteful but stable. Used as a safety net when stability issues appear.

### The "sprinkle layer norms" heuristic

When stability problems arise during training, the empirical answer is: **add more layer norms**. Inside attention. Inside the MLP. After the soft-cap. Wherever. Strange but reliably effective.

### Hook

> "Modern LLMs all put LayerNorm outside the residual stream — pre-norm or post-norm-outside, sometimes both. Original transformer was wrong on this. Keep the residual stream clean and gradients propagate straight through."

---

## 2. RMS Norm replaces LayerNorm

### Consensus: use RMSNorm, not LayerNorm

RMSNorm drops two operations from LayerNorm:
- The mean subtraction.
- The bias term.

```
LayerNorm:  y = γ * (x − mean) / std + β
RMSNorm:    y = γ * x / RMS(x)        where RMS(x) = sqrt(mean(x²))
```

### Why

- Representationally, LayerNorm is more expressive. In practice, **the gap is negligible** — models train to nearly identical loss.
- LayerNorm is **memory-bound** (low arithmetic intensity). Even though it's only ~0.17% of FLOPs, it can be **up to 25% of runtime** on small models due to memory movement.
- Dropping the mean-subtraction and bias halves the memory traffic.
- Free systems win.

This is the canonical example of "drop arithmetically-light, memory-heavy ops."

### Bias terms more broadly

**Drop the bias terms in linear layers too.** Same argument: bias adds memory movement, contributes minimally to expressiveness, sometimes induces stability issues. Most modern models drop biases everywhere.

### Hook

> "RMSNorm = LayerNorm minus mean-centering minus bias. Same quality, faster. Free systems win driven by GPU memory-bandwidth pressure on a low-arithmetic-intensity op."

---

## 3. Activations — Gated Linear Units win

### The activation zoo

- **ReLU** (vanilla). Works fine. Chinchilla used it.
- **GELU** (Gaussian Error Linear Unit). ReLU with a smooth divot near zero. GPT-3 used it.
- **Swish / SiLU**. `x · sigmoid(x)`. Smooth nonlinearity.
- **Gated** variants (the modern winners): **ReGLU, GeGLU, SwiGLU**.

### What's a Gated Linear Unit?

Standard FFN: `out = W2 · activation(W1 · x)`.

GLU FFN: `out = W2 · (activation(W1 · x) ⊙ V·x)` where ⊙ is elementwise multiplication.

Two extra paths multiplied together: one through the activation, one through a separate linear `V`. The gate `V·x` modulates the activation output.

- **ReGLU** = ReLU as the activation.
- **GeGLU** = GELU as the activation. Used by Gemma family, T5 v1.1.
- **SwiGLU** = Swish as the activation. Used by PaLM, Llama family, Mistral, Qwen, DeepSeek.

### Param count adjustment

GLU adds a third matrix (V). To keep total parameter count the same as a non-gated FFN with `d_ff = 4·d_model`, **shrink `d_ff` by 2/3**:

```
non-gated:  d_ff = 4 · d_model
GLU:        d_ff = (8/3) · d_model ≈ 2.67 · d_model
```

This is the universal correction. Llama added another 1.33× multiplier (history-dependent), giving `d_ff ≈ 3.5 · d_model`.

### The empirical case

Noam Shazeer's GLU-variants paper used **error-bar comparisons across multiple training replicates**. GLU variants beat their non-gated counterparts at parameter-matched comparison. Consistently. The Narang et al. 2020 controlled architecture study reproduced this on T5.

### Exceptions worth knowing

- GPT-3: GELU (no gating).
- Nemotron 340B: squared ReLU (no gating).

Both work. But the modern bias is firmly toward GLUs.

### Hook

> "SwiGLU or GeGLU. Add a gate; shrink `d_ff` by 2/3 to match params. Almost universal in modern LLMs."

---

## 4. Parallel vs Serial Transformer Blocks

### The choice

**Serial** (original):
```
x' = x + Attention(LN(x))
x'' = x' + MLP(LN(x'))
```

**Parallel** (GPT-J, PaLM, Cohere):
```
x' = x + Attention(LN(x)) + MLP(LN(x))
```

Both attention and MLP read the same normalized residual; their outputs add into the residual together. Allows fused matmuls and shared LayerNorm — systems win.

### Status: falling out of favor

PaLM showed parallel was free (no quality drop, +15% systems). But:
- Serial-block optimizations have improved (FlashAttention etc.).
- Parallel is essentially "half as deep" in a representational sense.
- Recent Google models have moved away from it.

Modern default: **serial**. Parallel was an interesting experiment that didn't win.

---

## 5. Position Embeddings — RoPE is the consensus

### The historical menu

| Type | Models | Idea |
|---|---|---|
| Sinusoidal | Vaswani 2017 | Add fixed sin/cos vectors to inputs |
| Absolute (learned) | BERT, GPT-2 | Each position has a learned embedding |
| Relative (added to attention) | T5, Chinchilla | Bias the attention matrix by `(i-j)` |
| **RoPE (Rotary Position Embedding)** | **Most post-2024** | **Rotate Q/K vectors by position-dependent angles** |
| ALiBi | MPT, BLOOM | Linear-distance penalty added to attention |
| NoPE | Some recent | No position info; rely on causal mask |
| Pi-RoPE | Gemma 4 | Rotate only first 2 coords |

### RoPE intuition (the only one you must know cold)

**Constraint we want.** For `f(x_i, i)` and `f(x_j, j)`, the inner product `⟨f(x_i, i), f(x_j, j)⟩` should depend **only on `i − j`** (relative position), not on absolute `i` and `j`.

**Key fact.** Inner products are invariant under arbitrary rotation. If we rotate `x_i` by an angle proportional to `i`, then translating both positions by `+k` rotates both by the same extra amount, leaving the relative angle (and inner product) unchanged.

**The construction.**

```
2D case:    x_i is a 2-vector. Rotate by θ·i.
            Rotation matrix R(θ·i) acts on x_i.
            ⟨R(θ·i)·x_i, R(θ·j)·x_j⟩ = ⟨x_i, R(θ·(j-i))·x_j⟩  →  depends only on (j−i).

D-dim case: split D dims into D/2 pairs. Rotate each pair independently.
            Different pairs use different θ_k frequencies:
            θ_k = base^(-2k/d), base = 10000 (typical).
            Low-frequency pairs capture long-range; high-frequency capture local.
```

**Where it's applied.** Only to Q and K (not to V). Inside the attention layer, after the QKV projection, before the QK matmul.

**Why it dominates.** Pure relative encoding via inner-product geometry. No additive cross terms. Extends to arbitrary length (with adjustable base / NTK / YaRN scaling for context extension).

### Pi-RoPE / proportional RoPE (Gemma 4)

Rotate only the first two coordinates per head. Drops all the low-frequency channels. Surprisingly, works.

### Hook

> "RoPE rotates Q and K vector-pairs by an angle proportional to position. Inner product becomes a function of relative offset only. D-dim case: chunk into 2D pairs, rotate each at a different frequency. Universal in post-2024 LLMs."

---

## 6. Hyperparameters — the wide forgiving basins

The "wow there are so many choices" panic. Reality: most have a wide basin where almost any value works.

### 6.1 FFN ratio (`d_ff / d_model`)

Standard rule: `d_ff = 4 · d_model`. With GLUs: `d_ff = 2.67 · d_model` (after 2/3 correction). Llama variant: `3.5`.

**Wide basin.** Kaplan et al. 2020 showed the loss is nearly flat from ratio = 1 to 10. Above 10, loss climbs quadratically.

**One outlier:** T5 used **64×**. Argued for big matmuls = better hardware utilization. They abandoned this in T5 v1.1 → reverted to 2.5×. So even radical choices technically work but are compute-inefficient.

### 6.2 Head dim × num heads

Standard rule: `head_dim · num_heads = d_model`. So `head_dim = d_model / num_heads`.

Almost all models follow this. Wide basin around the rule.

### 6.3 Aspect ratio (`d_model / n_layers`)

Standard: ~**100**. GPT-3, Llama, most modern models hover here.

**Why this band:**
- Too deep: pipeline parallelism becomes a nightmare; gradient propagation harder.
- Too wide: tensor parallelism has limits; under-utilizes depth's expressiveness.
- ~100 is the systems-vs-expressiveness sweet spot.

**Kaplan et al. and EK et al.** showed: across a wide depth-width sweep, **FLOPs determine quality**, not aspect ratio. As long as you're roughly in the band, you're fine.

### 6.4 Vocabulary size

| Class | Vocab | Examples |
|---|---|---|
| Monolingual (early open) | ~30K | GPT-2, OPT, LLaMA 1 |
| Multilingual / production | 100–200K+ | GPT-4, Gemini, Llama 3 (128K), Qwen 3, Gemma 3 |

Bigger model → can handle bigger vocab. Bigger vocab → better multilingual coverage and shorter token sequences. Multimodal adds image-token vocabularies on top.

### 6.5 Summary of "default-and-be-done" hyperparameters

| Knob | Default | Wide-basin? |
|---|---|---|
| `d_ff / d_model` | 4 (vanilla) / 2.67 (GLU) / 3.5 (Llama) | Very wide, ~1–10 |
| `head_dim · num_heads` | = `d_model` | Wide |
| `d_model / n_layers` | ≈ 100 | Wide |
| Vocab | 30K (mono) / 128–200K (multi) | Discrete buckets |
| Layer norm placement | Pre-norm or post-norm-outside | Discrete: never inside residual |
| Activation | SwiGLU / GeGLU | Strong default |
| Position embedding | RoPE | Strong default |
| Bias terms | Off | Off |

---

## 7. Regularization — counterintuitive

### Dropout

Standard ML101 says: regularize. But for LLM pretraining, you typically only do **one pass** over data → no overfitting → dropout is mostly removed.

Some models still use small dropout at attention. Most don't.

### Weight decay — actually an optimization intervention

Most modern LLMs **use weight decay**, despite no overfitting concern. Why?

**The finding (multiple recent papers):** with single-pass SGD/AdamW and learning-rate decay, weight decay is not regularizing — there's no train/val gap to close. Instead, weight decay **interacts with the learning-rate schedule** to land at a better minimum.

Concretely: weight decay + learning-rate decay together push the model toward smoother basins. Weight decay alone with constant LR doesn't help much.

**Practical implication:** use weight decay (~0.1) and a cosine LR schedule. Don't reason about regularization; reason about optimization dynamics.

### Hook

> "In one-epoch LLM pretraining, weight decay is an optimization intervention not a regularizer. It interacts with LR decay to find better minima."

---

## 8. Stability — the late-2024 frontier

As models grow, **stability** matters more than per-step quality. A single irrecoverable spike can waste a multi-million-dollar run.

The **danger zones** are softmaxes (exp + division). LLMs have two:
- Output softmax (token logits).
- Attention softmax (over Q·K^T scores).

### 8.1 z-loss — output-softmax stability

The output log-probability decomposes as `log p = u − log Z`, where `Z = Σ_v exp(u_v)`. If `Z` blows up or collapses to 0, training is unstable.

**Trick.** Add `λ · (log Z)²` to the loss. Penalizes `log Z` away from 0. The softmax is invariant to adding a constant to all logits, so this just regularizes the normalizer.

Used by: Bichuan, DCLM, OLMo. Cheap, effective.

### 8.2 QK-Norm — attention-softmax stability

Apply RMSNorm to Q and K **before** the QK matmul. This bounds the input to the attention softmax to a fixed scale, preventing pathological score blow-up.

```
Q, K, V = QKV(LN(x))
Q_norm  = RMSNorm(Q)        # NEW
K_norm  = RMSNorm(K)        # NEW
scores  = Q_norm @ K_norm^T / sqrt(d_h)
attn    = softmax(scores)
out     = attn @ V
```

Originally from multimodal training (IDEFICS, Chameleon). Now in many open LLMs as a stability baseline. **No quality cost; allows higher learning rates.**

### 8.3 Logit soft-capping — Gemma's hard guardrail

```
scores = soft_cap · tanh(scores / soft_cap)
```

Hard ceiling on logit magnitude. Prevents extreme attention. Used by Gemma 2/3/4.

**Tradeoff.** Stronger guardrail than QK-Norm but **slight quality loss** (you cap how confident the softmax can be). NVIDIA's systematic comparison showed QK-Norm > soft-capping if you only pick one.

### 8.4 The "sprinkle layer norms everywhere" heuristic

Half-joke, fully real. Stability issue → add a layer norm. Around attention. Around MLP. After soft-cap. Wherever. Empirically works.

### Hook

> "Two soft-maxes are the danger zones. z-loss for the output one. QK-Norm for the attention one. Sprinkle more layer norms when desperate. Logit soft-capping is a bigger hammer but costs a bit of quality."

---

## 9. Attention Variants — the inference economics story

### 9.1 Why MHA hurts at inference

**Training / prefill:** arithmetic intensity is good — large matmuls saturate tensor cores.

**Decode (autoregressive generation):** one token at a time. Must reload all weights from HBM **every step**. Memory-bandwidth-bound.

For the KV cache:
```
arithmetic_intensity ≈ 1 / (n / (h·d_h) + 1/B)
```

Where `n` = sequence length, `h` = num heads, `d_h` = head dim, `B` = batch. The `n / (h·d_h)` term explodes for long context.

### 9.2 MQA — Multi-Query Attention

Share **one** K and V across all query heads. Drastically shrinks KV cache.

- KV cache size: `2 · n · num_kv_heads · head_dim · n_layers · bytes` → with `num_kv_heads = 1`, the smallest possible.
- Quality cost: real. Significant degradation on hard tasks.

Used by: PaLM, Falcon.

### 9.3 GQA — Grouped Query Attention

The sweet spot. Use `g` groups of K/V heads (typically 8), each shared across `num_heads / g` query heads. Llama 2 70B uses GQA-8.

- KV cache: shrinks by `num_heads / g`.
- Quality: nearly identical to full MHA.
- Now the universal default for production LLMs.

### 9.4 MLA — Multi-head Latent Attention (DeepSeek V2/V3)

Project Q, K, V into a low-rank latent space. Compute attention in that latent space. Project back.

- KV cache: ~10× smaller than MHA at near-equal quality.
- Most aggressive compression that still trains well.
- Specific to DeepSeek family for now; others may adopt.

### 9.5 The decision

| Inference budget | Choice |
|---|---|
| Don't care about cost | MHA |
| Production default | **GQA-8** |
| Memory-extreme (very long context) | MLA |
| Most aggressive compression | MQA (with quality compromise) |

### Hook

> "Decode is memory-bandwidth-bound; KV cache is what you have to shrink. MHA → MQA → GQA → MLA in order of compression and complexity. GQA is the production default; MLA when you're memory-desperate."

---

## 10. Long-Context — alternating local + global

### The pattern

The 2024–2025 frontier for handling 128K+ context **without** going to state-space models: **alternate** local attention (sliding window) and full attention.

**Cohere Command A** (the recent reviver):
```
Layers 1–3: sliding window attention (window = 4096)
Layer 4:    full attention
Layers 5–7: sliding window
Layer 8:    full attention
... (every 4th layer is full)
```

Local layers handle the bulk. Full-attention layers periodically aggregate global structure. Inference cost stays bounded; long-range dependencies still get captured.

Adopted by: **Llama 4, Gemma 3, Qwen 3.5**, and others.

### Embedding twist

Some implementations use **NoPE** (no position embedding) on local-attention layers — the layers attend to a small enough window that pure causal masking suffices. Other implementations keep RoPE everywhere.

### Hook

> "Alternate sliding-window and full attention every N layers. Bounds inference cost; preserves long-range modeling. The 2024–2026 default for non-SSM long context."

---

## 11. The Big Convergence Table

What modern open LLMs have settled on:

| Choice | Convention | Notable exceptions |
|---|---|---|
| Layer norm placement | Pre-norm OR post-norm-outside-residual | Some double-norm |
| Layer norm type | RMSNorm | A few still use LayerNorm |
| Bias terms | None (drop everywhere) | Vanilla transformer keeps them |
| Activation | SwiGLU (Llama line), GeGLU (Gemma/T5 line) | GPT-3 (GELU), Nemotron (squared ReLU) |
| Block structure | Serial | GPT-J, PaLM, Cohere (parallel — falling out) |
| Position embedding | RoPE | Some recent NoPE on local layers |
| FFN ratio | 2.67× (GLU) or 3.5× (Llama) | T5 used 64×, retracted in v1.1 |
| Head dim × num heads | `= d_model` | Few exceptions |
| Aspect ratio | ~100 d_model / n_layers | Forgiving wide basin |
| Vocab size | 128K (multilingual default) | 30K monolingual |
| Dropout | None | Tiny in some |
| Weight decay | ~0.1 | Universal |
| Output softmax stabilizer | z-loss | Optional |
| Attention softmax stabilizer | QK-Norm | Universal in 2025 |
| Logit soft-capping | Gemma only | Costs quality |
| KV-attention pattern | GQA-8 | MQA (Falcon), MLA (DeepSeek) |
| Long context | Sliding-window + full alternating | Pure full (older), SSM hybrids (Qwen 3.5) |

---

## 12. Senior Signals (interview takeaways)

- **You distinguish wide-basin from narrow-basin hyperparameters.** Most knobs are wide; stability tricks are narrow.
- **You name choices by source.** "Llama family uses SwiGLU and the 3.5 ratio; Gemma uses GeGLU and logit soft-capping."
- **You have a stability vocabulary**: pre-norm, RMS, QK-Norm, z-loss, logit soft-cap, "sprinkle norms."
- **You know RoPE's geometric intuition**: rotate vector pairs by position, inner product is invariant under absolute shift.
- **You know GQA is the default, MLA is the cutting edge.**
- **You can read a model card in 30 seconds** by checking these 7 axes: norm placement, norm type, activation, position embedding, FFN ratio, attention variant, vocab.
- **You don't get warmup-vs-no-warmup wrong**: warmup is still standard even with pre-norm.
- **You separate FLOPs from runtime** in your hardware reasoning (LayerNorm at 0.17% FLOPs but 25% runtime).
- **You frame weight decay as optimization not regularization.**

---

## 13. The 30-Second Oral Pitches

### "What's the modern transformer?"

> Pre-norm RMSNorm wrapped around SwiGLU MLP and GQA-8 attention with RoPE position embeddings, no biases, vocab around 128K, FFN ratio 2.67–3.5×, aspect ratio ~100. Cosine LR with weight decay 0.1. QK-Norm and optional z-loss for stability. Long context via alternating sliding-window and full attention layers. That's basically Llama 3, Qwen 3, Gemma 3, Mistral.

### "Why pre-norm?"

> Keeps the residual stream clean — gradients propagate straight through without normalization at every layer. Fixes the deep-network training instability of the original transformer. Universal except in OPT-350M (and OPT was a mess).

### "Why RMSNorm?"

> LayerNorm is memory-bandwidth-bound. RMSNorm drops the mean centering and bias — same expressiveness in practice, ~25% runtime savings on small models because of memory traffic reduction. Free systems win.

### "Why SwiGLU?"

> Gated linear units consistently outperform non-gated ones at parameter-matched comparison (Shazeer 2020 with replicate error bars; Narang 2020 controlled). SwiGLU is the Llama line; GeGLU is the Gemma line. Both work.

### "Why RoPE?"

> Pure relative position via 2D rotation in vector pairs. Inner-product invariant to absolute shifts. Extends naturally to arbitrary length with frequency tuning (NTK, YaRN). Replaced sinusoidal/absolute/relative attention bias post-2023.

### "Why GQA over MHA?"

> Decode is memory-bandwidth-bound, dominated by KV cache reads. MHA's KV cache scales with all heads. GQA shares K/V across groups of query heads (typically 8), shrinking KV cache by `h/g`× with negligible quality loss. MQA (full sharing) goes further but costs quality. MLA (DeepSeek) is even more aggressive via low-rank latent factorization.

### "Why is weight decay still used?"

> Despite no overfitting in single-pass pretraining, weight decay improves final loss when combined with learning-rate decay. The mechanism is optimization-dynamics, not regularization — weight decay + LR decay finds smoother minima.

---

## 14. Interview Grill — 60 questions

### Layer norm and bias (Q1–10)
1. Where do modern LLMs put LayerNorm? Why?
2. Pre-norm vs post-norm-inside-residual — which is broken?
3. Pre-norm vs post-norm-outside-residual — both viable?
4. What's "double norm"?
5. Why doesn't the residual stream like having LayerNorms in it?
6. Why RMSNorm vs LayerNorm?
7. What's the runtime impact of LayerNorm even at 0.17% FLOPs?
8. Why drop bias terms?
9. What's "sprinkle layer norms" and when do you do it?
10. Which model used post-norm-inside-residual and was a mess?

### Activations (Q11–18)
11. What's a Gated Linear Unit?
12. SwiGLU vs GeGLU?
13. Why the 2/3 correction in FFN dim?
14. What's Llama's 3.5× ratio about?
15. Did Shazeer's GLU paper have error bars?
16. Which model used squared ReLU?
17. ReLU → GELU → Swish — what changes?
18. Why did GLUs win over non-gated?

### Parallel vs serial blocks (Q19–22)
19. What's a parallel transformer block?
20. Which models used it?
21. Why did it fall out of favor?
22. What's the systems argument for parallel?

### Position embeddings (Q23–32)
23. State the relative-position constraint RoPE satisfies.
24. Sketch the 2D RoPE rotation.
25. How does RoPE generalize to D dimensions?
26. Why low frequencies and high frequencies in different pairs?
27. Which models use RoPE today?
28. What's NoPE?
29. What's Pi-RoPE / proportional RoPE?
30. Sinusoidal vs absolute vs relative-bias vs RoPE — which is which?
31. Where in the attention layer is RoPE applied?
32. Is RoPE applied to V?

### Hyperparameters (Q33–42)
33. FFN ratio default? GLU correction?
34. T5's anomaly?
35. Head dim × num heads?
36. Aspect ratio?
37. What goes wrong if you go too deep?
38. What goes wrong if you go too wide?
39. What did Kaplan 2020 show about FFN ratio?
40. Vocab size: monolingual vs multilingual?
41. Why does bigger model = bigger usable vocab?
42. Multimodal vocab implications?

### Regularization (Q43–46)
43. Why is dropout rare in LLM pretraining?
44. Why is weight decay still common?
45. How does weight decay interact with LR decay?
46. What's the standard weight decay value?

### Stability (Q47–52)
47. What's z-loss?
48. What's QK-Norm and what does it stabilize?
49. What's logit soft-capping?
50. NVIDIA's comparison: QK-Norm vs soft-cap?
51. Where are the two softmaxes in an LLM?
52. What's the "sprinkle norms" heuristic?

### Attention variants (Q53–60)
53. Why is decode memory-bound?
54. Sketch arithmetic intensity for decode.
55. MHA → MQA → GQA: what changes?
56. Which models use GQA-8?
57. What's MLA?
58. Sliding-window + full alternation: what's the pattern?
59. Which models use it? Window size?
60. NoPE on local layers — why?

---

## 15. Drill plan

- **Day 1:** Read this chapter end-to-end. Note any unfamiliar concept.
- **Day 2:** Drill the 30-second oral pitches in §13 from memory.
- **Day 3:** Drill the convergence table (§11) — be able to fill it from memory.
- **Day 4:** RoPE geometry — be able to draw it on a whiteboard.
- **Day 5:** Stability tricks (§8) — write each in pseudocode.
- **Day 6:** Attention variants (§9) with KV cache math.
- **Day 7:** Mock interview — 10 random questions from §14, 60-second answers each.

---

## Single sentence to remember

**Modern LLM = pre-norm RMSNorm + SwiGLU + GQA-8 + RoPE + no bias + ~128K vocab + ~100 aspect ratio + AdamW with weight decay 0.1 and cosine LR + QK-Norm for stability + alternating sliding-window for long context. Differences across labs are micro-tuning of these axes; the original transformer got most of it right.**

---

## Source

Distilled from Tatsu Hashimoto's CS336 *Architecture and Hyperparameters* lecture (Stanford Language Modeling From Scratch, 2025), with cross-references to canonical papers (Vaswani 2017, Shazeer 2020 GLU Variants, Su 2021 RoPE, Ainslie 2023 GQA, DeepSeek-V2 MLA, Salazar & Yen on pre-norm, Kaplan 2020 scaling laws, Narang 2020 architecture comparisons) and the open-model cards (Llama 2/3/4, Qwen 2.5/3, Gemma 2/3/4, Mistral, OLMo, DeepSeek V2/V3).
