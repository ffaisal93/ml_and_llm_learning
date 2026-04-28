# Attention Mechanisms: A Frontier-Lab Interview Deep Dive

> **Why this exists.** "Attention" is not one mechanism — it's a family. Vanilla MHA is just the starting point. Modern LLMs use MQA/GQA/MLA for KV memory, sliding window for long context, sparse and linear variants for sub-quadratic compute, and FlashAttention for I/O efficiency. This document covers the full taxonomy with the math, trade-offs, and interview gotchas. For the basic scaled-dot-product derivation, see `04_transformers/TRANSFORMERS_DEEP_DIVE.md`.

---

## 1. The taxonomy

Modern attention variants live on a few orthogonal axes:

| Axis | Options |
|---|---|
| Direction | Bidirectional / causal (autoregressive) / cross |
| Sparsity | Full / sliding window / local-global / random / learned |
| KV sharing | MHA / MQA / GQA / MLA |
| Compute pattern | Quadratic / linear / sub-quadratic with kernel approx |
| Implementation | Standard / FlashAttention / paged / fused |

A specific attention mechanism is a point in this space. "MHA + causal + full + standard" = vanilla GPT attention. "GQA + causal + sliding window + FlashAttention" = Mistral-style. Frontier interviews probe both the individual variants and how they combine.

---

## 2. Multi-Head Attention (MHA): the baseline

Already covered in `04_transformers/TRANSFORMERS_DEEP_DIVE.md`. One-line recap, for $h$ heads of dim $d_h = d/h$:

$$
\text{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i, \qquad \text{Output} = \mathrm{concat}(\text{heads}) \cdot W_O
$$

The key memory cost: each layer caches $2 \cdot h \cdot d_h \cdot N = 2 \cdot d \cdot N$ bytes per sequence per layer. For long contexts and large models, this dominates.

---

## 3. Multi-Query Attention (MQA, Shazeer 2019)

### The change

All $h$ heads share **a single $K$ projection and a single $V$ projection**. Only $Q$ is per-head.

$$
\text{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K^\top}{\sqrt{d_h}}\right) V \qquad (K, V \text{ shared across heads, dim} = d_h)
$$

### KV cache savings

Instead of $2 \cdot h \cdot d_h \cdot N$ bytes per layer per sequence, you get $2 \cdot d_h \cdot N$. **Factor of $h$ reduction in KV memory** — typically 8–64×.

### Quality cost

Empirically, ~1–2% degradation on benchmarks for moderate-size models. Significant for some specialized tasks but acceptable for most.

### Use cases

PaLM-1 used MQA. Some smaller models still do. Mostly superseded by GQA, which gets most of the savings with less quality loss.

---

## 4. Grouped-Query Attention (GQA, Ainslie et al. 2023)

### The change

Compromise between MHA and MQA: $n_{\text{kv\_heads}}$ shared groups, where each group's $K/V$ is shared by $g = h / n_{\text{kv\_heads}}$ query heads.

For group $j$ (with $g$ query heads sharing $K_j, V_j$), for each query head $i$ in group $j$:

$$
\text{head}_i = \mathrm{softmax}\!\left(\frac{Q_i K_j^\top}{\sqrt{d_h}}\right) V_j
$$

### KV cache savings

Factor of $g = h / n_{\text{kv\_heads}}$. Typical: $n_{\text{kv\_heads}} = 8$, $h = 64$ → 8× reduction.

### Quality cost

Almost zero with $n_{\text{kv\_heads}} \geq 8$. The original GQA paper and subsequent ablations show GQA-8 essentially matches MHA quality.

### Use cases

LLaMA-2 70B (8 KV heads vs 64 query heads), LLaMA-3, Qwen, Mistral. **The de facto standard for modern LLMs.**

---

## 5. Multi-Latent Attention (MLA, DeepSeek-V2)

### The change

Compress KV into a low-rank latent before caching, then re-expand on use:

$$
c_t = x_t \cdot W_{DKV} \in \mathbb{R}^{d_c} \qquad (d_c \ll d, \text{e.g. } d_c = 512 \text{ vs } d = 4096)
$$

$$
K_t = c_t \cdot W_{UK} \in \mathbb{R}^d, \qquad V_t = c_t \cdot W_{UV} \in \mathbb{R}^d
$$

Only the compressed $c_t$ is cached; full $K, V$ are reconstructed at attention time.

### KV cache savings

Compressed dimension $d_c \approx d/8$ typical → 8× reduction. Comparable to GQA-8 or better, but with potentially less quality cost.

### Trade-off

Extra compute at attention time (re-expand $K, V$ from $c$). For decode (where memory bandwidth dominates anyway), this is a good trade. For prefill, the extra matmul cost matters.

### Use cases

DeepSeek-V2 and DeepSeek-V3. Reportedly state-of-the-art KV memory efficiency.

---

## 6. Cross-Attention

The mechanism in encoder-decoder models. Decoder queries attend to encoder outputs.

$$
Q = \text{decoder\_state} \cdot W_Q, \qquad K, V = \text{encoder\_output} \cdot W_K, W_V
$$

$$
\text{output} = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$

### Asymmetries

- $Q$ comes from decoder (length $M$ tokens during generation).
- $K, V$ come from encoder (length $N$ input tokens, fixed throughout decode).
- The decoder cross-attention KV cache for the encoder output is computed **once** per request and reused for every decode step. Substantial cost saving.

### Why pure decoder LLMs don't have cross-attention

They handle "looking at" inputs by putting the input in the context window. The model uses self-attention over $[\text{input}, \text{output\_so\_far}]$. Architecturally simpler; functionally similar.

### Used in

T5, BART, original transformer, multimodal models (image encoder + language decoder), retrieval-augmented decoders.

---

## 7. Causal (autoregressive) attention

The defining trick that makes decoder-only LLMs possible. Add a triangular mask to the attention scores:

$$
M_{i,j} = \begin{cases} 0 & j \leq i \\ -\infty & j > i \end{cases}
$$

$$
\text{scores} = \frac{Q K^\top}{\sqrt{d_k}} + M
$$

$$
\text{weights} = \mathrm{softmax}(\text{scores}) \qquad \text{(masked positions become 0)}
$$

### Why this works during training

The model sees the entire sequence in parallel. The causal mask ensures position $i$ can only "see" positions $\leq i$ despite being computed alongside future positions. Loss is computed at every position simultaneously — the model is trained on $N$ next-token-prediction tasks per sequence.

### Why this works during inference

The mask is implicit during autoregressive decode: you only compute attention between the new token's query and all previous keys/values. No future positions exist. The KV cache stores $K, V$ from previous steps; the new query attends over them.

---

## 8. Sliding window attention (Mistral, Longformer)

### The idea

Each position attends only to the previous $W$ tokens (window size $W$, e.g. 4096). Not the full prefix.

$$
M_{i,j} = \begin{cases} 0 & i - W < j \leq i \\ -\infty & \text{otherwise} \end{cases}
$$

### Compute and memory

Quadratic cost reduces to $O(N \cdot W)$ from $O(N^2)$. KV cache also bounded: only the most recent $W$ positions per sequence per layer.

### Capacity argument

With $L$ layers and window $W$, the receptive field of the top layer is $L \cdot W$ (each layer can "look" back $W$, and stacking gives $L \cdot W$ effective span). Mistral 7B has $W = 4096$, $L = 32$, so theoretical receptive field ~131K tokens despite each layer only seeing 4K.

### Failure modes

- Information far outside the window can't reach later layers efficiently.
- Sliding window models often paired with "global tokens" (a few positions that attend everywhere) to handle exceptional cases.
- Quality drops on tasks requiring genuinely long-range reasoning.

### Use cases

Mistral 7B, Longformer, Sparse Transformers.

---

## 9. Local-global attention (Longformer, BigBird)

Extends sliding window with **global tokens**: specific positions (often `[CLS]` or task-specific markers) that attend to and are attended by all other positions.

- For local positions: sliding window attention.
- For global positions: full attention to all positions, and attended by all positions.

### Compute

$O(N \cdot W + N \cdot g)$ where $g$ is the number of global tokens. Linear in $N$.

### Why global tokens help

They aggregate information across the whole sequence — short-circuit for the receptive-field argument that takes many sliding-window layers.

### Use cases

Longformer (4096 window + small set of global tokens), BigBird (window + global + random sparse).

---

## 10. Sparse attention (Sparse Transformer, Reformer)

Different sparsity patterns can give sub-quadratic compute:

### Strided / dilated patterns

Attend to positions $i, i-1, i-2, i-4, i-8, i-16, \ldots$. Logarithmic receptive field expansion. Used in Sparse Transformer (Child et al. 2019).

### LSH-based

Reformer (Kitaev et al. 2020): hash queries and keys; attend only within the same hash bucket. Approximates full attention with $O(N \log N)$ compute.

### Random + structured

BigBird (Zaheer et al. 2020): random attention pattern + sliding window + global tokens. Theoretically retains expressiveness of full attention; empirically reasonable.

### Trade-offs

All of these reduce compute and memory at some quality cost. They're more popular in research than production — modern long-context production models tend to use full attention with efficient kernels (FlashAttention) and KV memory tricks (paged, quantized) rather than sparsifying.

---

## 11. Linear attention

The full softmax attention is $O(N^2)$. Linear attention removes the softmax to get $O(N)$:

### The setup

Replace $\mathrm{softmax}(Q K^\top) V$ with a kernel approximation:

$$
\mathrm{attention}(Q, K, V) = \frac{\phi(Q) \cdot (\phi(K)^\top V)}{\phi(Q) \cdot \phi(K)^\top \mathbf{1}}
$$

where $\phi$ is some feature map. Order of operations matters: compute $\phi(K)^\top V$ first (size $d \times d$), then $\phi(Q) \cdot \cdots$. The result is $O(N \cdot d^2)$ — linear in $N$.

### Common choices for $\phi$

- **Performer** (Choromanski et al. 2020): random Fourier features approximating softmax kernel.
- **Linear Transformer** (Katharopoulos et al. 2020): $\phi(x) = \mathrm{elu}(x) + 1$.
- **RetNet, RWKV**: discrete kernel functions optimized for autoregressive generation.

### The autoregressive form

Linear attention naturally factorizes over time. Track two running quantities — a $d \times d$ value-state and a $d$-dim normalizer:

$$
S_t = S_{t-1} + \phi(K_t) V_t^\top, \qquad z_t = z_{t-1} + \phi(K_t)
$$

$$
\text{output}_t = \frac{\phi(Q_t)^\top S_t}{\phi(Q_t)^\top z_t}
$$

This is a **recurrent state update** — $S_t$ is the matrix-valued state, $z_t$ the running key-sum normalizer, both updated incrementally. So linear attention models can decode in $O(d^2)$ per step regardless of context length, much like RNNs. (Some variants drop the normalizer for speed at minor quality cost.)

### Quality trade-off

Linear attention models are usually weaker than softmax attention at the same scale, especially for in-context learning and recall. Whether they catch up at frontier scale is an open question.

### State Space Models (SSMs)

Mamba, S4, etc. are essentially structured linear attention with specific kernel choices and parallelizable training. See `42_state_space_models/`.

---

## 12. FlashAttention recap

Already covered in `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`. One-paragraph recap:

> FlashAttention (Dao et al. 2022) computes standard softmax attention with the same FLOPs but radically less memory traffic by tiling $Q, K, V$ to fit in SRAM and using online softmax to compute partial softmax statistics block-by-block. 2–4× wall-clock speedup, especially at long sequences. FA-2 (2023) and FA-3 (2024) extend with better parallelization and Hopper-native primitives.

---

## 13. Attention head analyses and interpretability

### Empirical findings

Many heads are interpretable as specialized circuits:

- **Induction heads** (Olsson et al. 2022, "In-context Learning and Induction Heads"): heads that copy a token from earlier in the context when a similar prefix recurs. Critical for in-context learning.
- **Previous token heads**: attend to position $t-1$ from position $t$. Used in counting, parsing.
- **Duplicate heads**: identify repeated tokens.
- **Coreference heads**: attend from a pronoun to its antecedent.

### Why this matters

For interview questions about how transformers learn / generalize, interpretability findings are increasingly relevant. Frontier-lab AS interviews often ask about induction heads specifically.

### Head pruning

Many heads can be removed without loss (Voita et al., Michel et al.). Suggests redundancy. In production, this rarely matters; the attention is small relative to FFN. In research, head pruning gives insights into which heads are "essential."

---

## 14. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why does attention scale quadratically?" | The $Q K^\top$ matrix is $N \times N$ — every query against every key. Compute and memory both $O(N^2)$. |
| "How does FlashAttention reduce complexity?" | It doesn't. Same FLOPs. It reduces **memory access** by tiling and online softmax. |
| "MQA vs GQA?" | MQA: one $K/V$ across all heads. GQA: groups share $K/V$. GQA-8 is the modern default; MQA is more aggressive. |
| "Why does sliding window still capture long range?" | Stacking: with $L$ layers and window $W$, top layer has receptive field $L \cdot W$. Plus global tokens / first token sinks. |
| "Linear attention vs softmax?" | Linear is $O(N)$ but typically weaker on recall. Trades compute for quality. |
| "Why doesn't pure decoder need cross-attention?" | Inputs are placed in context; self-attention handles everything. |
| "What's an induction head?" | Head that copies a token after a prefix matches earlier. Mechanism for in-context learning. |
| "Cross-attention KV cost?" | Encoder $KV$ computed once per request, reused for every decode step. Cheap relative to self-attention KV. |

---

## 15. Drill plan

1. Master the MHA → MQA → GQA → MLA hierarchy with KV memory savings for each.
2. Understand sliding window's receptive-field argument for $L \cdot W$.
3. Know one linear-attention example (Performer or Mamba) at a sketchy level.
4. Cite induction heads as the canonical interpretability finding.
5. Drill `INTERVIEW_GRILL.md`.

---

## 16. Further reading

- Vaswani et al., "Attention is All You Need" (2017).
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA, 2019).
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023).
- DeepSeek, "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" (MLA, 2024).
- Beltagy et al., "Longformer: The Long-Document Transformer" (2020).
- Zaheer et al., "Big Bird: Transformers for Longer Sequences" (2020).
- Child et al., "Generating Long Sequences with Sparse Transformers" (2019).
- Choromanski et al., "Rethinking Attention with Performers" (2020).
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020).
- Dao et al., "FlashAttention" series (2022/23/24).
- Olsson et al., "In-context Learning and Induction Heads" (2022).
- Mistral, "Mistral 7B" (2023) — sliding window in production.
