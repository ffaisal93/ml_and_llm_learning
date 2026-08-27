# Transformers: A Frontier-Lab Interview Deep Dive

> **Why this exists.** The transformer is the architecture every modern LLM is built on. Interviewers probe it at every level: derive the math, justify every design choice, explain what happens if any component is removed. This document covers the architecture from first principles — not as a list of components, but as a sequence of design decisions, each motivated, each derivable on a whiteboard.

---

## 1. The single sentence

> A transformer is a stack of identical blocks, where each block does **token-wise mixing** (attention) followed by **per-token computation** (FFN), with **residual connections** and **layer normalization** holding the whole thing together.

That's it. Every other detail elaborates one of those four ideas: token mixing (attention), per-token compute (FFN), residual flow, normalization placement. If you can speak that sentence cleanly and then expand each phrase, you can answer most "explain transformers" questions.

> **Saying it out loud.** If I get sixty seconds to explain a transformer, this is the sentence: it's a stack of identical blocks, and each block mixes information *across* tokens with attention and then does computation *per* token with a feed-forward net, held together by residual connections and normalization. Everything else is elaboration on one of those four things. The reason that framing is worth memorizing is that it turns 'explain transformers' from a recall problem into an expansion problem — you say the sentence, then unpack whichever phrase they push on.

---

## 2. Why this architecture won

The 2017 transformer paper (Vaswani et al., "Attention is All You Need") replaced recurrent networks for sequence tasks. The reasons are now obvious in hindsight:

### Parallelism over sequence length

RNNs process tokens sequentially: token $t$ requires hidden state from token $t-1$. Cannot parallelize across the sequence dimension during training. Transformers compute all positions in parallel — same FLOPs but vastly better hardware utilization on GPUs. **This is the single biggest reason transformers beat RNNs at scale.**

### Constant-depth gradient flow

In RNNs, the gradient between positions $t$ and $t+k$ flows through $k$ matrix multiplications, leading to vanishing/exploding gradients (the central problem LSTMs partially solved). In transformers, every position can attend to every other in **one** layer of operations. Gradient flows through the residual stream in roughly constant depth regardless of sequence length.

### Long-range dependencies for free

RNN attention to a token 1000 positions back requires gradient flow through 1000 hidden states. Transformer attention to that token is one matrix multiply. The "expressivity per parameter" is far higher for long-range patterns.

### Inductive bias = none (which is good at scale)

RNNs encode "sequential processing" as an inductive bias. CNNs encode "local connectivity." Transformers encode almost nothing — pure attention is permutation-equivariant. With abundant data, this absence of bias is an advantage: the model learns the right inductive biases from data rather than having them imposed. With small data, transformers are sample-inefficient compared to CNN/RNN baselines for this reason.

> **Saying it out loud.** Transformers beat RNNs mostly on parallelism, and I'd lead with that rather than with attention being clever. An RNN has to compute token 500 after token 499, so training time scales with sequence length regardless of how many GPUs you own; a transformer processes every position at once. Second is gradient flow — reaching a token a thousand positions back is one matrix multiply instead of a thousand sequential hops, so nothing vanishes. And third, transformers bake in almost no assumptions, which is a liability on small data and a huge advantage once you have enough of it to learn better structure than we could design.

---

## 3. Self-attention from first principles

*In plain terms:* this section builds attention from the question 'how should a token decide which other tokens to listen to?'. Every token emits three vectors — a query (what I'm looking for), a key (what I'm about), and a value (what I'll hand over) — and the matching between queries and keys decides the mixing weights. The matrices below are just three learned linear projections plus a softmax.

The fundamental operation. Derive it once and you can re-derive any variant.

### What we want

For each position $i$, produce a new representation that's a weighted combination of all positions, where the weights depend on **content** (not position).

### Setup

Input: $X \in \mathbb{R}^{N \times d}$ — $N$ tokens, each a $d$-dim vector.

Project into three views:

$$
Q = X W_Q \in \mathbb{R}^{N \times d_k} \qquad \text{(queries: ``what am I looking for?'')}
$$

$$
K = X W_K \in \mathbb{R}^{N \times d_k} \qquad \text{(keys: ``what am I about?'')}
$$

$$
V = X W_V \in \mathbb{R}^{N \times d_v} \qquad \text{(values: ``what info do I provide?'')}
$$

$W_Q, W_K, W_V$ are learned projection matrices. The split into $Q/K/V$ is a deliberate inductive choice: query and key live in a "compatibility" space; value lives in a "content" space; they need not be the same dimension or have the same role.

### Compute attention

$$
\text{scores} = \frac{Q K^\top}{\sqrt{d_k}} \in \mathbb{R}^{N \times N} \qquad \text{(query-key compatibility)}
$$

$$
\text{weights} = \mathrm{softmax}(\text{scores}) \in \mathbb{R}^{N \times N} \qquad \text{(row-wise)}
$$

$$
\text{output} = \text{weights} \cdot V \in \mathbb{R}^{N \times d_v}
$$

Each row of weights is a probability distribution over the $N$ keys, telling you how much each value contributes to that position's output.

### Why scale by $\sqrt{d_k}$?

Without scaling, $Q K^\top$ entries grow with $d_k$. For random $Q, K$ with entries $\sim \mathcal{N}(0, 1)$, the dot product $q \cdot k$ of two $d_k$-dimensional vectors has variance $d_k$. So entries of $Q K^\top$ are typically $O(\sqrt{d_k})$ in magnitude.

Large softmax inputs $\to$ softmax saturates $\to$ gradients vanish. (Concretely: the Jacobian of softmax is $\partial p_i / \partial z_j = p_i(\delta_{ij} - p_j)$. When the distribution is one-hot — one $p$ near 1, rest near 0 — every entry of the Jacobian is near 0.)

Dividing by $\sqrt{d_k}$ keeps the variance of the scores $O(1)$ regardless of $d_k$, so softmax stays in its linear regime and gradients flow.

**Interview trap.** If asked "why $\sqrt{d_k}$ specifically?", the answer is the variance argument above — divide by the standard deviation to keep $O(1)$ magnitude. Not "because the paper said so."

> **Saying it out loud.** The scaling factor is there to stop the softmax from saturating. A dot product of two $d_k$-dimensional vectors is a sum of $d_k$ terms, so its typical magnitude grows like $\sqrt{d_k}$ — at $d_k = 128$ you're feeding numbers in the tens into a softmax, which makes it effectively a one-hot. And once softmax is one-hot, its Jacobian is essentially all zeros, so no gradient reaches the queries or keys and attention never learns. Dividing by the standard deviation, which is $\sqrt{d_k}$, keeps the score variance at $O(1)$ no matter how wide the head. The point to make is that it's a variance argument, not a convention.

### Why softmax?

Three properties:

1. **Non-negative weights** that **sum to 1** (probability simplex). Average-of-values is well-defined.
2. **Differentiable everywhere** with non-trivial gradients (unlike argmax).
3. **Sharpens the highest-scoring keys** — softmax with temperature 1 is "soft argmax." Exponential makes the largest score dominate.

Alternatives like sigmoid gate per key, linear normalization, or polynomial kernels exist but softmax is the empirical winner for the task.

### Computational complexity

$$
Q K^\top: O(N^2 \cdot d_k) \qquad \text{(the famous quadratic cost)}
$$

$$
\text{softmax} \cdot V: O(N^2 \cdot d_v)
$$

$$
\text{total}: O(N^2 \cdot d)
$$

This $N^2$ is **the central obstacle to long-context transformers**. FlashAttention reduces the memory access (not the FLOPs); linear attention variants attempt to reduce the FLOPs. Mostly we just pay the $N^2$ and use clever tricks to fit longer sequences.

> **Saying it out loud.** Here's the whole mechanism in one breath: each token asks a question, every token advertises what it has, you match the questions against the advertisements to get weights, and each token's output is the weighted average of everyone's content. The softmax makes those weights a proper distribution, so the output is a convex combination and can't blow up. And the cost of doing this is that you compare every token to every other token — an $N \times N$ matrix, which is $O(N^2 d)$ compute and $O(N^2)$ memory. That quadratic term is the single fact that drives long-context research.

---

## 4. Multi-head attention

Single-head attention has all of $Q, K, V$ living in one shared space. Empirically, multiple heads attending to different things in parallel work better.

### The construction

Split $d$ into $h$ heads of size $d_h = d / h$:

$$
\begin{aligned}
Q_i &= X W_Q^{(i)} \in \mathbb{R}^{N \times d_h} \\
K_i &= X W_K^{(i)} \in \mathbb{R}^{N \times d_h} \\
V_i &= X W_V^{(i)} \in \mathbb{R}^{N \times d_h} \\
\text{head}_i &= \mathrm{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i
\end{aligned}
$$

$$
\text{Output} = \mathrm{concat}(\text{head}_1, \ldots, \text{head}_h)\, W_O \in \mathbb{R}^{N \times d}
$$

Same total parameters as one big head (give or take $W_O$), but allows the model to attend to different patterns simultaneously: one head might track syntax, another semantics, another references, etc.

### Why heads help

The key intuition: a single $d$-dim attention space must compress all "what to attend to" decisions into one similarity metric. Multi-head splits this into $h$ parallel attention spaces, each computing its own similarity. Empirically, ablations show that ~5–10 heads provide most of the benefit; very high head counts plateau.

### Common interview detail: the head dimension

In the original paper, $d = 512, h = 8, d_h = 64$. Modern LLMs often have $d = 4096+$, $h = 32+$, $d_h = 128$. The constraint that $d_h = d / h$ keeps total compute constant.

### What each head learns

Empirical analyses (Voita et al., Clark et al.) show interpretable specialization in some heads — syntactic dependency heads, coreference heads, position-tracking heads. Most heads, however, are not so cleanly interpretable. Ablating individual heads often costs surprisingly little, suggesting redundancy.

> **Saying it out loud.** Multi-head is just running several smaller attentions side by side instead of one wide one. The reason it matters is that a single softmax can only commit to one pattern — if a token needs to look at its syntactic head *and* at the previous mention of the same entity, one head has to compromise between them. Split into $h$ heads and each gets its own normalization, so they can specialize independently, at the same total parameter count. Ablations show you get most of the benefit by five to ten heads, and the honest caveat is that most heads aren't cleanly interpretable and many can be pruned.

---

## 5. The FFN block

After attention mixes information across positions, the FFN does **per-token** computation. It's just a 2-layer MLP applied independently to each position's output.

$$
\mathrm{FFN}(x) = W_2 \cdot \mathrm{activation}(W_1 \cdot x + b_1) + b_2
$$

Standard activation: ReLU (original), GELU (BERT, GPT-2), SwiGLU (modern LLMs).

### The 4× expansion

$W_1: \mathbb{R}^{d \times 4d}, W_2: \mathbb{R}^{4d \times d}$ — the inner dimension is $4 \times$ the model dimension. Why 4? Empirical ablation in the original paper. Some recent models use $8/3$ (especially with SwiGLU), but $4 \times$ is the iconic ratio.

### Why FFN is necessary

Without it, transformers are just stacks of attention. Attention is **linear** in the values: $\text{output} = \mathrm{softmax}(Q K^\top) V$ — the weights depend on inputs but the value mixing is linear. Stacking linear operations gives more linear operations. The FFN is where **non-linearity** enters and where **per-token feature transformation** happens.

### How the FFN is sometimes described

The FFN can be viewed as a **key-value memory** (Geva et al. 2021): $W_1$ matches the input against $4d$ "keys"; the activation gates them; $W_2$ retrieves the corresponding "values." This view explains why FFNs hold most of the model's "factual knowledge" — they're literally a learned look-up table.

### Where the parameters live

In a typical LLM, the FFN holds **2/3 to 3/4 of all parameters**. Attention parameters are $4 d^2 \cdot L$ (roughly); FFN parameters are $8 d^2 \cdot L$ ($4d$ expansion $\times$ 2 matrices). Modern FFN-heavy designs (and MoE) push this further.

> **Saying it out loud.** The FFN is where the model actually thinks about each token on its own, and where most of what it knows is stored. Attention only ever produces a weighted average of values, which is a linear operation — so without a non-linearity in between, stacking blocks would collapse into one big linear map. The FFN blows each token up 4x, applies a non-linearity, and squeezes it back, which behaves a lot like a key-value memory: the first matrix matches patterns, the second writes out the associated fact. And it's not a side dish — two-thirds or more of all the parameters live there, which is exactly why decode is memory-bound on FFN weights.

---

## 6. Residual connections and the residual stream view

*In plain terms:* a residual connection means a layer never replaces the representation, it only *adds* to it. That has two big consequences — gradients get an unobstructed path back to early layers, and you can think of the representation as a shared channel that every layer reads from and writes into. The second idea is what interpretability people call the residual stream.

Every block has the form:

$$
\begin{aligned}
x &\leftarrow x + \mathrm{Attention}(\mathrm{LayerNorm}(x)) \\
x &\leftarrow x + \mathrm{FFN}(\mathrm{LayerNorm}(x))
\end{aligned}
$$

The $x \leftarrow x + \cdots$ is a **residual connection**. The unchanged $x$ flows through the layer, and the layer **adds** to it. This single design choice has enormous consequences:

### Gradient flow

The gradient of the final output w.r.t. early-layer activations decomposes into a sum of terms, one per layer, plus an "identity" term. The identity term ensures the gradient cannot vanish through depth: even if every layer's gradient becomes tiny, the residual passes the gradient through unchanged. This is **the** reason transformers can be stacked 100+ layers deep without optimization breakdown.

### The residual stream metaphor

Modern interpretability framing (Elhage et al., "Mathematical Framework for Transformer Circuits"): think of each token's representation as a **stream** of information that flows from layer to layer. Each block reads from the stream (via LayerNorm + projection) and **writes back** what it computes (via the residual addition). Layers communicate by reading and writing to this stream.

This view is interview-relevant because: (a) it's how frontier-lab researchers actually talk about transformers, (b) it makes the role of LayerNorm clear (rescale before reading), (c) it explains the "rank one update" interpretability literature.

### Why this matters for training

In a residual stream, the magnitude of the stream grows with depth (each block adds something). Without normalization, this growth would destabilize training. The LayerNorm before each block keeps reads from the stream in a consistent magnitude.

> **Saying it out loud.** Residuals are why we can train hundred-layer models at all. Without them the gradient reaching layer one is a product of a hundred Jacobians, and a product of a hundred things slightly below one is zero — training just stalls. With the addition, there's always an identity path where the gradient passes through untouched. The modern reframing is even more useful: think of the representation as a shared bus that every block reads from and adds to, which is what lets interpretability people ask 'what did head 7 in layer 12 write into the stream?'

---

## 7. LayerNorm: pre-LN vs post-LN

The original transformer used **post-LN**: $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. The norm comes after the residual.

Modern transformers use **pre-LN**: $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$. The norm comes before the sublayer, and the residual is unnormed.

### Why pre-LN won

Post-LN has the well-known training instability that the original paper papered over with elaborate warmup schedules. The reason: in post-LN, the residual stream is renormalized after every block, which can amplify perturbations.

Pre-LN keeps the residual stream "clean" — the unnormed $x$ is what gets added. The sublayer sees a normed input but writes to the unnormed stream. This decouples the read magnitude (which the LayerNorm controls) from the write magnitude (which the sublayer controls).

Empirically: pre-LN trains stably without elaborate warmup; post-LN does not at modern scales. **Every modern LLM uses pre-LN or a close variant (e.g., RMSNorm).**

### RMSNorm

LayerNorm: $x_{\text{norm}} = (x - \mu) / \sigma$. Two normalizations: zero-mean, unit-variance.

RMSNorm: $x_{\text{norm}} = x / \mathrm{RMS}(x)$, where $\mathrm{RMS}(x) = \sqrt{\mathrm{mean}(x^2)}$. **Just unit-variance**, no mean subtraction. Cheaper (one fewer reduction) and empirically as good as LayerNorm. **LLaMA family uses RMSNorm; many modern open models follow.**

See `44_normalization/` for the full deep dive on this.

> **Saying it out loud.** Pre-LN versus post-LN is about whether you normalize the residual highway or leave it alone. The original design added the sublayer output and then normalized the sum, so the stream got squashed at every block — and any small perturbation gets re-amplified layer after layer, which is why post-LN needs elaborate warmup and still goes unstable at depth. Pre-LN normalizes only the input to the sublayer and leaves the residual untouched, so reads are controlled but the gradient path stays clean. RMSNorm is a further simplification that drops the mean-centering, saving about 30% of the cost for no measurable quality loss — which is why LLaMA, Mistral, and Gemma all use it.

---

## 8. Positional information

Pure attention is **permutation-equivariant**: shuffle the input tokens and the output is shuffled the same way. So the model has no notion of order — exactly the wrong inductive bias for language.

Positional encoding injects order information. Three eras:

### Sinusoidal (original)

Add a deterministic per-position vector $\mathrm{PE}(t)$ with sinusoids of varying frequencies:

$$
\mathrm{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad \mathrm{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)
$$

Pros: extrapolates to longer sequences than seen at training (in theory). Cons: not as good empirically at length extrapolation as later methods.

### Learned (BERT, GPT-2)

Treat position as another categorical feature; learn an embedding per position. Simple, works well within training-length range, **does not extrapolate** beyond max position seen.

### Rotary (RoPE, Su et al. 2021)

Encode position by **rotating** $Q$ and $K$ by an angle proportional to position. The dot product $Q \cdot K$ then depends on relative position (not absolute). Used in LLaMA, GPT-J, most modern open LLMs. Best-known method for length extrapolation.

### ALiBi

Add a bias to the attention scores that linearly penalizes attending to distant positions. Simple, fast, extrapolates well. Used in BLOOM and a few others.

See `14_advanced_positional_embeddings/` for the deep dive.

> **Saying it out loud.** Attention on its own has no idea what order anything is in — shuffle the tokens and the outputs shuffle right along with them, because dot products don't care where their operands sat. So you have to inject position. The original sinusoidal scheme gave every position a fixed fingerprint; learned embeddings just made a lookup table, which works fine and then falls off a cliff past the training length. RoPE won because it *rotates* the queries and keys by an angle proportional to position, so the dot product ends up depending only on the gap between two tokens — relative, not absolute, which is why it extends past training length far more gracefully.

---

## 9. Encoder, decoder, encoder-decoder

The original transformer was encoder-decoder (for translation). Modern LLMs are decoder-only. Three flavors matter:

### Encoder (BERT-style)

- Bidirectional attention: every token attends to every other (no mask).
- No autoregressive generation; outputs are contextualized representations.
- Pretrained with masked-language-modeling (predict masked tokens from context).
- Used for: classification, NER, embeddings.

### Decoder (GPT-style)

- Causal attention: token $t$ attends only to tokens $\leq t$ (lower-triangular mask).
- Autoregressive generation: emit token $t+1$ from current state, append, repeat.
- Pretrained with next-token prediction.
- Used for: open-ended generation, chat, modern LLMs.

### Encoder-decoder (T5, original Transformer)

- Encoder produces representations of source.
- Decoder generates target autoregressively, with **cross-attention** to encoder outputs.
- Used for: translation, summarization, structured generation.

**Why decoder-only won for LLMs.** Simpler architecture (one tower not two), one objective (next-token), naturally extends to in-context learning. Empirically scales better than encoder-decoder for general-purpose generation. Still: encoder models like BERT remain dominant for embedding/retrieval tasks.

### Causal mask

The causal (autoregressive) mask is a lower-triangular matrix $M$ with $0$ on and below the diagonal and $-\infty$ above:

$$
\text{attention-scores} = \frac{Q K^\top}{\sqrt{d_k}} + M
$$

The $-\infty$ entries become $0$ after softmax, ensuring position $i$ cannot attend to position $j > i$. This is the entire mechanism of "causal attention."

### Cross-attention

In encoder-decoder: the decoder's queries come from the decoder's previous layer; the keys and values come from the **encoder's** output. This is how the decoder "looks at" the source while generating the target.

$$
Q_{\text{dec}} = \text{decoder-state} \cdot W_Q
$$

$$
K_{\text{enc}}, V_{\text{enc}} = \text{encoder-output} \cdot W_K,\ W_V
$$

$$
\text{output} = \mathrm{softmax}\!\left(\frac{Q_{\text{dec}}\, K_{\text{enc}}^\top}{\sqrt{d_k}}\right) V_{\text{enc}}
$$

Pure decoder LLMs don't have cross-attention; they handle "looking at" inputs by putting them in the context window.

> **Saying it out loud.** The three flavors differ in one thing: who is allowed to see whom. An encoder is bidirectional, so every token sees the whole sequence — great for understanding, useless for generating, since you'd be reading the answer. A decoder is causal, so token $i$ sees only up to $i$, which is exactly the setup for predicting the next one. Encoder-decoder splits the job and joins them with cross-attention. Decoder-only won for LLMs because it's one tower and one objective, it trains on effectively unlimited raw text, and generation falls straight out of the training task.

---

## 10. The full block, end to end

```python
def transformer_block(x, attn, ffn, ln1, ln2):
    # Pre-LN, decoder-style
    x = x + attn(ln1(x))      # token mixing
    x = x + ffn(ln2(x))       # per-token compute
    return x

def transformer(tokens, embed, blocks, ln_final, unembed):
    x = embed(tokens) + positional_embeddings   # or RoPE applied inside attention
    for block in blocks:
        x = block(x)
    x = ln_final(x)
    logits = unembed(x)
    return logits
```

That's a transformer LLM. ~30 lines of pseudocode for the architecture; the rest of the work is in attention internals, embeddings, and training.

> **Saying it out loud.** The thing that surprises people is how little code this is. A block is two lines — add attention of the normalized input, then add the FFN of the normalized input — and the model is a loop over blocks with an embedding at the front and an unembedding at the back. That's thirty lines of pseudocode for an architecture that runs the industry. All the difficulty has moved elsewhere: efficient attention kernels, tokenization, data, and the training recipe. Being able to write it from memory on a whiteboard is table stakes, and it's a fair thing to practice cold.

---

## 11. Why transformers scale

Scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022 / Chinchilla) say loss scales as a power law in compute, parameters, and data **independently**. Transformers exhibit this scaling more cleanly than other architectures. Why?

- **Parallelism enables compute efficiency.** Doubling parameters and doubling compute scale near-linearly.
- **Attention provides flexible mixing.** The model can learn to use whatever capacity it has.
- **Residual streams enable depth.** You can stack 100+ layers without optimization breakdown.
- **Few inductive biases.** With enough data, learned biases beat imposed biases.

Chinchilla's central finding: most LLMs at the time were undertrained — for a given parameter count, they should use 20× more tokens than common practice. The Chinchilla-optimal compute-loss frontier is now standard.

> **Saying it out loud.** Transformers scale cleanly, meaning loss follows a smooth power law in compute over many orders of magnitude with no plateaus — which is what lets anyone justify a nine-figure training run from small-scale experiments. The architectural reasons are that it parallelizes so compute converts into progress efficiently, residual streams let you stack depth without optimization breaking, and the weak inductive bias means more data keeps buying you something. The number to have ready is Chinchilla's: roughly 20 tokens per parameter is compute-optimal, and most pre-2022 models were badly undertrained relative to that.

---

## 12. Training instabilities

Common interview material because frontier-lab work involves diagnosing them.

### Loss spikes

Mid-training loss explosions, sometimes recovered, sometimes not. Causes:

- A bad batch (single sequence with extreme tokens or long repetition).
- Adam $\hat v$ accumulating an outlier; the next step has a huge effective LR.
- Operating at the edge of stability (Cohen et al.).

Mitigations: gradient clipping at norm 1.0, careful warmup, smaller peak LR, occasionally rolling back to a checkpoint.

### Loss NaN at init

Often: weights initialized too large, attention scores blow up, softmax saturates. Fix: scaled initialization ($\text{std} = 0.02$ or $1/\sqrt{d}$), check forward pass magnitudes layer-by-layer.

### Loss decreasing then plateau

Often: schedule decayed too fast, or warmup too short causing residual stream miscalibration, or KV cache issue at inference (not training). Diagnostic: loss vs LR sanity check.

### Embedding collapse

All embeddings drift toward the same vector. Symptom: low embedding entropy. Cause: weight tying with insufficient regularization, or bug in init. Fix: standard init for embeddings, weight decay.

> **Saying it out loud.** The failures worth knowing are loss spikes, NaN at init, and plateaus, and each has a different signature. A NaN in the first handful of steps is almost always the learning rate or an fp16 overflow in the forward pass, not anything subtle. A spike in the low hundreds of steps is the classic end-of-warmup signature — the peak LR is too high or the ramp was too short. And a spike mid-run is usually one pathological batch or Adam's second moment absorbing an outlier, which gradient clipping at norm 1.0 handles. The judgment call that separates a good answer: don't reflexively lower the LR for occasional spikes, because at a good learning rate you're supposed to be riding the edge of stability.

---

## 13. Comparisons (interview-grade)

### Transformer vs RNN

| Property | RNN | Transformer |
|---|---|---|
| Parallelism (training) | No (sequential) | Yes (per-position) |
| Long-range deps | Poor (vanishing grad) | Strong (single attention layer) |
| Compute per step | $O(d^2)$ | $O(d^2 + N \cdot d)$ per token |
| Memory per step | $O(d)$ | $O(N \cdot d)$ (KV cache) |
| Interpretability | Hidden state evolution | Attention patterns |
| Inductive bias | Sequential | Permutation-equivariant + position |

### Transformer vs CNN

CNNs assume local connectivity and translation equivariance. Good inductive biases for vision; mediocre for language (where dependencies are non-local). Transformers have weaker inductive bias, scale better with data. ViT (Vision Transformer) showed transformers eventually dominate vision too at sufficient scale.

### Transformer vs State Space Model (Mamba/S4)

SSMs reintroduce sequential structure but in a way that's parallelizable on GPU (via the convolutional view). They give $O(N)$ instead of $O(N^2)$ complexity. Empirically competitive at smaller scale; whether they match transformers at frontier scale is an active research question. See `42_state_space_models/`.

> **Saying it out loud.** Against an RNN, the transformer trades memory for parallelism — constant-memory sequential decoding for a KV cache that grows with context but trains in one parallel pass. Against a CNN, it trades built-in assumptions for scale: convolutions know that nearby pixels relate, transformers have to learn it, which loses at small data and wins at large. Against an SSM like Mamba, it's the same tradeoff in a new place — a fixed-size recurrent state gives you $O(N)$ and constant decode memory, but compressing the past into a bounded vector costs you exact recall. That's why the practical answer right now is hybrids: a few attention layers for recall, cheaper layers for everything else.

---

## 14. The 12 most-asked transformer interview questions

(Brief; full grilling in [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).)

1. **Why scale by $\sqrt{d_k}$?** Variance of dot products grows with $d_k$; scaling keeps softmax in linear regime.
2. **Why softmax?** Non-negative weights summing to 1, differentiable, sharpens dominant scores.
3. **Why multi-head?** Multiple parallel attention spaces for different patterns; ablation shows ~5–10 heads suffice.
4. **What does the FFN do?** Per-token non-linear computation; key-value memory view; holds 2/3+ of parameters.
5. **Why pre-LN over post-LN?** Stable training without elaborate warmup; cleaner residual stream.
6. **Why residual connections?** Gradient flow through depth; identity path makes vanishing impossible.
7. **What's the residual stream?** A communication channel that layers read from and write to via $+$.
8. **Why are transformers permutation-equivariant?** Pure attention has no built-in position; positional encoding fixes this.
9. **Encoder vs decoder?** Bidirectional vs causal mask; different training objectives.
10. **Cross-attention?** Decoder $Q$, encoder $K/V$; how seq-to-seq models look at source.
11. **Complexity of attention?** $O(N^2 d)$ compute, $O(N^2)$ memory naively.
12. **Why did transformers win over RNNs?** Parallelism, gradient flow, scale.

---

## 15. Drill plan

1. Whiteboard scaled dot-product attention with the variance argument for $\sqrt{d_k}$.
2. Whiteboard the full block (pre-LN, residual, FFN).
3. Explain residual stream metaphor in 60 seconds.
4. Cite encoder/decoder/cross-attention differences.
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md) until 40+/60 cold.

---

## 16. Further reading

- Vaswani et al., "Attention is All You Need" (2017).
- Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021).
- Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories" (2021).
- Voita et al., "Analyzing Multi-Head Self-Attention" (2019).
- Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020) — pre-LN vs post-LN.
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020).
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022).
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE, 2021).

If you internalize this document, the transformer stops being a list of components and becomes a sequence of design choices, each of which you can defend.
