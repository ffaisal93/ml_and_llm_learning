# Transformers — Interview Grill

> 60 brutal questions on the transformer architecture. Drill until you can answer 40+ cold.

---

## A. Core architecture

**1. What does a transformer block do?**
Two operations: attention (mixes information across positions) and FFN (per-token non-linear computation), each wrapped in a residual connection and preceded by LayerNorm in modern (pre-LN) transformers. That's the entire architecture; everything else elaborates these two.

**2. Walk me through scaled dot-product attention.**
Project input $X$ into queries $Q = X W_Q$, keys $K = X W_K$, values $V = X W_V$. Compute $\text{scores} = Q K^\top / \sqrt{d_k}$, apply softmax row-wise to get attention weights, multiply by $V$ to get output. Each output position is a weighted average of all values, where the weights come from query-key compatibility.

**3. Why divide by $\sqrt{d_k}$?**
**Without it, scores get huge in high dim, softmax saturates, gradients die.** The math: dot product of two $d_k$-dim unit-variance vectors has variance $d_k$, so entries grow as $\sqrt{d_k}$. Dividing by $\sqrt{d_k}$ keeps score variance $O(1)$ regardless of dimension → softmax stays in its linear regime, gradients flow.

**4. Why softmax and not sigmoid?**
Softmax produces a **probability distribution** over keys: non-negative weights summing to 1. This makes $\mathrm{softmax}(\text{scores}) \cdot V$ a proper convex combination — a weighted average. Sigmoid would give independent gates per key, not a normalized mixture, and you'd need ad-hoc renormalization. Softmax also sharpens: largest score dominates exponentially.

**5. What's the computational complexity of attention?**
$O(N^2 \cdot d)$ time and $O(N^2)$ memory for the attention matrix. The $N^2$ is the limiting factor for long contexts. FlashAttention reduces memory access but not FLOPs; linear attention variants try to reduce FLOPs at some quality cost.

**6. What's multi-head attention and why?**
Split $d$ into $h$ heads of size $d_h = d/h$. Run attention separately in each head; concatenate outputs; project. Same total parameters as single-head but allows the model to attend to multiple patterns simultaneously (e.g., one head for syntax, another for coreference). Empirically, ~5–10 heads provide most of the benefit.

**7. What does each attention head learn?**
Empirical analyses show some heads learn interpretable functions (positional, syntactic dependency, coreference) but most are not cleanly interpretable. Ablating individual heads usually costs little, suggesting heavy redundancy. Voita et al., Clark et al. for canonical analyses.

**8. What does the FFN do?**
Per-token non-linear computation. $\mathrm{FFN}(x) = W_2 \cdot \mathrm{activation}(W_1 \cdot x)$. The 4× expansion ($W_1: d \to 4d$, $W_2: 4d \to d$) is the standard ratio. Provides non-linearity (attention is linear in values) and acts as a key-value memory holding factual knowledge. Holds 2/3+ of all transformer parameters.

**9. Why 4× expansion in the FFN?**
Empirical choice from the original paper. Larger $d_{\text{ff}}/d$ improves quality up to a point; 4× is the sweet spot for vanilla FFN. Some modern architectures use 8/3 with SwiGLU activation (Llama). Going much larger increases parameters without proportional benefit.

**10. What's SwiGLU and why is it used?**
Gated activation: $\mathrm{SwiGLU}(x) = (\mathrm{Swish}(x W_1) \odot x W_2) W_3$. Adds a gating term that empirically outperforms ReLU/GELU for LLMs. Triples the FFN parameter count compared to single-matrix FFN, so the inner dimension is reduced (often to $8/3 \cdot d$) to keep total parameters comparable.

**11. What's the role of $W_O$ in multi-head attention?**
The output projection. After $h$ heads each produce a $d_h$-dim output, you concatenate to get a $d$-dim vector, then project with $W_O: d \to d$. $W_O$ allows the model to mix information across heads — without it, each head's output would be confined to its own subspace.

**12. Why are residual connections critical?**
Gradient flow. Without them, gradients through depth multiply across layers and vanish. With them, the gradient w.r.t. early layers includes an identity term (the residual passes the gradient through unchanged), making vanishing-through-depth impossible. Enables stacking 100+ layers.

**13. What's the "residual stream" view?**
Each token's representation flows through layers as a stream. Each block reads from the stream (via LayerNorm + projections), computes something, and writes back via the residual $+$. Layers communicate by reading and writing to this shared stream — like a bus in computer architecture. This is the modern interpretability framing.

---

## B. Normalization placement

**14. Pre-LN vs post-LN?**
Post-LN (original, 2017): $x \leftarrow \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$. Norm after residual.
Pre-LN (modern): $x \leftarrow x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))$. Norm before sublayer; residual is unnormed.
Pre-LN trains stably without elaborate warmup; post-LN does not at modern scales. Every modern LLM uses pre-LN or RMSNorm.

**15. Why does pre-LN train more stably?**
**Pre-LN keeps a clean signal flowing through the residual stream; post-LN keeps re-normalizing it and amplifies any wobble.** Mechanically: in pre-LN, the sublayer reads a normed input but writes back to the unnormed residual — the residual path's gradient flows unchanged. Post-LN renormalizes the stream every block, which can amplify small perturbations into instability.

**16. What's RMSNorm and why is it used?**
LayerNorm: $(x - \mu) / \sigma$. RMSNorm: $x / \mathrm{RMS}(x)$ where $\mathrm{RMS}(x) = \sqrt{\mathrm{mean}(x^2)}$. Just unit-variance normalization, no mean centering. ~30% cheaper (one fewer reduction). Empirically as good as LayerNorm. Used in LLaMA, Gemma, Mistral, etc.

**17. Why not BatchNorm in transformers?**
BN normalizes across the batch dimension, which is bad for sequences: (a) different sequence lengths in a batch, (b) at inference time you may want to process single sequences without batch statistics, (c) the running statistics lag during training. LayerNorm normalizes per-token, sidestepping all of these.

---

## C. Positional information

**18. Why do transformers need positional encoding?**
Pure attention is permutation-equivariant: shuffle the input tokens and the output is shuffled the same way. So attention has no notion of order. Positional encoding injects the position information that the architecture otherwise lacks.

**19. What's sinusoidal positional encoding?**
The original method. Add deterministic sinusoidal vectors per position: $\mathrm{PE}(t, 2i) = \sin(t / 10000^{2i/d})$, $\mathrm{PE}(t, 2i+1) = \cos(\cdots)$. Properties: same encoding regardless of training, in principle extrapolates to longer sequences than training (in practice, mediocre extrapolation).

**20. What's learned positional encoding?**
Treat position as a categorical feature; learn an embedding per position. Used in BERT, GPT-2. Simple, works well within max-position seen in training, **does not extrapolate** beyond.

**21. What's RoPE?**
Rotary Position Embedding (Su et al. 2021). Encode position by **rotating** $Q$ and $K$ by an angle proportional to position before computing the dot product. The dot product $Q \cdot K$ then depends on the **relative position** (position difference), not absolute positions. Standard in modern LLMs (LLaMA, Mistral, GPT-J, etc.).

**22. What's ALiBi?**
Attention with Linear Biases (Press et al. 2021). Add a per-head bias to attention scores that linearly penalizes attending to distant positions: $\text{scores} \mathrel{-}= m_h \cdot |i - j|$, where $m_h$ is a head-specific slope. No positional embeddings at all. Extrapolates well to longer sequences. Used in BLOOM and others.

**23. Why does RoPE extrapolate better than learned PE?**
RoPE encodes **relative** positions, so it doesn't matter what absolute position you've seen at training. Learned PE only knows positions seen in training. When you go beyond, learned PE is undefined; RoPE just keeps rotating. (RoPE still has practical limits — frequencies trained at short range may not generalize. YaRN and dynamic NTK scaling extend this.)

---

## D. Encoder vs decoder

**24. Encoder vs decoder vs encoder-decoder?**
Encoder: bidirectional attention, no mask, contextualizes input (BERT, embeddings).
Decoder: causal attention, autoregressive generation (GPT, modern LLMs).
Encoder-decoder: encoder processes source, decoder generates target with cross-attention to encoder (T5, original transformer for translation).

**25. Why are modern LLMs decoder-only?**
Simpler architecture (one tower not two), one objective (next-token), naturally extends to in-context learning. Empirically scales better than encoder-decoder for general-purpose generation. The bidirectional encoder objective (masked LM) doesn't extend cleanly to long contexts and generation.

**26. What's the causal mask?**
A lower-triangular matrix $M$ with $0$ on/below diagonal and $-\infty$ above, added to the attention scores: $\text{scores} = Q K^\top / \sqrt{d_k} + M$. The $-\infty$ entries become $0$ after softmax, so position $i$ cannot attend to position $j > i$. Implements autoregressive constraint without changing the attention algorithm.

**27. What's cross-attention?**
The mechanism in encoder-decoder models. Decoder queries: $Q = \text{decoder-state} \cdot W_Q$. Encoder keys/values: $K, V = \text{encoder-output} \cdot W_K, W_V$. The decoder attends to the encoder output. Pure decoder LLMs don't have cross-attention; they handle inputs via in-context.

**28. What's masked language modeling (MLM)?**
BERT's pretraining objective. Mask 15% of tokens; train the model to predict them from bidirectional context. Bidirectional → encoder. Doesn't directly enable generation; the model learns rich representations but can't be sampled token-by-token.

**29. What's causal/autoregressive language modeling (CLM)?**
GPT's pretraining. Predict each token from preceding context only. Causal mask in attention. Naturally generates: sample token, append, repeat. The dominant pretraining objective in modern LLMs.

---

## E. Subtleties and gotchas

**30. What's weight tying?**
Sharing the embedding matrix between input embeddings and output unembedding. Saves parameters (one of the largest weight tensors), and empirically helps (the input/output spaces are dual). Standard in many LLMs but not all (GPT-2 ties; some recent open-source models don't).

**31. What's the maximum context length determined by?**
At training: the longest sequence in the training data, plus positional encoding range. At inference: KV cache memory. The architecture itself doesn't impose a hard limit; it's compute, memory, and positional encoding that do. RoPE/ALiBi extend the practical range; brute KV memory limits the rest.

**32. What's flash attention's contribution?**
I/O-aware tiled attention with online softmax. Same FLOPs as standard attention, but avoids materializing the $N \times N$ matrix in HBM. 2–4x wall-clock speedup on long sequences. See `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` for details.

**33. Why do transformers use tied positional encoding for QK but not V?**
RoPE rotates $Q$ and $K$ (so the dot product depends on relative position) but does not rotate $V$ (which is the content being mixed). Mixing positional information into $V$ would unnecessarily entangle position with content; rotating only $Q$ and $K$ cleanly separates the two roles.

**34. What's the attention sink phenomenon?**
Empirical observation: in long-context transformers, the first few tokens get disproportionate attention from many heads. They act as "sinks" that absorb attention mass. Removing them (when streaming) breaks the model. Mitigation: keep them in the KV cache (StreamingLLM, sink tokens).

**35. What's a "logit lens"?**
Interpretability tool: project intermediate-layer activations through the unembedding matrix to read out token probabilities at intermediate layers. Reveals how predictions sharpen across depth — early layers give vague predictions, late layers sharpen.

**36. What does an attention pattern look like for a "copy head"?**
A nearly-diagonal attention pattern shifted by a fixed offset. Position $t$ attends primarily to position $t - k$ for some $k$. Useful for copying tokens from a previous position (e.g., variable names in code, repeated structure in lists).

---

## F. Computational and scaling

**37. Where do parameters live in a transformer?**
Roughly: 2/3 to 3/4 in FFN matrices (4d expansion × 2 matrices = $8d^2$ per layer); attention has $Q, K, V, O$ projections ($4 d^2$ per layer). Embeddings ($\text{vocab} \times d$) and unembeddings can be substantial. Layer norms: trivial. So FFN dominates.

**38. Compute breakdown for a forward pass?**
For a transformer with $L$ layers, sequence $N$, dim $d$:

- Attention: $O(L \cdot N^2 \cdot d)$
- FFN: $O(L \cdot N \cdot d^2)$ (per-token, scales with hidden $4d$)
- Embeddings: $O(N \cdot d)$

For $N \gg d$, attention dominates; for $N \ll d$, FFN dominates. With vanilla 2-matmul FFN at $4d$ expansion, attention FLOPs $\sim 4N^2 d$ and FFN FLOPs $\sim 8 N d^2$ → crossover at $N \approx 2d$. With SwiGLU's 3 matmuls scaled to match params, the constants shift but the same-order conclusion holds: crossover scales as $O(d)$, on the order of $2d$–$4d$.

**39. Memory breakdown for a forward pass?**
Activations: $O(L \cdot N \cdot d)$ for stream + intermediate FFN states + attention scores $O(N^2)$ per layer.
Weights: roughly $12 d^2 L + d V$ ($V$ = vocab size) for a vanilla transformer.
KV cache (during decode): $2 L \cdot n_{\text{kv-heads}} \cdot d_{\text{head}} \cdot N$ per sequence.

**40. What are scaling laws?**
Kaplan et al. 2020, Hoffmann et al. 2022 (Chinchilla): loss scales as a power law in compute, parameters, and tokens. Chinchilla optimal: ~20 tokens per parameter (Llama-2 increased this further). Scaling laws guide compute allocation between model size and training data.

**41. Chinchilla vs Kaplan?**
Kaplan suggested overparameterized + undertrained models. Chinchilla showed those models are suboptimal: at fixed compute, smaller models trained on more tokens beat larger models trained on fewer. The "Chinchilla-optimal" frontier rebalanced the ML community's training recipes.

---

## G. Design choices and ablations

**42. Why use multi-head and not single-head?**
Empirically, parallel attention to multiple patterns helps. Theoretically, multiple heads form a richer function class than a single head with $h \times d_h = d$ dimensions, because each head has its own softmax (heads can't share normalization).

**43. Why not normalize Q and K explicitly (instead of scaling by $\sqrt{d_k}$)?**
Some recent work does (QK-norm). Empirically helps stability for very large models. The original paper's $\sqrt{d_k}$ is the simpler version that works for moderate scales.

**44. Why don't we make $d_k$ and $d_v$ different?**
We could; the original paper does have them as separate hyperparameters. In practice, almost all implementations use $d_k = d_v = d / h$. Empirically, no clear win from making them different.

**45. What happens if you remove the FFN?**
Catastrophic. Pure-attention transformers are essentially linear (attention is linear in V). Cannot represent non-linear functions, cannot store factual knowledge well. Some "linear transformer" variants reintroduce non-linearity elsewhere; they are not standard.

**46. What happens if you remove the residual connections?**
Training fails for deep networks. Gradients vanish through depth. Even a few layers becomes hard to train. Residuals are not optional.

**47. What happens if you remove LayerNorm?**
Activations grow uncontrolled across depth (the residual stream accumulates layer outputs). Training becomes very unstable, often diverges. In Pre-LN, removing LayerNorm makes the model unable to limit the read magnitude from the stream.

---

## H. Comparisons

**48. Transformer vs RNN?**
Transformer wins on parallelism (no sequential bottleneck), gradient flow (one attention layer reaches anything), and scaling. RNN wins on inference memory (constant per step vs growing KV cache) and on data efficiency at small scales.

**49. Transformer vs CNN?**
CNNs have local connectivity and translation equivariance (good biases for vision). Transformers have weaker biases but better scaling. ViT showed transformers can match/beat CNNs on vision at sufficient scale.

**50. Transformer vs SSM (Mamba)?**
SSMs reintroduce sequential processing in a parallelizable form (convolutional view). $O(N)$ sequence complexity vs transformer's $O(N^2)$. Empirically competitive at smaller scale; whether they match transformers at frontier scale is open. Hybrid models (combining attention layers and SSM layers) are an active area.

---

## I. Quick-fire

**51.** *Original paper?* Vaswani et al. 2017.
**52.** *Standard activation in original FFN?* ReLU.
**53.** *Modern activation in LLM FFN?* SwiGLU.
**54.** *Standard FFN expansion ratio?* 4× (or 8/3 with SwiGLU).
**55.** *Pre-LN or post-LN in modern LLMs?* Pre-LN (or RMSNorm variant).
**56.** *Most common positional encoding in modern LLMs?* RoPE.
**57.** *Encoder-only flagship?* BERT.
**58.** *Decoder-only flagship?* GPT family, LLaMA, etc.
**59.** *Encoder-decoder flagship?* T5.
**60.** *Attention complexity?* $O(N^2 d)$ compute, $O(N^2)$ memory naively.

---

## Self-grading

If you can't answer 1–10, you don't know transformers. If you can't answer 11–25, you can't pass an LLM-focused MLE round. If you can't answer 26–50, frontier-lab applied scientist screens will expose gaps.

Aim for 40+/60 cold before any architecture interview.
