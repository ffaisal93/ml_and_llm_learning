# Attention Mechanisms — Interview Grill

> 50 questions on the attention family beyond the basic transformer. For the foundational scaled-dot-product material, see `04_transformers/INTERVIEW_GRILL.md`.

---

## A. KV-sharing variants

**1. What's MHA?**
Multi-Head Attention. $h$ attention heads each with their own $Q, K, V$ projection matrices. Standard in the original transformer. KV cache cost: $2 \cdot h \cdot d_h \cdot N \cdot L$ per sequence (all heads have separate $K$ and $V$).

**2. What's MQA?**
Multi-Query Attention (Shazeer 2019). All $h$ query heads share **one** $K$ projection and **one** $V$ projection. KV cache: $2 \cdot d_h \cdot N \cdot L$. Reduction factor of $h$ (typically 8–64×).

**3. What's GQA?**
Grouped-Query Attention (Ainslie et al. 2023). Compromise: $n_{\text{kv\_heads}}$ shared groups; each group's $K/V$ used by $g = h / n_{\text{kv\_heads}}$ query heads. Reduction factor $g$. LLaMA-2 70B: $n_{\text{kv\_heads}} = 8, h = 64$ → 8× reduction with negligible quality loss.

**4. What's MLA?**
Multi-Latent Attention (DeepSeek-V2 2024). Compress KV into a low-rank latent $c_t = x \cdot W_{DKV}$ (dim $d_c \ll d$). Cache only $c_t$; reconstruct $K = c_t \cdot W_{UK}, V = c_t \cdot W_{UV}$ on use. Memory savings comparable to GQA, sometimes better quality. Trade-off: extra compute at attention time.

**5. Why has GQA become the default?**
GQA-8 gets nearly all of MQA's KV savings (8× reduction) with minimal quality cost (~0.1% on benchmarks vs MQA's ~1–2%). Sweet spot.

**6. What does MQA cost in quality?**
Empirically ~1–2% on benchmarks. Larger drops on tasks needing fine-grained head specialization (some reasoning tasks). Acceptable for many production models.

**7. KV cache savings ranked?**
MLA $\approx$ GQA-8 > GQA-16 > MHA, in terms of memory. Quality: MHA $\geq$ GQA-16 $\approx$ GQA-8 $\approx$ MLA > MQA. So GQA-8 and MLA Pareto-dominate.

**8. Walk me through the MQA forward computation.**
Given input $X \in \mathbb{R}^{N \times d}$, project: $Q = X W_Q \in \mathbb{R}^{N \times d}$ (split into $h$ heads of size $d_h$); $K = X W_K \in \mathbb{R}^{N \times d_h}$ (single, shared); $V = X W_V \in \mathbb{R}^{N \times d_h}$ (single, shared). For each head $i$: $\text{head}_i = \operatorname{softmax}(Q_i K^\top / \sqrt{d_h}) V$. Concatenate, project with $W_O$.

---

## B. Causal and cross attention

**9. What's the causal mask?**
Lower-triangular $M$: 0 on/below diagonal, $-\infty$ above. Add to attention scores: $\text{scores} = Q K^\top / \sqrt{d_k} + M$. The $-\infty$ becomes 0 after softmax. Position $i$ cannot attend to $j > i$. Implements autoregressive constraint without changing the attention algorithm.

**10. Why does the causal mask work during parallel training?**
The model sees the entire sequence in parallel. The causal mask ensures each position only "sees" earlier ones in the attention computation. Loss is computed at every position simultaneously — $N$ next-token-prediction tasks per sequence in parallel. This parallelism is why transformers train so much faster than RNNs.

**11. What's cross-attention?**
$Q$ from one source (decoder state), $K$ and $V$ from another (encoder output). The decoder attends to the encoder output. Used in encoder-decoder models (T5, BART, original transformer).

**12. Why is cross-attention's KV cache cheap?**
The encoder output is fixed for the request. $K$ and $V$ from encoder output are computed **once** per request and reused for every decode step. Unlike self-attention, where $K$ and $V$ grow with each generated token, cross-attention $K/V$ are static.

**13. Why don't pure decoder LLMs have cross-attention?**
They handle "looking at" inputs by placing the input in the context window. Self-attention then handles input-to-output relationships. Architecturally simpler; functionally similar to encoder-decoder for many tasks.

---

## C. Sliding window and sparse attention

**14. What's sliding window attention?**
Each position attends only to the previous $W$ tokens. Causal triangular mask becomes a band: $M_{i,j} = 0$ if $i - W < j \leq i$, else $-\infty$. Compute $O(N \cdot W)$ instead of $O(N^2)$. KV cache bounded to most recent $W$ per sequence per layer.

**15. Why does sliding window still capture long range?**
Stacking. With $L$ layers and window $W$, the top layer has receptive field $L \cdot W$ (each layer extends the effective span by $W$). Mistral 7B ($L = 32, W = 4096$) has theoretical receptive field 131K tokens despite each layer only seeing 4K. Plus global tokens / attention sinks help.

**16. What are global tokens?**
A small set of positions that attend to and are attended by all other positions. Short-circuit the receptive-field problem. Used in Longformer (4K window + global tokens like `[CLS]`), BigBird.

**17. Trade-off of sliding window?**
Saves compute and KV memory linearly. Cost: information far outside the window can't propagate efficiently to later layers. Quality drops on tasks requiring genuinely long-range exact recall (e.g., needle-in-haystack tests at long range).

**18. Why aren't sparse attention patterns more popular in production?**
Modern long-context production models tend to use full attention with efficient kernels (FlashAttention) and KV memory tricks (paged, quantized) rather than approximate sparsity. The quality cost of sparsity is usually unacceptable; the compute savings are achievable other ways.

**19. What's BigBird?**
Block-structured sparse attention: sliding window + global tokens + random sparse pattern. Theoretically retains expressiveness of full attention; empirically reasonable. Less popular than full-attention + FlashAttention.

**20. What's the Reformer trick?**
Locality-Sensitive Hashing. Hash queries and keys; only attend within the same hash bucket. $O(N \log N)$ compute with quality close to full attention. Notable for being ~theoretically nice but not widely used in production.

---

## D. Linear attention

**21. What's linear attention?**
Replace softmax with a kernel approximation: $\operatorname{attention}(Q, K, V) = \phi(Q) \cdot (\phi(K)^\top V) / (\phi(Q) \cdot \phi(K)^\top \mathbf{1})$ for some feature map $\phi$. Order: compute $\phi(K)^\top V$ first (size $d \times d$), then $\phi(Q) \cdot \cdots$. Result: $O(N \cdot d^2)$ — linear in $N$.

**22. Why does linear attention enable RNN-style decoding?**
The attention has the form $\text{output}_t = \phi(Q_t) \cdot S_t$, where $S_t = \sum_{i \leq t} \phi(K_i) V_i^\top$ is updated as $S_t = S_{t-1} + \phi(K_t) V_t^\top$. So decoding is a **recurrent state update** with constant memory $O(d^2)$ and constant time $O(d^2)$ per step — like an RNN.

**23. Trade-off of linear vs softmax attention?**
Linear: $O(N)$ compute, constant-memory decode. Quality: usually weaker than softmax, especially for in-context learning and exact recall. Whether linear attention catches up at frontier scale is an open question.

**24. What's Performer?**
Linear attention via Random Fourier Features (Choromanski et al. 2020). $\phi$ chosen to approximate the softmax kernel. Provably good approximation; empirically works at moderate scales.

**25. What's RWKV?**
A linear-attention model designed specifically for autoregressive generation. Replaces transformer attention with a recurrent rule that's parallelizable in training. Strong open-source models exist (RWKV-4, -5, -6). Not yet at frontier-LLM scale.

**26. What's the relationship between linear attention and SSMs?**
Both have $O(N)$ sequence complexity and a recurrent decode form. Dao & Gu's *Transformers are SSMs* (2024) formalized that *fixed-parameter* SSMs and certain linear-attention variants are duals (via structured semiseparable matrices, used in Mamba-2). But Mamba's *selective* SSM has input-dependent $A, B, C$ — that selectivity makes Mamba strictly more expressive than vanilla linear attention. "SSMs are linear attention" holds for non-selective variants but oversimplifies for Mamba.

---

## E. FlashAttention quick recap

**27. What's FlashAttention?**
I/O-aware tiled attention with online softmax. Same FLOPs as standard attention; far less memory access. Tiles $Q, K, V$ to fit in SRAM; computes partial softmax statistics block-by-block. 2–4× wall-clock speedup at long sequences.

**28. Online softmax mechanism?**
Maintain running max $m$ and running sum $s$. For each new value: $m' = \max(m, x); s' = s \cdot \exp(m - m') + \exp(x - m'); m = m'; s = s'$. Single pass, numerically equivalent to two-pass softmax. Lets attention be computed block-by-block without materializing the full matrix.

**29. Does FlashAttention reduce theoretical complexity?**
No. Same FLOPs. Reduction is in memory access: from $O(N^2)$ reads/writes to HBM down to $O(N)$ for intermediate steps.

---

## F. Attention head analyses

**30. What's an induction head?**
A head that copies a token from earlier in the context when a similar prefix recurs. E.g., if the context has "X Y ... X", an induction head attends from the second X to Y and copies. Mechanism for in-context learning. Olsson et al. 2022 ("In-context Learning and Induction Heads").

**31. Why do induction heads matter?**
Their emergence in training correlates with the transition to in-context learning ability. Removing them ablates ICL. Provides a mechanistic story for how transformers do few-shot learning.

**32. What's a previous-token head?**
A head that attends from position $t$ to position $t-1$. Used in counting and copying tasks. Common in lower layers of trained transformers.

**33. Why are some heads ablation-resistant?**
Empirical observation that you can remove many heads without hurting quality (Voita et al., Michel et al.). Suggests redundancy: multiple heads compute similar functions; removing one doesn't fail because others cover it. Doesn't generalize: some heads are essential, and their identity differs across models.

**34. What does an "attention sink" look like?**
The first few tokens (often `[BOS]` or just position 0) attract disproportionate attention from many heads. They act as a "sink" — heads that don't have anything specific to attend to dump attention there. StreamingLLM showed that removing attention sinks from the KV cache breaks long-context behavior.

**35. Why does temperature appear in attention scores?**
The $1/\sqrt{d_k}$ is effectively a temperature on the softmax. Standard convention: scores divided by $T$, so larger $T$ (here $\sqrt{d_k}$) → softer/smoother distribution; smaller $T$ → sharper. We *want* a smoother distribution at large $d_k$ to prevent softmax saturation, so dividing by $\sqrt{d_k}$ — which grows with $d_k$ — is exactly right.

---

## G. Production and engineering

**36. What's "paged attention"?**
KV cache allocation in fixed-size blocks (vLLM). Eliminates fragmentation. Block table per request maps logical positions to physical blocks. See `06_llm_inference/`.

**37. What's the difference between "pre-fill attention" and "decode attention" in serving systems?**
Pre-fill: process the entire prompt in one parallel forward pass. Compute-bound (large matmuls). Decode: one-token-at-a-time autoregressive. Memory-bound (read full weights for one token). Different optimal kernels for each phase (FlashAttention vs FlashDecoding).

**38. Why does the KV cache grow linearly with context?**
Each new token contributes one $K$ vector and one $V$ vector per layer per head. KV cache size for a sequence of length $N$: $2 \cdot n_{\text{kv\_heads}} \cdot d_h \cdot N \cdot L \cdot \text{bytes}$. Linear in $N$. This is the fundamental memory cost of long context.

**39. What's the receptive field of layer $L$ in a sliding-window-$W$ transformer?**
$L \cdot W$. Each layer extends the effective receptive field by $W$ because layer $L+1$ can attend to layer $L$'s outputs at positions $i, i-1, \ldots, i-W$, each of which represents $i-W, i-2W$, etc.

**40. Sliding window + global tokens — why combine?**
Sliding window gives $O(N \cdot W)$ compute and bounded KV memory. Global tokens give a small number of "broadcast" channels that don't suffer the receptive-field limitation. Combined, you keep most of sliding window's efficiency while preserving long-range information through the globals.

---

## H. Quick fire

**41.** *Original MHA paper?* Vaswani et al. 2017.
**42.** *MQA paper?* Shazeer 2019.
**43.** *GQA paper?* Ainslie et al. 2023.
**44.** *MLA paper?* DeepSeek-V2, 2024.
**45.** *Standard $n_{\text{kv\_heads}}$ for LLaMA-2 70B?* 8.
**46.** *Default Mistral sliding window?* 4096 tokens.
**47.** *Linear attention sequence complexity?* $O(N \cdot d^2)$.
**48.** *FlashAttention sequence complexity?* Same $O(N^2 \cdot d)$ FLOPs as standard, lower memory access.
**49.** *Reformer complexity?* $O(N \log N)$.
**50.** *Induction head function?* Copy a token after prefix recurrence.

---

## Self-grading

If you can't answer 1-10, you don't know modern attention variants. If you can't answer 11-25, you can't pass an LLM-architecture round. If you can't answer 26-50, frontier-lab interviews will go deeper than you can follow.

Aim for 35+/50 cold.
