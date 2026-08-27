# Attention Mechanisms — Interview Grill

> 50 questions on the attention family beyond the basic transformer. For the foundational scaled-dot-product material, see [`04_transformers/INTERVIEW_GRILL.md`](../04_transformers/INTERVIEW_GRILL.md).

---

## A. KV-sharing variants

**1. What's MHA?**
Multi-Head Attention. $h$ attention heads each with their own $Q, K, V$ projection matrices. Standard in the original transformer. KV cache cost: $2 \cdot h \cdot d_h \cdot N \cdot L$ per sequence (all heads have separate $K$ and $V$).

> **Saying it out loud.** MHA is the original recipe: every head gets its own private queries, keys, and values. With $h$ heads you're caching $h$ separate $K$ tensors and $h$ separate $V$ tensors for every token, at every layer. That's maximally expressive — heads can specialize completely independently — but it makes the KV cache enormous, and the KV cache is what caps how many users you can serve concurrently. Everything that came after MHA is an attempt to shrink that cache without losing the specialization.

**2. What's MQA?**
Multi-Query Attention (Shazeer 2019). All $h$ query heads share **one** $K$ projection and **one** $V$ projection. KV cache: $2 \cdot d_h \cdot N \cdot L$. Reduction factor of $h$ (typically 8–64×).

> **Saying it out loud.** MQA is the aggressive version: keep all the query heads, but have them all share a single key and a single value head. You still get $h$ different questions being asked, they're just all looking up the same table. That cuts your KV cache by a factor of $h$ — often 32 or 64x — which is a massive win for long-context serving. The cost is real though: about 1 to 2 percent on benchmarks, because the heads lose the ability to look at genuinely different content.

**3. What's GQA?**
Grouped-Query Attention (Ainslie et al. 2023). Compromise: $n_{\text{kv-heads}}$ shared groups; each group's $K/V$ used by $g = h / n_{\text{kv-heads}}$ query heads. Reduction factor $g$. LLaMA-2 70B: $n_{\text{kv-heads}} = 8, h = 64$ → 8× reduction with negligible quality loss.

> **Saying it out loud.** GQA splits the difference. Instead of $h$ key-value heads or one, you have a handful of groups — say eight — and each group's K and V are shared by a bunch of query heads. LLaMA-2 70B does exactly this: 64 query heads, 8 KV heads, so an 8x cache reduction. The reason it's the default is that the quality curve is very flat between 8 groups and full MHA, so you get almost all of MQA's savings for almost none of MQA's quality loss.

**4. What's MLA?**
Multi-Latent Attention (DeepSeek-V2 2024). Compress KV into a low-rank latent $c_t = x \cdot W_{DKV}$ (dim $d_c \ll d$). Cache only $c_t$; reconstruct $K = c_t \cdot W_{UK}, V = c_t \cdot W_{UV}$ on use. Memory savings comparable to GQA, sometimes better quality. Trade-off: extra compute at attention time.

> **Saying it out loud.** MLA takes a different route: instead of sharing K and V heads, it compresses them. You project the token down into a small latent vector, cache only that, and reconstruct the full keys and values on the fly when you need them. It's basically low-rank compression of the cache rather than head-sharing, and DeepSeek showed it can hit GQA-level memory with equal or better quality. The tradeoff you should name is that you're trading memory for compute — you pay an extra projection at every attention step.

**5. Why has GQA become the default?**
GQA-8 gets nearly all of MQA's KV savings (8× reduction) with minimal quality cost (~0.1% on benchmarks vs MQA's ~1–2%). Sweet spot.

> **Saying it out loud.** Because it sits at the knee of the curve. GQA-8 gets you essentially the same 8x cache reduction that makes long-context serving affordable, but the quality hit is around a tenth of a percent versus MQA's one to two percent. In other words, going all the way to MQA buys you very little additional memory and costs you real accuracy. When one option dominates on the tradeoff curve that clearly, everybody converges on it, and that's what happened after the 2023 GQA paper.

**6. What does MQA cost in quality?**
Empirically ~1–2% on benchmarks. Larger drops on tasks needing fine-grained head specialization (some reasoning tasks). Acceptable for many production models.

> **Saying it out loud.** About one to two percent on standard benchmarks, and it's not uniform. The damage concentrates on tasks that need heads to look at genuinely different things — multi-step reasoning, precise retrieval from context — because a shared K and V means all heads are reading the same representation. For a lot of production workloads that's an acceptable trade for a 32x smaller cache. But it's exactly why GQA replaced it: you don't have to accept that tradeoff anymore.

**7. KV cache savings ranked?**
MLA $\approx$ GQA-8 > GQA-16 > MHA, in terms of memory. Quality: MHA $\geq$ GQA-16 $\approx$ GQA-8 $\approx$ MLA > MQA. So GQA-8 and MLA Pareto-dominate.

> **Saying it out loud.** On memory: MLA and GQA-8 are the best, then GQA-16, then plain MHA at the bottom. On quality it's roughly the reverse, except that GQA-8, GQA-16, and MLA are all statistically close to MHA and MQA is the clear straggler. The useful way to say this is that GQA-8 and MLA Pareto-dominate — nothing else is both cheaper and better. MQA only makes sense if you're memory-desperate and can eat the quality hit.

**8. Walk me through the MQA forward computation.**
Given input $X \in \mathbb{R}^{N \times d}$, project: $Q = X W_Q \in \mathbb{R}^{N \times d}$ (split into $h$ heads of size $d_h$); $K = X W_K \in \mathbb{R}^{N \times d_h}$ (single, shared); $V = X W_V \in \mathbb{R}^{N \times d_h}$ (single, shared). For each head $i$: $\text{head}_i = \mathrm{softmax}(Q_i K^\top / \sqrt{d_h}) V$. Concatenate, project with $W_O$.

> **Saying it out loud.** You project the input three times, but the shapes are asymmetric. $Q$ is full width and gets split into $h$ heads; $K$ and $V$ are only $d_h$ wide — a single head's worth, not split at all. Then each query head does its own softmax against that one shared $K$ and mixes that one shared $V$, you concatenate the head outputs and project with $W_O$. The thing to notice is that $W_K$ and $W_V$ shrank from $d \times d$ to $d \times d_h$, so you're also saving parameters, not just cache.

---

## B. Causal and cross attention

**9. What's the causal mask?**
Lower-triangular $M$: 0 on/below diagonal, $-\infty$ above. Add to attention scores: $\text{scores} = Q K^\top / \sqrt{d_k} + M$. The $-\infty$ becomes 0 after softmax. Position $i$ cannot attend to $j > i$. Implements autoregressive constraint without changing the attention algorithm.

> **Saying it out loud.** The causal mask is how you stop a token from reading ahead. You add a matrix to the scores that's zero on and below the diagonal and negative infinity above it — and negative infinity exponentiates to zero, so those weights vanish in the softmax. Nothing about the attention algorithm changes; you're just adding a constant before the softmax. That's what enforces the autoregressive property, and it costs you nothing at training time.

**10. Why does the causal mask work during parallel training?**
The model sees the entire sequence in parallel. The causal mask ensures each position only "sees" earlier ones in the attention computation. Loss is computed at every position simultaneously — $N$ next-token-prediction tasks per sequence in parallel. This parallelism is why transformers train so much faster than RNNs.

> **Saying it out loud.** Because masking lets you get $N$ training examples out of one forward pass. The model sees the whole sequence at once, but position 5's output only depended on tokens 1 through 5, so the prediction it makes there is a legitimate next-token prediction — and the same is true at every position simultaneously. So you compute the loss at all $N$ positions in parallel from a single pass. That's the entire reason transformers train orders of magnitude faster than RNNs, which have to walk the sequence one step at a time.

**11. What's cross-attention?**
$Q$ from one source (decoder state), $K$ and $V$ from another (encoder output). The decoder attends to the encoder output. Used in encoder-decoder models (T5, BART, original transformer).

> **Saying it out loud.** Cross-attention is attention where the queries come from one sequence and the keys and values come from another. In a translation model the decoder asks the questions and the encoded source sentence answers them — 'which French words matter for the English word I'm about to write?'. That's what produces the classic alignment heatmaps you see in translation papers. It's the mechanism that lets an encoder-decoder keep the source and the target in separate representational spaces instead of jamming them into one stream.

**12. Why is cross-attention's KV cache cheap?**
The encoder output is fixed for the request. $K$ and $V$ from encoder output are computed **once** per request and reused for every decode step. Unlike self-attention, where $K$ and $V$ grow with each generated token, cross-attention $K/V$ are static.

> **Saying it out loud.** Because the thing being attended to never changes. The encoder runs once on the input, and its keys and values are fixed for the rest of the request — you compute them at step one and reuse them for every single decode step. Compare that to self-attention, where every token you generate appends another K and V, so the cache grows with output length. So cross-attention's memory cost is constant in the number of generated tokens, which is a genuinely nice property that decoder-only models gave up.

**13. Why don't pure decoder LLMs have cross-attention?**
They handle "looking at" inputs by placing the input in the context window. Self-attention then handles input-to-output relationships. Architecturally simpler; functionally similar to encoder-decoder for many tasks.

> **Saying it out loud.** Because they don't need it — they just put the input in the context window. Once everything is one sequence, ordinary self-attention already relates the output tokens to the input tokens; a separate mechanism would be redundant. And having one tower instead of two means one set of weights, one objective, and no decisions about where to put cross-attention layers. The thing you give up is the cheap fixed cross-attention cache, and you pay for it by carrying the prompt in the KV cache for the whole generation.

---

## C. Sliding window and sparse attention

**14. What's sliding window attention?**
Each position attends only to the previous $W$ tokens. Causal triangular mask becomes a band: $M_{i,j} = 0$ if $i - W < j \leq i$, else $-\infty$. Compute $O(N \cdot W)$ instead of $O(N^2)$. KV cache bounded to most recent $W$ per sequence per layer.

> **Saying it out loud.** Sliding window means each token only attends to the last $W$ tokens instead of everything before it. Your causal triangle becomes a narrow band along the diagonal. That turns compute from $N^2$ into $N$ times $W$, and — the part that actually matters in production — it caps the KV cache at $W$ tokens per layer no matter how long the conversation gets. Mistral 7B ships with $W = 4096$, so a 100K-token context costs the same cache as a 4K one.

**15. Why does sliding window still capture long range?**
Stacking. With $L$ layers and window $W$, the top layer has receptive field $L \cdot W$ (each layer extends the effective span by $W$). Mistral 7B ($L = 32, W = 4096$) has theoretical receptive field 131K tokens despite each layer only seeing 4K. Plus global tokens / attention sinks help.

> **Saying it out loud.** Through stacking — it's the same trick as receptive fields in a CNN. Layer one lets a token see 4,000 back; but those tokens already summarized *their* 4,000, so at layer two you're effectively reaching 8,000. With 32 layers and a 4K window, Mistral's theoretical reach is around 131K tokens. The honest caveat is that this is a *blurred* reach, not an exact one — information gets compressed at every hop, so it's fine for gist and bad for needle-in-a-haystack recall, which is exactly where sliding-window models measurably underperform.

**16. What are global tokens?**
A small set of positions that attend to and are attended by all other positions. Short-circuit the receptive-field problem. Used in Longformer (4K window + global tokens like `[CLS]`), BigBird.

> **Saying it out loud.** Global tokens are a small set of positions that everyone can see and that can see everyone. Think of them as a broadcast channel bolted onto an otherwise local model — Longformer's CLS token is the canonical example. They fix the receptive-field problem cheaply: instead of needing 30 layers to route information across a document, anything important can be written to a global token and read anywhere in one hop. Cost is negligible since there are only a handful of them, which is why BigBird and Longformer both combine them with a sliding window.

**17. Trade-off of sliding window?**
Saves compute and KV memory linearly. Cost: information far outside the window can't propagate efficiently to later layers. Quality drops on tasks requiring genuinely long-range exact recall (e.g., needle-in-haystack tests at long range).

> **Saying it out loud.** You buy linear compute and a bounded KV cache; you pay in long-range exact recall. That's a great trade if your workload is 'summarize this long document' and a bad one if it's 'find the one API key buried at token 40,000'. The failure mode is specific and measurable — sliding-window models do noticeably worse on needle-in-a-haystack evals at distances beyond the effective receptive field. So the question to ask is always whether your task needs precise retrieval or just gist.

**18. Why aren't sparse attention patterns more popular in production?**
Modern long-context production models tend to use full attention with efficient kernels (FlashAttention) and KV memory tricks (paged, quantized) rather than approximate sparsity. The quality cost of sparsity is usually unacceptable; the compute savings are achievable other ways.

> **Saying it out loud.** Because the compute savings turned out to be available elsewhere without the quality tax. FlashAttention made full attention memory-efficient, paged and quantized KV caches made the memory manageable, and hardware got faster — so the pressure that motivated sparsity partly evaporated. Meanwhile approximate patterns keep costing you a little quality on exactly the long-range tasks people buy long context for. Add in that irregular sparse patterns are hard to make fast on a GPU, and you get the current state: full attention with good kernels wins.

**19. What's BigBird?**
Block-structured sparse attention: sliding window + global tokens + random sparse pattern. Theoretically retains expressiveness of full attention; empirically reasonable. Less popular than full-attention + FlashAttention.

> **Saying it out loud.** BigBird is three patterns stitched together — a sliding window for local structure, a few global tokens for broadcast, and some random connections for everything else. The random edges are the clever bit: they give you short paths between arbitrary positions, which is what lets the authors prove it retains the expressiveness of full attention. It works fine empirically. It just lost to full attention plus FlashAttention, because a block-sparse pattern that's hard to schedule on a GPU beats dense math less often than the FLOP count suggests.

**20. What's the Reformer trick?**
Locality-Sensitive Hashing. Hash queries and keys; only attend within the same hash bucket. $O(N \log N)$ compute with quality close to full attention. Notable for being ~theoretically nice but not widely used in production.

> **Saying it out loud.** Reformer uses locality-sensitive hashing to guess which tokens are worth comparing. The insight is that softmax attention is dominated by a few large scores, so if you can hash queries and keys such that similar vectors land in the same bucket, you only need to attend within buckets. That gets you $O(N \log N)$ instead of quadratic with surprisingly little quality loss. It's a lovely idea that never made it to production, mostly because the hashing and sorting overhead eats the theoretical win at practical sequence lengths.

---

## D. Linear attention

**21. What's linear attention?**
Replace softmax with a kernel approximation: $\mathrm{attention}(Q, K, V) = \phi(Q) \cdot (\phi(K)^\top V) / (\phi(Q) \cdot \phi(K)^\top \mathbf{1})$ for some feature map $\phi$. Order: compute $\phi(K)^\top V$ first (size $d \times d$), then $\phi(Q) \cdot \cdots$. Result: $O(N \cdot d^2)$ — linear in $N$.

> **Saying it out loud.** Linear attention is a reassociation trick. Softmax forces you to compute the $N \times N$ score matrix first, but if you replace it with a kernel feature map, matrix multiplication is associative and you can compute $\phi(K)^\top V$ first instead — that's a $d \times d$ object, independent of sequence length. Then you multiply the queries against it. Cost goes from $N^2 d$ to $N d^2$, so it's linear in sequence length, and the entire trick is just changing the order you multiply in.

**22. Why does linear attention enable RNN-style decoding?**
The attention has the form $\text{output}_t = \phi(Q_t) \cdot S_t$, where $S_t = \sum_{i \leq t} \phi(K_i) V_i^\top$ is updated as $S_t = S_{t-1} + \phi(K_t) V_t^\top$. So decoding is a **recurrent state update** with constant memory $O(d^2)$ and constant time $O(d^2)$ per step — like an RNN.

> **Saying it out loud.** Because once there's no softmax coupling everything together, the summary you need is just a running sum. The state $S_t$ is the accumulated outer products of keys and values up to time $t$, and generating a new token means adding one more outer product and reading off $\phi(Q_t) S_t$. So decoding is a fixed-size recurrent state update — constant memory, constant time per token, no cache that grows with context. That's an enormous serving advantage, and it's the whole reason people keep coming back to linear attention.

**23. Trade-off of linear vs softmax attention?**
Linear: $O(N)$ compute, constant-memory decode. Quality: usually weaker than softmax, especially for in-context learning and exact recall. Whether linear attention catches up at frontier scale is an open question.

> **Saying it out loud.** Linear attention gives you $O(N)$ training and a constant-size decode state; softmax gives you better quality, especially at recall. The reason is that a fixed $d \times d$ state has to compress the entire past into a bounded object, whereas a KV cache keeps every token verbatim — so copying an exact string from 10,000 tokens back is easy for softmax and hard for linear. In-context learning suffers for the same reason. Whether that gap closes at frontier scale is genuinely open, and hybrids exist precisely because nobody wants to bet on it.

**24. What's Performer?**
Linear attention via Random Fourier Features (Choromanski et al. 2020). $\phi$ chosen to approximate the softmax kernel. Provably good approximation; empirically works at moderate scales.

> **Saying it out loud.** Performer is linear attention with a principled feature map. Instead of picking $\phi$ heuristically, they use random Fourier features chosen so that the inner product $\phi(q) \cdot \phi(k)$ approximates $\exp(q \cdot k)$ in expectation — so you're approximating actual softmax attention, with error bars. That's the appeal: it comes with a proof, not just a benchmark. In practice the approximation variance grows enough at scale that it never displaced softmax, but it's the standard reference for 'linear attention done rigorously'.

**25. What's RWKV?**
A linear-attention model designed specifically for autoregressive generation. Replaces transformer attention with a recurrent rule that's parallelizable in training. Strong open-source models exist (RWKV-4, -5, -6). Not yet at frontier-LLM scale.

> **Saying it out loud.** RWKV is a serious attempt to build a real language model on a recurrent core. It replaces attention with a time-decay-weighted recurrence that can be evaluated in parallel across the sequence during training but runs as a plain RNN at inference — so you train like a transformer and serve with constant memory. There are genuinely usable open models through RWKV-6. The honest caveat is that they haven't been trained at frontier budgets, so 'is the architecture worse or just smaller?' isn't settled.

**26. What's the relationship between linear attention and SSMs?**
Both have $O(N)$ sequence complexity and a recurrent decode form. Dao & Gu's *Transformers are SSMs* (2024) formalized that *fixed-parameter* SSMs and certain linear-attention variants are duals (via structured semiseparable matrices, used in Mamba-2). But Mamba's *selective* SSM has input-dependent $A, B, C$ — that selectivity makes Mamba strictly more expressive than vanilla linear attention. "SSMs are linear attention" holds for non-selective variants but oversimplifies for Mamba.

> **Saying it out loud.** They're closely related, and there's a theorem about it — but the popular one-liner overshoots. Dao and Gu's Transformers-are-SSMs paper showed that non-selective, fixed-parameter SSMs and certain linear-attention variants are two views of the same structured matrix, which is what Mamba-2 is built on. But Mamba's actual selling point is *selectivity*: the state-transition parameters depend on the input, so it can decide what to remember and what to forget. That input dependence puts it strictly beyond vanilla linear attention, so 'SSMs are just linear attention' is true for the non-selective case and misleading for Mamba.

---

## E. FlashAttention quick recap

**27. What's FlashAttention?**
I/O-aware tiled attention with online softmax. Same FLOPs as standard attention; far less memory access. Tiles $Q, K, V$ to fit in SRAM; computes partial softmax statistics block-by-block. 2–4× wall-clock speedup at long sequences.

> **Saying it out loud.** FlashAttention doesn't do less math, it does less waiting. The naive kernel writes the full $N \times N$ score matrix out to GPU main memory and reads it back — and at long sequence lengths you're bottlenecked on that traffic, not the arithmetic. FlashAttention tiles Q, K, and V into blocks small enough to live in on-chip SRAM and uses an online softmax to accumulate the answer block by block. Identical FLOPs, identical numerics, 2 to 4x faster in wall clock, and memory drops from quadratic to linear.

**28. Online softmax mechanism?**
Maintain running max $m$ and running sum $s$. For each new value: $m' = \max(m, x); s' = s \cdot \exp(m - m') + \exp(x - m'); m = m'; s = s'$. Single pass, numerically equivalent to two-pass softmax. Lets attention be computed block-by-block without materializing the full matrix.

> **Saying it out loud.** Online softmax is how you normalize a stream without seeing all of it first. Normally softmax needs two passes — find the max for stability, then sum the exponentials. Instead you keep a running max and a running sum, and every time you see a bigger max you rescale the sum you've accumulated so far by $\exp(m_{old} - m_{new})$. The result is numerically identical to the two-pass version in one pass, which is exactly what lets FlashAttention finish a block and throw it away instead of storing the whole matrix.

**29. Does FlashAttention reduce theoretical complexity?**
No. Same FLOPs. Reduction is in memory access: from $O(N^2)$ reads/writes to HBM down to $O(N)$ for intermediate steps.

> **Saying it out loud.** No, and that's the point people get wrong. The FLOP count is unchanged — it's the same attention, same numbers, same result bit-for-bit up to reduction order. What changes is memory traffic: reads and writes to HBM drop from quadratic in sequence length to linear, because the intermediate score matrix never gets written out. It's a reminder that on modern GPUs the arithmetic is often free and the memory bandwidth is what you're actually paying for.

---

## F. Attention head analyses

**30. What's an induction head?**
A head that copies a token from earlier in the context when a similar prefix recurs. E.g., if the context has "X Y ... X", an induction head attends from the second X to Y and copies. Mechanism for in-context learning. Olsson et al. 2022 ("In-context Learning and Induction Heads").

> **Saying it out loud.** An induction head is a copy-what-happened-last-time mechanism. If the context contained 'Alice went to Paris' earlier and now says 'Alice went to', an induction head spots that the current prefix matches an earlier one and copies whatever followed it. Mechanically it's two heads working together — a previous-token head writes 'the token before me was X' into the stream, and the induction head matches on that and copies forward. Olsson et al. 2022 is the reference, and this is the closest thing we have to a mechanistic account of in-context learning.

**31. Why do induction heads matter?**
Their emergence in training correlates with the transition to in-context learning ability. Removing them ablates ICL. Provides a mechanistic story for how transformers do few-shot learning.

> **Saying it out loud.** Because they're the best evidence we have that in-context learning has a *mechanism*, not just a vibe. During training there's a visible bump in the loss curve, and it coincides almost exactly with induction heads forming — and after that point the model can suddenly do few-shot tasks. Ablate them and in-context learning degrades sharply. It's rare in interpretability to have a circuit, a capability, and a phase transition all line up, which is why this result gets cited so much.

**32. What's a previous-token head?**
A head that attends from position $t$ to position $t-1$. Used in counting and copying tasks. Common in lower layers of trained transformers.

> **Saying it out loud.** A previous-token head does exactly what the name says: from position $t$ it attends almost entirely to position $t-1$. Its attention map is a clean off-diagonal stripe. On its own it looks trivial, but it's the first half of the induction circuit — it's what stamps 'the token before me was X' into each position's residual stream so a later head can match on it. You find them reliably in the early layers of basically every trained transformer.

**33. Why are some heads ablation-resistant?**
Empirical observation that you can remove many heads without hurting quality (Voita et al., Michel et al.). Suggests redundancy: multiple heads compute similar functions; removing one doesn't fail because others cover it. Doesn't generalize: some heads are essential, and their identity differs across models.

> **Saying it out loud.** Because the model builds in a lot of redundancy — several heads end up computing near-duplicate functions, so knocking one out just shifts the work. Voita and Michel both showed you can prune a large fraction of heads with little quality loss. But don't overstate it: a minority of heads are genuinely load-bearing and removing those hurts immediately, and which ones they are differs between models and even between training runs. So the takeaway is 'most heads are redundant, a few are critical, and you have to measure which is which'.

**34. What does an "attention sink" look like?**
The first few tokens (often `[BOS]` or just position 0) attract disproportionate attention from many heads. They act as a "sink" — heads that don't have anything specific to attend to dump attention there. StreamingLLM showed that removing attention sinks from the KV cache breaks long-context behavior.

> **Saying it out loud.** It looks like a bright vertical stripe on the first column of the attention map — nearly every head, in nearly every layer, dumping a big chunk of its attention on token zero. The reason is structural: softmax weights must sum to one, so a head that has nothing it actually wants to look at still has to put its mass somewhere, and it learns to park it on a token that carries no information. The consequence is a real production gotcha — StreamingLLM found that if you evict those first tokens while sliding a window, the model's outputs fall apart, so you pin the first few tokens permanently.

**35. Why does temperature appear in attention scores?**
The $1/\sqrt{d_k}$ is effectively a temperature on the softmax. Standard convention: scores divided by $T$, so larger $T$ (here $\sqrt{d_k}$) → softer/smoother distribution; smaller $T$ → sharper. We *want* a smoother distribution at large $d_k$ to prevent softmax saturation, so dividing by $\sqrt{d_k}$ — which grows with $d_k$ — is exactly right.

> **Saying it out loud.** Because dividing scores by a constant is literally what temperature does in a softmax. Small divisor means sharp, peaky attention; big divisor means smooth, spread-out attention. The reason $\sqrt{d_k}$ is the right choice is that raw score magnitudes grow like $\sqrt{d_k}$ as you widen the head, so without correction, wider heads would automatically get sharper and eventually saturate. Dividing by exactly $\sqrt{d_k}$ cancels that growth, so a 128-dim head and a 64-dim head start out with comparable attention sharpness rather than the wide one being frozen from step one.

---

## G. Production and engineering

**36. What's "paged attention"?**
KV cache allocation in fixed-size blocks (vLLM). Eliminates fragmentation. Block table per request maps logical positions to physical blocks. See `06_llm_inference/`.

> **Saying it out loud.** Paged attention is virtual memory for the KV cache. Instead of reserving one big contiguous block per request sized for the worst case, vLLM chops the cache into fixed-size pages and keeps a per-request table mapping logical token positions to physical pages. That kills fragmentation and the massive over-allocation you get from sizing for max length. In practice it's a several-fold improvement in how many concurrent requests fit on a GPU, which is why it became the standard serving design.

**37. What's the difference between "pre-fill attention" and "decode attention" in serving systems?**
Pre-fill: process the entire prompt in one parallel forward pass. Compute-bound (large matmuls). Decode: one-token-at-a-time autoregressive. Memory-bound (read full weights for one token). Different optimal kernels for each phase (FlashAttention vs FlashDecoding).

> **Saying it out loud.** They're two completely different performance regimes wearing the same math. Pre-fill processes the whole prompt at once, so it's big matmuls and you're compute-bound — the GPU is actually busy. Decode generates one token at a time, so for every token you drag the entire model's weights and the whole KV cache out of memory to do a tiny amount of arithmetic, and you're memory-bandwidth-bound. That's why serving stacks use different kernels for each — FlashAttention for pre-fill, FlashDecoding for decode — and why batching helps decode enormously but pre-fill barely at all.

**38. Why does the KV cache grow linearly with context?**
Each new token contributes one $K$ vector and one $V$ vector per layer per head. KV cache size for a sequence of length $N$: $2 \cdot n_{\text{kv-heads}} \cdot d_h \cdot N \cdot L \cdot \text{bytes}$. Linear in $N$. This is the fundamental memory cost of long context.

> **Saying it out loud.** Because every token you process permanently adds one key and one value vector, per layer, per KV head. Nothing gets discarded, so the cache is $2 \cdot n_{kv} \cdot d_h \cdot N \cdot L$ times your dtype size, and it's dead linear in $N$. Put numbers on it: a 70B model with GQA-8 costs roughly 300KB per token, so a 100K-token context is about 30GB of cache — more than the weights of a smaller model. That linear growth is the fundamental cost of long context and the reason GQA, MLA, quantized caches, and paging all exist.

**39. What's the receptive field of layer $L$ in a sliding-window-$W$ transformer?**
$L \cdot W$. Each layer extends the effective receptive field by $W$ because layer $L+1$ can attend to layer $L$'s outputs at positions $i, i-1, \ldots, i-W$, each of which represents $i-W, i-2W$, etc.

> **Saying it out loud.** $L$ times $W$, because each layer extends the reach by another window. At layer one you can see $W$ tokens back; each of those tokens already aggregated its own $W$, so by layer two you're indirectly reaching $2W$, and it compounds linearly with depth. That's how Mistral claims a 131K reach from a 4K window across 32 layers. The caveat worth adding is that this is theoretical reach through multiple hops of averaging, so the information is increasingly blurred the further out you go.

**40. Sliding window + global tokens — why combine?**
Sliding window gives $O(N \cdot W)$ compute and bounded KV memory. Global tokens give a small number of "broadcast" channels that don't suffer the receptive-field limitation. Combined, you keep most of sliding window's efficiency while preserving long-range information through the globals.

> **Saying it out loud.** Because they fail in opposite directions. Sliding window is cheap and gets you local structure but starves long-range information, since it has to travel layer by layer to get anywhere. Global tokens are the opposite — a one-hop path to anywhere, but you can only afford a handful of them. Put them together and you keep the linear compute and bounded cache from the window while giving the model a few broadcast channels for anything genuinely long-range. That's the Longformer and BigBird recipe.

---

## H. Quick fire

**41.** *Original MHA paper?* Vaswani et al. 2017.
**42.** *MQA paper?* Shazeer 2019.
**43.** *GQA paper?* Ainslie et al. 2023.
**44.** *MLA paper?* DeepSeek-V2, 2024.
**45.** *Standard $n_{\text{kv-heads}}$ for LLaMA-2 70B?* 8.
**46.** *Default Mistral sliding window?* 4096 tokens.
**47.** *Linear attention sequence complexity?* $O(N \cdot d^2)$.
**48.** *FlashAttention sequence complexity?* Same $O(N^2 \cdot d)$ FLOPs as standard, lower memory access.
**49.** *Reformer complexity?* $O(N \log N)$.
**50.** *Induction head function?* Copy a token after prefix recurrence.

---

## Self-grading

If you can't answer 1-10, you don't know modern attention variants. If you can't answer 11-25, you can't pass an LLM-architecture round. If you can't answer 26-50, frontier-lab interviews will go deeper than you can follow.

Aim for 35+/50 cold.
