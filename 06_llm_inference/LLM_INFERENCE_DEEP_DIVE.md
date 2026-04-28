# LLM Inference: A Frontier-Lab Interview Deep Dive

> **Why this exists.** LLM inference is where applied scientist / ML engineer interviews go to separate candidates. The questions are technical (memory math, paged attention), system-design heavy (latency vs throughput), and rapidly evolving (FP8, speculative decoding, prefill chunking). This document covers what frontier labs and big-tech infra teams actually probe.

---

## 1. The two phases that make LLM inference unusual

Every LLM-generation request has two phases with fundamentally different bottlenecks:

### Prefill (a.k.a. prompt processing)
You're given the prompt of length $P$. You run one forward pass over all $P$ tokens in parallel, producing the first output token and populating the KV cache.

- **Compute pattern:** $P$ tokens × $d_{\text{model}}$ × matmul shapes — large, dense matrix-matrix multiplies.
- **Bottleneck:** **compute-bound** on modern GPUs. Tensor cores are saturated; arithmetic intensity is high.
- **Cost:** $O(P^2)$ for attention (full attention over the prompt), $O(P \cdot d_{\text{model}})$ for FFN.

### Decode (a.k.a. generation)
You generate output tokens one at a time. Each step:
- Take the most recent token, embed it.
- Run a forward pass that attends to the full KV cache (length $P + i$ so far).
- Sample the next token.
- Append to KV cache.
- Repeat.

- **Compute pattern:** 1 token × $d_{\text{model}}$ × matmul shapes — small matrix-vector multiplies.
- **Bottleneck:** **memory-bandwidth-bound**. You're reading the entire model's weights from HBM for one token's worth of compute. Tensor cores sit idle.
- **Cost per token:** $O((P+i) \cdot d_{\text{model}})$ for attention (KV cache read), $O(d_{\text{model}}^2)$ for FFN (weight read).

**The single most important fact about LLM inference:** prefill is compute-bound, decode is memory-bound. Almost every optimization in this document tries to address one of those two regimes specifically.

---

## 2. The arithmetic-intensity / roofline view

Arithmetic intensity = FLOPs per byte of memory accessed. The roofline model says:
- High intensity → compute-bound (peak limited by FLOPS).
- Low intensity → memory-bound (peak limited by bandwidth).

For a 70B model running on an H100 (~3 TB/s HBM, ~989 TFLOPS bf16 dense):
- **Prefill** with batch size 1, prompt length 2K: arithmetic intensity ~$P$ tokens × ops per byte → high → compute-bound.
- **Decode** with batch size 1: intensity $\approx 1$ op per byte → memory-bound.

The "balance point" of an H100 is around 330 ops/byte. Decode at batch size 1 is around 2 ops/byte — orders of magnitude below balance. **The fix is batching.**

Decode with batch size $B$: intensity scales roughly with $B$ (you read each weight once but reuse it for $B$ queries). So at batch size ~256+, decode starts to become compute-bound on H100. This is the entire reason continuous batching exists: it lets you reach the compute-bound regime by collecting many concurrent decode requests.

**Frontier-lab interview question:** "Why is decode slow even on the fastest GPU?" Memory-bandwidth-bound. Each output token requires reading the entire model weights from HBM. Speedup = batching to amortize the weight read.

---

## 3. KV cache: the central data structure

### What it is
Keys and values for every layer, every head, for every position seen so far. Reused on every subsequent decode step.

### Memory size

For one sequence:

$$
\text{KV\_size} = 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \cdot \text{seq\_len} \cdot \text{bytes\_per\_element}
$$

(factor of 2 for $K$ and $V$). With $n_{\text{heads}} \cdot d_{\text{head}} = d_{\text{model}}$ in standard MHA, this simplifies to $2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot \text{seq\_len} \cdot \text{bytes}$.

For batch size $B$: multiply by $B$.

**Concrete example: LLaMA-2 70B (which uses GQA-8) at fp16, 8K context, batch 32:**
- $2 \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{kv\_heads}} \cdot d_{\mathrm{head}} \cdot \mathrm{seq\_len} \cdot \mathrm{bytes} \cdot B$
- $= 2 \cdot 80 \cdot 8 \cdot 128 \cdot 8192 \cdot 2 \cdot 32 \approx \mathbf{86\text{ GB}}$

(If LLaMA-2 70B used full MHA with 64 heads, it would be $\sim 687$ GB — the $8\times$ saving from GQA-8 is exactly why modern LLMs adopt it.) The model weights alone are 140 GB at fp16. **KV cache often dominates memory in production**, especially with long context — and GQA/MQA/MLA exist to fight this.

### Why it matters for serving

KV cache memory is the binding constraint that limits how many concurrent users you can serve. Two requests with the same prompt length use 2x the KV cache. The serving system's max batch size is set by KV memory, not by compute.

### Reducing KV cache size (the active research frontier)

| Technique | Mechanism | Savings |
|---|---|---|
| MQA (Multi-Query Attention) | Single $K$ and $V$ across all heads | $n_{\text{heads}}$× reduction |
| GQA (Grouped-Query Attention) | Groups of heads share $K/V$ | $n_{\text{heads}} / n_{\text{groups}}$× |
| PagedAttention (vLLM) | Block-wise allocation, no fragmentation | Effectively 2–4x more concurrent users |
| Quantized KV cache | Store $K/V$ in int8 or int4 | 2–8x reduction |
| Sliding window attention | Drop old tokens | depends on window |
| YOCO / shared KV across layers | Layer-wise cache reuse | up to $n_{\text{layers}}$× |

LLaMA-2 70B uses GQA with 8 KV heads (vs 64 attention heads), giving an 8x reduction in KV cache size compared to plain MHA.

---

## 4. PagedAttention and the vLLM contribution

This is essential interview material as of 2024–2026.

### The problem
Naive KV cache allocation reserves `max_seq_len` worth of memory per request, even if the request only generates 100 tokens. Most of that is wasted — internal fragmentation. External fragmentation also occurs as requests of different lengths come and go.

### The fix (Kwon et al. 2023, vLLM)
Allocate KV cache in **fixed-size blocks** (e.g. 16 tokens each), like virtual-memory pages. A block table per request maps logical positions to physical blocks. Blocks are allocated on demand as the sequence grows.

### Benefits

- **No internal fragmentation.** You only allocate as many blocks as you've used.
- **No external fragmentation.** Free blocks fit any future request.
- **Memory sharing.** Multiple sequences with a common prefix can share blocks (relevant for beam search, parallel sampling, and prompt-prefix caching across requests).
- **2-4x more concurrent requests** for the same GPU memory in practice.

### Trade-offs

- Slightly more complex attention kernel: must follow block table during attention computation.
- Block size is a hyperparameter (typically 16): too small = overhead, too large = back to fragmentation.

vLLM and TensorRT-LLM both implement variants. Frontier-lab interviews on serving systems will ask about this by name.

---

## 5. Continuous batching (a.k.a. iteration-level scheduling)

Naive batching: collect `B` requests, run them all together, return results when **all** are done. Problem: short requests wait for the longest one.

Continuous batching (Yu et al. 2022, Orca paper): **at each decode step**, evict completed requests from the batch and admit new ones. Decode is the bottleneck, and decode is per-token, so you can rebalance the batch every token.

### Why it works
- Decode work per step is approximately constant per request.
- Adding/removing requests between steps is cheap (just KV cache pointers).
- Throughput goes up because GPU is never idle waiting for stragglers.

### Why it's hard

- Adding a new request mid-batch requires running its prefill, which has different compute characteristics than decode. Modern systems handle this with "chunked prefill" or by interleaving prefill and decode steps.
- The optimal scheduling is a function of compute, memory, and SLA targets — non-trivial.

**vLLM, TensorRT-LLM, TGI, and SGLang all implement continuous batching.** Without it, GPU utilization for serving is often <30%; with it, >70% is achievable.

---

## 6. FlashAttention and I/O-aware attention

Standard attention has `O(N²)` compute but also `O(N²)` memory access (the attention matrix `QK^T` materializes in HBM).

### The fix (Dao et al. 2022, FlashAttention)

Tile the attention computation. Compute `softmax(QK^T) V` in blocks that fit in SRAM (the GPU's on-chip cache). Use the **online softmax algorithm** (Milakov & Gimelshein) to compute partial softmax statistics block-by-block without materializing the full `N×N` matrix.

### Result

- Memory access drops from `O(N²)` to `O(N)` for the intermediate steps.
- Wall-clock speedup of 2-4x on long sequences.
- Same numerical result as standard attention (not an approximation).

### FlashAttention-2 and FlashAttention-3

- **FA2 (2023):** better parallelization, reduced non-matmul ops. Another ~2x speedup.
- **FA3 (2024):** uses Hopper/Blackwell-native features (TMA, async warpgroup matmul, optional FP8). Again ~2x. Closer to peak bf16 utilization on H100.

### Why this matters for inference

- Prefill: directly accelerates the `O(N²)` attention computation. For long prompts (8K+), 2-4x faster prefill.
- Decode: less direct since attention per step is `O(N)`, not `O(N²)`. But specialized decode kernels (FlashDecoding, FlashDecoding++) bring similar I/O optimizations to the decode case.

If asked "why is FlashAttention faster?" — the answer is **I/O-aware tiling combined with online softmax**, not algorithmic complexity reduction. Same FLOPs, far fewer memory transactions.

---

## 7. Speculative decoding

The most-asked LLM-inference algorithm question of 2024.

### The setup

Decode is memory-bound: each step reads the full model from HBM to produce one token. If you could verify multiple tokens in one forward pass, you'd amortize the memory read.

### The trick (Leviathan et al. 2023, Chen et al. 2023)

Use a small **draft model** to propose `k` tokens autoregressively (cheap because it's small). Then run the **target model** in **a single forward pass over those `k` tokens** to verify them. Accept tokens up to the first rejection.

### The acceptance rule (rejection sampling)

For each proposed token $x_i$:
- Let $p(x_i \mid \text{prefix})$ = target model's probability of $x_i$.
- Let $q(x_i \mid \text{prefix})$ = draft model's probability of $x_i$.
- **Accept with probability $\min(1, p/q)$.**
- If accepted, append $x_i$; if rejected, sample from $(p - q)_+ / \|(p-q)_+\|_1$ (the "residual distribution") and stop.

This rule guarantees the **distribution of accepted tokens is exactly the target model's distribution**. Speculative decoding is *exact* — same outputs as plain decoding, in distribution.

### What determines speedup

$$
\text{speedup} \approx \frac{1 + \alpha + \alpha^2 + \cdots + \alpha^k}{1 + (\text{draft\_cost} / \text{target\_cost}) \cdot k}
$$

where $\alpha$ is the per-token acceptance rate. When $\alpha$ is high (e.g. 0.7+) and $\text{draft\_cost} \ll \text{target\_cost}$, you can get 2–3x speedup. When $\alpha$ is low, you waste forward passes verifying rejected tokens.

### Variants

- **Vanilla speculative decoding.** Separate draft model.
- **Self-speculative.** Use shallow layers of the target as draft. No external model needed.
- **Lookahead decoding.** Use n-gram or Jacobi iteration to propose tokens. No external model.
- **Medusa, EAGLE.** Train extra heads on the target to predict multiple tokens.
- **Tree speculation.** Propose a tree of candidate continuations; verify with one target forward pass.

### Common interview gotchas

- **It's exact, not approximate.** This is the key correctness property.
- **Draft model quality matters.** A 7B drafting for a 70B works well (high acceptance). A 1B drafting for a 70B works less well.
- **Latency vs throughput.** Speculative decoding helps **per-request latency**. It can hurt aggregate throughput because total compute increases (you do compute on rejected tokens). On a busy server already running large batches, naive batching may be better.

---

## 8. Quantization for inference

Reducing precision saves memory and (often) compute. The major dimensions:

### Weight quantization

| Format | Bits | Trade-off |
|---|---|---|
| FP32 | 32 | Reference; rarely used in inference |
| BF16 / FP16 | 16 | Common baseline |
| FP8 (E4M3, E5M2) | 8 | Hopper / Blackwell native; minimal quality loss |
| INT8 | 8 | Common for serving; ~2x speedup |
| INT4 | 4 | ~4x memory; quality loss varies by method |
| W2A16 | 2 | Aggressive; usually round-trip with retraining |

### KV cache quantization

Stores K and V at lower precision. Memory savings without changing weights or compute. INT8 is common; INT4 requires care (group-wise quantization).

### Activation quantization

Activations have **outliers** (a few channels are 100x bigger than typical). Naive INT8 activation quantization clips outliers and degrades quality. **SmoothQuant** (Xiao et al. 2022) shifts outlier magnitudes from activations to weights pre-quantization, restoring INT8 quality.

### Calibration-based methods

| Method | Idea | When to use |
|---|---|---|
| **GPTQ** (Frantar et al. 2022) | Per-layer Hessian-based weight quantization | INT4 weights, minimal calibration data |
| **AWQ** (Lin et al. 2023) | Activation-aware weight quantization (preserve weights with high activation impact) | INT4 weights, faster than GPTQ |
| **SmoothQuant** | Re-balance activations and weights | INT8 weights + INT8 activations |
| **SpQR** (Dettmers et al. 2023) | Sparse + quantized: keep top outliers in fp16 | Best-quality INT4 |
| **NF4** (Dettmers et al. 2023, QLoRA) | Information-theoretically optimal 4-bit format for normally-distributed weights | LoRA fine-tuning of quantized models |

### What to know in interviews

- "Why does INT8 quantization sometimes hurt?" Activation outliers (SmoothQuant fix) and calibration data mismatch.
- "What's W4A16?" 4-bit weights, 16-bit activations. Most common LLM inference quantization. Used by GPTQ, AWQ.
- "What's W8A8?" 8-bit weights, 8-bit activations. Compute uses INT8 tensor cores (often ~2x faster than fp16). SmoothQuant is the canonical method.
- "What's FP8?" 8-bit float with 4 or 5 mantissa bits. Native on H100/B100. No quantization-aware compute path needed; less calibration sensitivity than INT8.

---

## 9. Multi-GPU inference: parallelism

Single GPU isn't enough for 70B+ models. Three orthogonal parallelism axes:

### Tensor parallelism (intra-layer)
Split each weight matrix across GPUs. Multiply locally; all-reduce results. Each forward pass requires inter-GPU communication. Used widely for inference; dominates above ~13B models.

### Pipeline parallelism (inter-layer)
Different GPUs hold different layers. Forward passes flow GPU-to-GPU. **Avoid in low-latency inference** because of pipeline bubbles. Used for training, less for inference.

### Expert parallelism (MoE)
Different GPUs hold different experts. Routing decides which GPU(s) handle each token. Used for MoE models like Mixtral, GPT-4-class.

### Data parallelism (across requests)
Each GPU holds a full copy of the model and serves different requests. Embarrassingly parallel. Standard for small models or when memory allows it.

**For inference, tensor parallelism within a node + data parallelism across nodes is the typical setup.** Pipeline parallel is more common in training than inference because of latency cost.

---

## 10. Latency metrics: TTFT and TPOT

The two numbers that matter most in serving:

- **TTFT (Time To First Token).** From request arrival to the first generated token. Dominated by **prefill** time, which scales with prompt length. Critical for chat UX.
- **TPOT (Time Per Output Token), a.k.a. ITL (Inter-Token Latency).** Average time between consecutive output tokens during decode. Dominated by memory bandwidth and batch utilization.

A streaming chatbot wants:
- TTFT < ~500ms for "feels responsive."
- TPOT < ~50ms for natural reading pace.

These are different optimizations. TTFT is helped by prefill chunking, FlashAttention, fewer layers. TPOT is helped by batching, KV cache compression, speculative decoding.

**Interview question:** "Your model has 200ms TTFT and 30ms TPOT — how would you reduce TTFT without hurting TPOT?" Prefill chunking, FlashAttention, smaller initial pass with retrieval. Don't lower batch size (would hurt throughput).

---

## 11. Throughput vs latency: the tension

Throughput (tokens/sec across all users) and latency (per-request token time) are in tension. Bigger batches = higher throughput, slower per-request decode.

In practice, serving systems target a **latency SLA** (e.g. p99 TPOT < 100ms) and maximize throughput subject to the SLA. Continuous batching, paged attention, and speculative decoding all push the Pareto frontier outward.

**Interview question:** "How do you choose batch size for serving?" Set the latency SLA, find the largest batch that meets it under realistic mixed workload, monitor and adjust. Workload mix matters: long contexts need smaller batches to fit KV memory; short contexts allow bigger batches.

---

## 12. Prompt caching and prefix sharing

If many requests share a common prefix (system prompt, RAG context, multi-turn conversation), the prefix's KV cache can be **computed once and shared** across requests. This is especially valuable for:

- Multi-turn chats (the previous turns are shared).
- RAG (the retrieved documents are shared across users asking similar questions).
- Function-calling agents (long tool-use prompts).

PagedAttention's block-level reference counting makes this natural. Anthropic and OpenAI both expose prompt caching as a feature, with significant cost reduction for long-prefix workloads.

---

## 13. The "cost per token" mental model

For interview-grade reasoning about serving cost:

$$
\text{Cost per output token} \approx \frac{\text{model\_params} \cdot \text{bytes\_per\_param} + \text{KV\_cache\_bytes\_per\_token}}{\text{memory\_bandwidth}} + \text{overhead}
$$

Concretely, for a 70B fp16 model at $70\text{B} \times 2 = 140$ GB. (Note: 140 GB doesn't fit on a single 80 GB H100 — this analysis assumes the weights are sharded across $\geq 2$ GPUs via tensor parallelism, with aggregate HBM bandwidth scaling roughly linearly.)
- Per-GPU HBM bandwidth ~3 TB/s; with TP=2, aggregate ~6 TB/s reading $\sim 70$ GB per GPU.
- Theoretical lower bound on per-token decode time ≈ 140 GB / 6 TB/s ≈ 23 ms.
- This is an absolute floor for the configuration — no software can beat it without quantization, speculation, or denser packing.

This calculation tells you what's achievable. Quantization (W4A16) drops the weight read to 35 GB → fits on one H100 → faster per-step. Speculative decoding multiplies effective throughput by acceptance rate. The intent of the exercise: latency on memory-bound decode is set by *bytes read per step / HBM bandwidth*, not compute.

**Interview question:** "What's the theoretical maximum decode speed of a 70B model on one H100?" Compute it from memory bandwidth. ~21 tok/s in fp16, ~80 tok/s in 4-bit, more with speculation.

---

## 14. The 2024-2025 inference frontier

These are the techniques frontier interviews increasingly ask about.

### MLA (Multi-head Latent Attention) — DeepSeek-V2/V3

The biggest KV-cache reduction in years. Project keys and values into a low-rank latent before caching:

$$
c_t^{KV} = W_{DKV} h_t \quad (\text{compress to dim } d_c \ll n_h \cdot d_h)
$$

Cache only $c_t^{KV}$. At attention time, project back up via two learned matrices to recover $K$ and $V$. KV cache shrinks ~10× vs MHA, ~3-4× vs GQA-8.

**Subtlety**: RoPE doesn't commute with the down-projection. DeepSeek splits each head into a "RoPE part" (kept full-dim, no compression) and a "no-RoPE part" (compressed). This is why MLA descriptions get complicated.

**Why this matters**: with 128K-context inference, KV cache dominates memory. MLA shrinks it enough that DeepSeek-V3's 671B-parameter model is *cheaper to serve per token* than smaller GQA-only models.

### Chunked prefill

Long prompts cause two problems: (a) prefill takes too long → bad TTFT; (b) the long prefill blocks decode steps for other batched requests. **Chunked prefill** splits a long prefill into chunks of ~512–2K tokens, interleaved with decode steps from other requests in the same batch.

Result: better TTFT for long-prompt requests, no decode starvation. Standard in vLLM 0.6+, SGLang, TensorRT-LLM.

### Disaggregated serving (DistServe, Mooncake)

Prefill and decode have completely different bottlenecks (compute-bound vs memory-bound). Using the same GPUs for both is wasteful. **Disaggregated serving** splits the cluster:

- **Prefill workers**: optimized for compute (large batches, lots of tensor cores busy).
- **Decode workers**: optimized for memory bandwidth (KV cache locality, smaller batches).
- KV cache transferred between them after prefill finishes.

Used by frontier serving systems (Mooncake — Moonshot AI's Kimi serving system, DistServe, Sglang router). Offers 2-4× higher goodput at fixed SLO compared to monolithic serving.

### Advanced speculative decoding: Medusa, EAGLE, lookahead

The basic speculative-decoding setup (small draft model + big target model) has variants:

- **Medusa**: instead of a separate draft model, add multiple "Medusa heads" to the target model that predict the next-2, next-3, etc. tokens. No separate draft model needed; ~2× speedup with slightly trained extra heads.
- **EAGLE**: train a small "feature predictor" that uses the target model's penultimate-layer features. Higher acceptance rate (60–80%) than vanilla draft models because it shares representation. Used in production at multiple labs.
- **Lookahead decoding** (Medusa variant): use the target model's own n-gram completions from early layers as draft. No extra parameters.

**EAGLE-2 / EAGLE-3** push acceptance rates to 70–85% by adding more sophisticated tree-structured drafts.

### KV cache eviction (StreamingLLM, H2O, FastGen)

For very long contexts, even with MLA the KV cache is huge. **Evict** less-important tokens.

- **StreamingLLM**: keep "attention sinks" (first 4 tokens — empirically dominate attention) plus a sliding window of recent tokens. Evict the middle. Works because most attention concentrates on either very early or very recent tokens.
- **H2O (Heavy Hitter Oracle)**: track which tokens accumulate the most attention; evict the rest. Adapts per-prompt.
- **SnapKV / Quest**: more sophisticated eviction policies.

Trade-off: faster + cheaper but lossy — the evicted KV is gone. Used for very long context, less common at 8K-32K.

### KV cache quantization

Quantize K and V to INT8 or INT4 in the cache. Common:

- **KIVI**: per-channel for K, per-token for V (matches their distribution shapes).
- **KVQuant**: more sophisticated outlier-aware scheme.
- **FP8 KV**: native on H100, easiest path; ~2× cache reduction at minimal quality loss.

Combined with MLA / GQA, KV memory becomes manageable even at 128K+.

### FP4 / FP6 quantization (Blackwell-era)

NVIDIA Blackwell GPUs natively support FP4 (W4A4) matmul. Inference can run at $\sim 4 \times$ FP16 throughput, modulo accuracy loss.

- **FP4 weights, FP8 activations**: production sweet spot for Blackwell.
- **FP4 weights, FP4 activations**: maximum throughput, requires careful calibration.
- **NVFP4 / MXFP4**: micro-block formats with per-block scales — better numerical properties than naive FP4.

This is the next big quantization wave — most frontier serving will use these by mid-2025.

### Continuous batching internals

Specifically: how does the scheduler choose which requests to admit / pause / run?

- **First-come-first-served (FCFS)**: simple, fair, suboptimal.
- **Prioritized**: favor short requests for better tail latency.
- **Memory-aware**: don't admit a new request if its expected KV consumption would push existing requests out.
- **Preemption**: if memory tight, pause some requests, swap their KV to CPU, resume later.

The scheduler is the load-bearing piece in production serving. vLLM, SGLang, TensorRT-LLM each have somewhat different policies.

### Prompt caching at scale (Anthropic / OpenAI)

Beyond block-level prefix sharing within a batch, providers cache long prompt prefixes across requests / users for hours. With pricing-tier discounts for cache hits (e.g., 90% off cached input tokens). Critical for agentic workloads where long system prompts + tool definitions repeat across thousands of requests.

### Prefill-decode latency budget in modern systems

For chat applications: **TTFT < 500 ms, TPOT < 50 ms** is the rough target. For voice (e.g., real-time conversation): **TTFT < 200 ms, TPOT < 30 ms**. Achieving these on 70B+ models requires the full stack: GQA/MLA + chunked prefill + continuous batching + speculative decoding + FP8/FP4.

### Long-context architectures: Ring Attention

For sequences too long to fit on one GPU: **Ring Attention** distributes the sequence across GPUs, passes K/V slices in a ring while each GPU computes its query's attention against the rotating K/V. Works because attention only needs partial K/V at a time. Used to train and serve million-token models (Gemini 1.5 Pro and later).

Cost: extra inter-GPU communication. Pairs naturally with FlashAttention.

---

## 15. The 12 most-asked LLM-inference interview questions

(Brief answers; full grilling in `INTERVIEW_GRILL.md`.)

1. **Why is decode memory-bound?** You read the full model weights for one token of compute → arithmetic intensity ≈ 1.
2. **What does KV cache cost in memory?** $2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot \text{seq\_len} \cdot \text{bytes} \cdot \text{batch}$.
3. **What's PagedAttention?** Block-wise KV allocation eliminating fragmentation; enables 2-4x more concurrent users.
4. **What's continuous batching?** Iteration-level scheduling: rebuild the batch every decode step.
5. **Walk me through speculative decoding.** Draft model proposes; target model verifies in one pass; rejection sampling guarantees exact target distribution.
6. **What's FlashAttention?** I/O-aware tiled attention with online softmax; same FLOPs, fewer memory transactions, 2-4x faster.
7. **What's W4A16?** 4-bit weights, 16-bit activations; standard for LLM inference.
8. **What's the difference between GPTQ and AWQ?** Both INT4 weight quantization; GPTQ is Hessian-based per-layer, AWQ preserves activation-important weights. AWQ is faster to apply.
9. **What's MQA vs GQA?** MQA = one K/V across all heads (max KV savings, some quality loss); GQA = groups of heads share K/V (compromise; LLaMA-2 default).
10. **What's TTFT vs TPOT?** Time to first token (prefill cost) vs time per output token (decode cost).
11. **Why does throughput rise but latency get worse with bigger batches?** Bigger batch amortizes weight read (good for throughput) but each request shares compute (bad for per-request latency).
12. **What's prefix caching?** Reusing computed KV cache for shared prompt prefixes across requests; large cost win for chat and RAG.

---

## 16. Recommended drill plan

1. Memorize the prefill-vs-decode dichotomy and the arithmetic-intensity argument.
2. Whiteboard the KV cache memory formula. Compute it for one model you care about.
3. Whiteboard the speculative decoding rejection rule.
4. Know FlashAttention's mechanism (tiling + online softmax) — not just "it's faster."
5. Know the four major quantization methods (GPTQ, AWQ, SmoothQuant, FP8) and when each applies.
6. Drill `INTERVIEW_GRILL.md`.

---

## 17. Further reading

- Pope et al., "Efficiently Scaling Transformer Inference" (2022) — foundational paper for serving math.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023).
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (2022) — continuous batching origin.
- Dao et al., "FlashAttention" (2022), "FlashAttention-2" (2023), "FlashAttention-3" (2024).
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023).
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (2023, DeepMind).
- Frantar et al., "GPTQ" (2022).
- Lin et al., "AWQ" (2023).
- Xiao et al., "SmoothQuant" (2022).
- Dettmers et al., "QLoRA" (2023) and "SpQR" (2023).
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (MQA, 2019).
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023).

If you internalize this document, LLM inference stops being a black box and becomes a system you can reason about quantitatively.
