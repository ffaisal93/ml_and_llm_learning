# LLM Inference — Interview Grill

> 50 brutal questions on LLM inference optimization. Drill until you can answer 40+ cold.

---

## A. Foundational: prefill vs decode

**1. What are the two phases of LLM inference and how do they differ?**
Prefill: process the entire prompt in one parallel forward pass. Compute-bound on modern GPUs (high arithmetic intensity, dense matmuls). Decode: generate one token at a time, attending to the growing KV cache. Memory-bandwidth-bound (read full model weights for one token of work). Almost every inference optimization addresses one of these regimes specifically.

**2. Why is decode memory-bound rather than compute-bound?**
Each decode step reads all model weights from HBM but performs only enough math for one token's worth of forward pass. Arithmetic intensity is ~1–2 ops/byte; H100's balance point is ~330 ops/byte. Tensor cores sit idle waiting for memory. The fix is batching: more concurrent decodes amortize the weight read.

**3. What's the theoretical lower bound on decode latency?**
$(\text{model-size-bytes} / \text{HBM-bandwidth}) + (\text{KV-cache-per-token} / \text{HBM-bandwidth}) + \text{overhead}$. For 70B fp16 = 140 GB (doesn't fit on a single 80 GB H100, so assume TP$\geq 2$): each GPU reads ~70 GB at ~3 TB/s ≈ 23 ms per token. No software optimization can beat this without reducing bytes per step (quantization, speculation).

**4. Why is prefill compute-bound while decode is memory-bound?**
Prefill batches $P$ tokens into a single forward pass, so weights are reused across $P$ queries → arithmetic intensity scales with $P$. Decode has one token at a time → intensity is $\sim 1$. The same hardware behaves entirely differently in the two regimes.

---

## B. KV cache

**5. Write the KV cache memory formula.**
$\text{KV-size} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot \text{seq-len} \cdot \text{bytes} \cdot \text{batch-size}$. Factor of 2 for $K$ and $V$. $d_{\text{model}} = n_{\text{heads}} \cdot d_{\text{head}}$. With GQA, replace $d_{\text{model}}$ with $n_{\text{kv-heads}} \cdot d_{\text{head}}$.

**6. Compute the KV cache size for LLaMA-2 70B at 8K context, batch 32, fp16.**
$80 \text{ layers} \times 8192 \text{ } d_{\text{model}} \times 8192 \text{ seq} \times 2 \text{ bytes} \times 2 \text{ } (K, V) \times 32 \text{ batch} \approx 687$ GB without GQA. LLaMA-2 70B uses GQA with 8 KV heads (vs 64 attention heads), so divide by 8: $\sim 86$ GB. Still substantial.

**7. What problem does KV cache solve?**
Without it, every decode step recomputes attention keys and values for the entire prefix → $O(n^2)$ total compute over $n$ decode steps. With it, you compute $K/V$ incrementally for the new token only → $O(n)$ total. KV cache makes decode tractable.

**8. What problem does KV cache create?**
Memory. KV cache scales linearly with sequence length and batch size and often dominates GPU memory in serving — limiting how many concurrent users you can fit. Long contexts make this worse.

**9. What's MQA?**
Multi-Query Attention (Shazeer 2019). All attention heads share a single $K$ matrix and a single $V$ matrix. KV cache shrinks by factor $n_{\text{heads}}$. Quality loss is small for many tasks but non-trivial for some.

**10. What's GQA?**
Grouped-Query Attention (Ainslie et al. 2023). Compromise between MHA and MQA: groups of attention heads share $K/V$. LLaMA-2 70B uses 8 KV heads with 64 attention heads (group size 8). KV cache shrinks 8x with minimal quality loss.

**11. What's PagedAttention?**
Allocate KV cache in fixed-size blocks (e.g. 16 tokens per block) instead of contiguous per-sequence buffers (Kwon et al. 2023, vLLM). A block table per request maps logical positions to physical blocks. Eliminates internal fragmentation (no need to reserve max-length up front) and external fragmentation. Enables block-level sharing for prefix caching.

**12. Why does PagedAttention enable 2–4x more concurrent requests?**
Naive allocation reserves max-length per request → most memory is wasted on slack. PagedAttention only allocates blocks as the sequence grows. With variable-length workloads, this typically frees 50–75% of the KV memory previously wasted on padding.

**13. Trade-offs of PagedAttention?**
Slightly more complex attention kernel (must follow block table during attention). Block size hyperparameter (16 typical: too small = overhead, too large = fragmentation returns). Worth it.

---

## C. Continuous batching and serving

**14. What's continuous batching?**
Iteration-level scheduling: at every decode step, evict completed requests and admit new ones. Don't wait for the longest request in the batch to finish before processing new requests. Origin: Yu et al. 2022 (Orca paper). Implemented in vLLM, TensorRT-LLM, TGI, SGLang.

**15. Why does continuous batching matter?**
Without it, GPU utilization for serving is often <30% because short requests finish early but the batch is held until the longest one completes. With continuous batching, GPU stays busy: completed slots are immediately filled. Throughput improvements of 5–20x are common.

**16. What's chunked prefill?**
Split a long prefill into chunks (e.g. 256 tokens at a time) so prefill can be interleaved with decode rather than blocking it for many milliseconds. Improves TPOT for ongoing requests when a new long-prompt request arrives.

**17. Why is mixing prefill and decode hard?**
They have different computational characteristics. Prefill is compute-bound and benefits from large effective batch sizes (many tokens at once). Decode is memory-bound and works at batch $\approx 1$ token per request. Naive interleaving wastes compute. Modern systems use specialized kernels and scheduling.

---

## D. Speculative decoding

**18. Walk me through speculative decoding.**
Use a small **draft model** to autoregressively propose $k$ tokens. Run the target model in a single forward pass over those $k$ tokens to get target probabilities for each. For each proposed token, accept with probability $\min(1, p_{\text{target}} / p_{\text{draft}})$. If accepted, keep going; if rejected, sample from the residual distribution $(p - q)_+$ and stop. Net effect: you generate (in expectation) more than 1 token per target forward pass.

**19. Is speculative decoding exact or approximate?**
**Exact.** The rejection-sampling rule is constructed to make the distribution of accepted tokens exactly the target's distribution. Output samples are statistically indistinguishable from regular target-model decoding. This is a critical correctness property.

**20. What controls the speedup?**
Acceptance rate $\alpha$ and the draft-to-target compute ratio.

$$
\text{Speedup} \approx \frac{1 + \alpha + \alpha^2 + \cdots + \alpha^k}{1 + (\text{draft-cost} / \text{target-cost}) \cdot k}
$$

High $\alpha$ (close draft to target) and small $\text{draft-cost} / \text{target-cost}$ give big speedups. Typical: 2–3x with a 7B draft for a 70B target.

**21. When does speculative decoding NOT help?**

- Throughput-limited servers running at large batch size, where decode is already compute-bound.
- Very low acceptance rates (mismatched draft model).
- Latency-tolerant batch jobs where total compute is the metric (speculation increases total compute).

**22. What's self-speculative decoding?**
Use shallow layers of the target model itself as the draft. No external model needed. Trade some quality of the draft for memory simplicity. EAGLE and Medusa are related: train extra heads to predict multiple future tokens.

**23. What's tree speculation?**
Propose a tree of candidate continuations (multiple branches from the same prefix). Verify with one target forward pass that batches all branches together. Higher effective acceptance because the target model picks the longest matching branch.

---

## E. FlashAttention

**24. What's FlashAttention?**
I/O-aware attention computation (Dao et al. 2022). Tile $Q, K, V$ so that blocks fit in SRAM. Use online softmax to compute partial softmax statistics block-by-block. Same FLOPs as standard attention, but avoids materializing the full $N \times N$ attention matrix in HBM. Result: 2–4x wall-clock speedup, especially at long sequence lengths.

**25. Why is FlashAttention faster — algorithmically or hardware-wise?**
Hardware-wise. The algorithm has the same FLOPs. The speedup comes from reducing HBM reads/writes. Modern GPUs are massively memory-bandwidth-limited for attention; reducing memory traffic is the lever.

**26. Walk me through online softmax.**
Standard softmax: compute max, subtract, exponentiate, sum, divide. Requires two passes over data (first for max, second for normalization). Online softmax: maintain running max $m$ and running sum $s$. For each new value $x$: $m' = \max(m, x); s' = s \cdot \exp(m - m') + \exp(x - m'); m = m'; s = s'$. Single pass. Numerically equivalent. Lets attention be computed block-by-block without materializing the full matrix.

**27. What's FlashAttention-2 vs FA-1?**
FA2 (2023): better parallelization across thread blocks (parallelize over the sequence dimension and not just heads), reduced non-matmul ops. ~2x faster than FA1.

**28. What's FlashAttention-3?**
FA3 (2024): uses Hopper-specific features (TMA, async warpgroup matmul, FP8 support). Reaches near-peak bf16 utilization on H100.

**29. Does FlashAttention help decode?**
Less than prefill, because decode is $O(n)$ attention not $O(n^2)$. FlashDecoding and FlashDecoding++ are decode-specialized variants that bring similar I/O-aware tiling to the decode case (parallel over the KV sequence dimension).

---

## F. Quantization

**30. What's W4A16?**
4-bit weights, 16-bit activations. Standard for LLM inference. Methods: GPTQ, AWQ. Memory savings ~4x over fp16; compute is dequantize-then-fp16 matmul, so speedup is ~2–3x not 4x.

**31. What's W8A8?**
8-bit weights, 8-bit activations. Compute uses INT8 tensor cores (often ~2x throughput vs fp16). Method: SmoothQuant. Quality more sensitive; activation outliers must be handled.

**32. What's GPTQ?**
Per-layer Hessian-based weight quantization. For each layer, quantize weights one at a time, after each step adjusting remaining weights to compensate for quantization error. Uses a small calibration set (~128 samples). Tractable for 70B+ models.

**33. What's AWQ?**
Activation-aware weight quantization. Identify the small fraction of weights that have high activation magnitude (~1%). Scale those before quantization to preserve them. Faster than GPTQ to apply; comparable or better quality.

**34. What's SmoothQuant?**
Migrates outlier activations to weights pre-quantization. Per-channel scaling factor $s_j$ such that $(X_j / s_j) \cdot (s_j \cdot W_j) = X_j W_j$. Activations become smaller (no outliers); weights become slightly larger. Both fit in INT8 cleanly.

**35. What's FP8?**
8-bit floating point. Two formats: E4M3 (4 exp bits, 3 mantissa) for forward; E5M2 (5/2) for backward / wider range. Native on H100/Blackwell. Less calibration-sensitive than INT8 because of dynamic range. Increasingly common for both training and inference.

**36. What's NF4?**
4-bit information-theoretically optimal float for normally-distributed weights. Used in QLoRA. Better dynamic-range matching than INT4 for typical weight distributions.

**37. Why does INT8 quantization sometimes hurt LLM quality more than expected?**
Activation outliers. A few channels per layer have activations 100x typical. Naive per-tensor quantization with a global scale clips them, destroying information. SmoothQuant or per-channel scaling fixes it. INT4 weight quantization (no activation quantization) is less affected.

**38. Can you quantize the KV cache?**
Yes, common. INT8 KV is straightforward; INT4 KV needs group-wise quantization (e.g., per 128 tokens). Saves 2–8x KV memory → enables longer contexts and larger batches with no compute change.

---

## G. Multi-GPU and parallelism

**39. What's tensor parallelism?**
Split each weight matrix across GPUs (e.g. attention $Q, K, V$ projections sharded along the head dimension). Each GPU computes part of the matmul; results all-reduced. Standard for inference of >13B models. Inter-GPU communication on every layer.

**40. What's pipeline parallelism?**
Different GPUs hold different layers; the forward pass flows GPU-to-GPU. Communication only at layer boundaries. Bad for inference latency because of pipeline bubbles (one request can't fill the pipeline). More common in training.

**41. What's expert parallelism?**
For Mixture-of-Experts models (Mixtral, GPT-4-class). Different experts on different GPUs; routing decides which GPU(s) handle each token. Communication pattern is all-to-all. Tricky to load-balance.

**42. For 70B inference, what's the typical multi-GPU setup?**
Tensor parallel = 4 or 8 within a node (intra-NVLink). Data parallel across nodes. PagedAttention manages KV cache. Continuous batching for throughput.

---

## H. Latency and metrics

**43. What's TTFT?**
Time To First Token: from request arrival to the first generated token streaming back. Dominated by prefill. Scales with prompt length. Critical for chat UX (target <500ms).

**44. What's TPOT (or ITL)?**
Time Per Output Token: average inter-token latency during decode. Dominated by memory bandwidth and batch utilization. Target <50ms for natural reading pace.

**45. How would you reduce TTFT?**
Prefill chunking, FlashAttention, smaller initial KV (sliding window), prompt caching (share prefix KV across users), faster networking for distributed prefill.

**46. How would you reduce TPOT?**
Continuous batching (more concurrent decodes amortize weight read), KV cache compression, quantization (less weight to read), speculative decoding, MQA/GQA, tensor parallelism (split memory bandwidth across more GPUs).

**47. Throughput vs latency — how do you trade them?**
Bigger batch → higher throughput (more amortization of weight read), worse per-request latency (more contention). Choose batch size to maximize throughput subject to latency SLA. Workload mix matters: long-context requests need smaller batches to fit KV.

---

## I. System / serving design

**48. What's prompt caching / prefix caching?**
Reuse computed KV cache for shared prompt prefixes across requests. Particularly valuable for: multi-turn chats (previous turns are shared), RAG (retrieved documents shared across users), tool-using agents (long system prompts). Implemented naturally on top of PagedAttention via block reference counting.

**49. How does an MoE model change inference?**
Each token only activates a subset of experts (typically 2 out of 8). Compute per token is reduced, but routing introduces overhead and load-balancing problems. Expert parallelism: experts spread across GPUs, all-to-all communication on routing. Memory still scales with total expert count, not active expert count. So MoE saves compute but not memory.

**50. Walk me through the cost-per-token mental model.**
$\text{cost} \approx (\text{weight-bytes} + \text{KV-bytes-for-step}) / \text{HBM-bandwidth} + \text{compute-overhead}$. For a 70B fp16 model sharded across 2× H100 (TP=2): each GPU reads ~70 GB at ~3 TB/s ≈ 23 ms per decode step → ~43 tok/s/request floor. Quantize to W4A16 (35 GB total → fits on one H100): ~12 ms → ~80 tok/s. Add speculation with 0.6 acceptance: another ~2x. This is how to reason quantitatively about latency budgets.

---

## J. 2024-2025 frontier inference

**51. What's MLA (Multi-head Latent Attention)?** (DeepSeek-V2/V3)
Project K and V into a low-rank latent (dim $d_c \ll n_h \cdot d_h$); cache only the latent. At attention time, project back up. KV cache shrinks $\sim 10\times$ vs MHA, $\sim 3$–$4\times$ vs GQA-8. Subtlety: RoPE doesn't commute with the down-projection — DeepSeek splits each head into a "RoPE part" (uncompressed) and a "no-RoPE part" (compressed).

**52. Why does MLA matter for serving cost?**
At long context (128K+), KV cache dominates memory. MLA shrinks KV enough that DeepSeek-V3 (671B params, MoE) is *cheaper to serve per token* than smaller GQA-only models. A practical disproof of "bigger model = always more expensive."

**53. What's chunked prefill?**
Long prompts have two problems: bad TTFT and they block decode for other batched requests. Split prefill into chunks of ~512–2K tokens, interleave with decode steps. Standard in vLLM 0.6+, SGLang, TensorRT-LLM. Improves TTFT and decode throughput simultaneously.

**54. What's disaggregated serving (DistServe / Mooncake)?**
Prefill is compute-bound, decode is memory-bound — using same GPUs for both is wasteful. Disaggregation: separate prefill workers (batched, large compute) from decode workers (smaller batches, memory locality). KV transferred between them. 2-4× higher goodput at fixed SLO. Mooncake = Moonshot AI's Kimi serving system.

**55. EAGLE vs Medusa — what's the difference?**
Both are speculative-decoding variants without a separate draft model. **Medusa**: extra "Medusa heads" predict next-2, next-3, etc. tokens from the target's last hidden state. Simple. **EAGLE**: trains a small "feature predictor" using the target's penultimate-layer features. Higher acceptance (60-80%) than vanilla draft models because it shares representation. EAGLE-2/3 push to 70-85%.

**56. What's StreamingLLM / attention sinks?**
Empirical observation: most attention concentrates on first ~4 tokens (the "attention sink") + recent tokens. Drop the middle of the KV cache; keep early + recent. Lossy but works for very long contexts. Used when MLA + GQA still aren't enough.

**57. KV cache quantization — what schemes?**
KIVI: per-channel for K, per-token for V (matches their distribution shapes). KVQuant: outlier-aware. FP8 KV: easiest path on H100; ~2× cache reduction at minimal quality loss. Combined with MLA/GQA, KV memory becomes manageable even at 128K+.

**58. FP4/FP6 on Blackwell — what changes?**
NVIDIA Blackwell natively supports FP4 matmul. Inference at $\sim 4\times$ FP16 throughput. Production sweet spot: FP4 weights + FP8 activations. Aggressive: FP4 + FP4 (needs careful calibration). Format variants like NVFP4/MXFP4 use micro-block scales for better numerics. Will dominate 2025+ serving.

**59. Continuous-batching scheduler — what does it actually decide?**
At each step: which requests to admit (memory permitting), which to pause / preempt, which to advance. Policies: FCFS (simple), priority by length (better tail latency), memory-aware admission, preemption with KV swap to CPU. The scheduler is the load-bearing piece in production serving.

**60. Modern latency targets?**
Chat: TTFT < 500 ms, TPOT < 50 ms. Voice: TTFT < 200 ms, TPOT < 30 ms. Achieving these on 70B+ models requires the full stack: GQA/MLA + chunked prefill + continuous batching + speculative decoding + FP8/FP4.

**61. What's Ring Attention?**
For sequences too long to fit on one GPU. Distribute the sequence across GPUs; pass K/V slices in a ring while each GPU computes its query's attention against rotating K/V. Used to train + serve million-token models (Gemini 1.5 Pro and later). Cost: extra inter-GPU communication. Pairs with FlashAttention.

**62. Provider-level prompt caching (Anthropic, OpenAI)?**
Beyond batch-level prefix sharing: providers cache long prompt prefixes across requests / users for hours. Pricing-tier discounts for cache hits (e.g., 90% off cached input tokens). Critical for agentic workloads with repeating system prompts + tool definitions across thousands of requests.

**63. The 2025 inference stack in one sentence?**
GQA or MLA + chunked prefill + continuous batching with memory-aware scheduler + speculative decoding (EAGLE-style) + FP8 weights/activations (FP4 on Blackwell) + KV quantization for long context + disaggregated prefill/decode at scale.

---

## Quick fire (under 10 seconds)

**64.** *KV cache scales how?* Linearly with seq_len, batch, layers; quadratically only via attention compute, not cache size.
**65.** *Default block size in vLLM?* 16 tokens.
**66.** *FlashAttention complexity reduction?* None — same FLOPs. Memory access is what changes.
**67.** *Speculative decoding correctness property?* Exact — same distribution as target.
**68.** *MQA vs GQA savings?* MQA: factor $n_{\text{heads}}$. GQA: factor $n_{\text{heads}} / n_{\text{groups}}$.
**69.** *Quantization bits used in modern serving?* W4A16 (GPTQ/AWQ) and W8A8 (SmoothQuant), increasingly FP8/FP4.
**70.** *Standard latency targets?* TTFT < 500ms, TPOT < 50ms.
**71.** *Tensor parallel across NVLink, what's typical?* TP=8 within node.
**72.** *Continuous batching origin paper?* Orca (Yu et al. 2022).
**73.** *PagedAttention origin?* vLLM (Kwon et al. 2023).
**74.** *MLA factor savings vs GQA-8?* ~3-4×.
**75.** *Chunked prefill — typical chunk size?* 512-2K tokens.
**76.** *EAGLE acceptance rate?* 60-85%.
**77.** *Mooncake = ?* Kimi's disaggregated serving system.
**78.** *Attention sink token count?* First ~4 tokens.
**79.** *Blackwell native low-precision format?* FP4 (with FP8 activations sweet spot).
**80.** *Ring Attention enables?* Million-token context.

---

## Self-grading

If you can't answer 1–10, you don't understand LLM inference at all. If you can't answer 11–25, you can't pass an inference-focused MLE round. If you can't answer 26–50, you'll fall short in frontier-lab serving-systems interviews. If you can't answer 51-63 (the 2024-2025 frontier — MLA, EAGLE, disaggregated serving, FP4, etc.), you'll be behind on what frontier serving teams actually deploy today.

Aim for 60+/80 cold before any LLM serving interview.
