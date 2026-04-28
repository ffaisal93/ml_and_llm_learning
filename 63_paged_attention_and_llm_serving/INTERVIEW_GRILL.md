# Paged Attention & LLM Serving — Interview Grill

> 50 questions on KV cache mechanics, PagedAttention, continuous batching, prefix caching, serving systems. Drill until you can answer 35+ cold.

---

## A. KV cache fundamentals

**1. Why does autoregressive decoding need a KV cache?**
Each new token attends to all previous tokens. Without caching, you re-compute K and V for the entire prefix every step → quadratic in sequence length.

**2. KV cache memory per token?**
$2 \cdot L \cdot H \cdot d_{\mathrm{head}} \cdot \mathrm{bytes}$. (2 for K and V; $L$ layers; $H$ heads.)

**3. Llama 2 70B KV cache per token?**
With GQA-8 (8 KV heads, 128 head dim, 80 layers, BF16): $2 \times 8 \times 128 \times 80 \times 2 \approx 327$ KB per token. For 8K context: ~2.6 GB per request. Without GQA (full MHA, 64 heads), it would be ~2.6 MB per token — GQA-8 saves $8\times$.

**4. How does GQA reduce KV cache?**
Fewer KV heads (shared across query head groups). KV cache shrinks by the group factor. GQA-8 with 64 query heads → 8× smaller cache.

**5. Why is MQA aggressive?**
One KV head shared by all queries. Smallest possible cache; some quality loss.

**6. MLA reduction strategy?**
Compress KV through low-rank latent projection. Stores compressed latent + reconstructs at attention time. Even smaller than MQA.

---

## B. The fragmentation problem

**7. What's the fragmentation problem in naive KV allocation?**
Pre-allocate max-length contiguous block per request. Most requests don't reach max length → wasted memory.

**8. External fragmentation?**
Free blocks scattered; can't fit a new max-length request even though total free ≥ needed.

**9. Internal fragmentation?**
Block allocated to a request but most of it unused (request finished early or never reached max length).

**10. Why is contiguous allocation hard for LLM serving?**
Variable lengths; arbitrary completion times; memory shape demand changes per step.

**11. How much memory is wasted with naive allocation?**
PagedAttention paper: 60–80% of KV memory wasted in production setups.

---

## C. PagedAttention

**12. What's the core PagedAttention idea?**
Apply OS-style paging to KV cache. Logical KV addresses → physical KV blocks via a per-request block table.

**13. What's a block?**
Fixed-size chunk of KV cache (typically 16 tokens). The unit of allocation.

**14. What's a block table?**
Per-request mapping: logical token positions → physical block indices. Like an OS page table.

**15. How does PagedAttention reduce fragmentation?**
Allocates per-block instead of per-request. Frees blocks back to a pool when finished. ~96% utilization vs ~30% naive.

**16. Memory access pattern in PagedAttention attention kernel?**
Block-table indirection: gather KV from non-contiguous blocks. Custom kernel handles the indirection efficiently.

**17. Block size trade-off?**
Larger blocks: less indirection overhead. Smaller blocks: less internal fragmentation. 16 is a common sweet spot.

---

## D. Prefix caching / sharing

**18. What's prefix caching?**
Multiple requests sharing the same prompt prefix can share KV blocks. Only the divergent suffix needs separate blocks.

**19. How is sharing implemented in PagedAttention?**
Block table entries can point to shared physical blocks. Reference counting determines when a block can be freed.

**20. When does prefix caching matter most?**
Repeated long system prompts (chat assistants, agents). Tool-use templates. Few-shot examples reused across queries.

**21. What's "RadixAttention" (SGLang)?**
Generalizes prefix caching to arbitrary subsequences via a radix tree of KV blocks. Captures shared patterns beyond just prefixes.

**22. What's a copy-on-write KV block?**
Shared block becomes per-request-private when one request would write into it. Standard OS-style technique.

---

## E. Continuous batching

**23. What's static batching?**
Form a batch; run all to completion together. Faster requests wait for slowest.

**24. What's continuous batching (a.k.a. inflight batching)?**
At each step, swap finished requests out and admit new ones. No request waits for others.

**25. Why does continuous batching help throughput?**
Eliminates idle GPU time waiting for the slowest request. Higher GPU utilization.

**26. What does continuous batching require from the kernel?**
Variable per-step batch composition; per-request length tracking; flexible scheduling.

**27. Iteration-level vs request-level scheduling?**
Iteration-level: schedule decisions every forward step. Request-level: at request boundaries. Continuous batching is iteration-level.

---

## F. Prefill vs decode

**28. Prefill phase — what is it?**
Process the entire input prompt in parallel; populate KV cache. Compute-bound.

**29. Decode phase — what is it?**
Generate one token at a time. Memory-bound (each step reads entire KV cache).

**30. Why are they characterized differently?**
Prefill has many tokens to process in parallel → high arithmetic intensity. Decode has one token per step → arithmetic intensity tiny → bandwidth-limited.

**31. Why do servers separate prefill and decode pools?**
Different bottlenecks → different optimal hardware/configurations. Disaggregated architectures (DistServe) split them.

**32. What's chunked prefill?**
Break a long prefill into smaller chunks; interleave with decode steps. Prevents long-prefill requests from blocking decode for short ones.

---

## G. Speculative decoding

**33. What's speculative decoding?**
Small "draft" model generates $K$ tokens; big model verifies them in parallel; accept run-length until first rejection.

**34. Why is verification fast?**
Big model processes $K$ candidates in a single batched forward pass — much cheaper than $K$ sequential decodes.

**35. Acceptance ratio — what determines it?**
How well draft model approximates big model. Smaller draft + similar architecture often gets 60-80% acceptance.

**36. Self-speculative variants?**
Same model used for both draft and verify, with skip-layer (Medusa) or extra heads. No need for separate draft model.

**37. EAGLE?**
Eagle: draft model trained to predict from big model's hidden states. Higher acceptance than independent drafts.

---

## H. Quantization for inference

**38. Why quantize at inference?**
Smaller weights → less memory bandwidth → faster decode. Plus more cache fits in GPU.

**39. INT8 / INT4 weights?**
INT8: 2× smaller than FP16. INT4: 4× smaller. Quality loss with proper calibration is small.

**40. Common quantization schemes?**
GPTQ (post-hoc weight quantization minimizing reconstruction error), AWQ (activation-aware weights), SmoothQuant (rotates outliers from activations to weights).

**41. KV cache quantization?**
Quantize K and V to INT8 or even INT4. Big cache savings; some quality loss.

**42. FP8 inference?**
Hopper/Blackwell GPUs natively support FP8. Faster matmul; 2× memory savings vs FP16.

---

## I. Production serving

**43. vLLM — what is it?**
Open-source LLM serving system. Implements PagedAttention, continuous batching, prefix caching, multiple quantization. Standard production choice.

**44. SGLang — what's the differentiation?**
RadixAttention for general subsequence sharing. Strong for agentic / tool-use workloads with many shared subtrees.

**45. TensorRT-LLM?**
NVIDIA's optimized inference engine. Highest throughput on NVIDIA hardware; less flexible than vLLM.

**46. Common metrics for serving?**
TTFT (time to first token), ITL (inter-token latency), throughput (tokens/sec), goodput (effective throughput meeting SLO).

**47. What's a goodput-vs-throughput trade-off?**
Higher batch size: better throughput, worse per-request latency. Pick based on SLO.

**48. Why do tail latencies (p99) matter?**
User experience: most users want predictable response times. Long tails fail SLO even at good average.

**49. What's "request preemption"?**
Pause a request mid-generation to admit a higher-priority one. Trade-off: better priority handling, more bookkeeping.

**50. Disaggregated serving?**
Separate prefill machines from decode machines. Each optimized for its workload. Lets you scale them independently.

---

## Quick fire

**51.** *KV cache reason?* Avoid recomputing K, V for prefix.
**52.** *PagedAttention block size typical?* 16.
**53.** *Continuous batching admits new requests?* Every step.
**54.** *Prefill bottleneck?* Compute.
**55.** *Decode bottleneck?* Memory bandwidth.
**56.** *Speculative decoding acceptance typical?* 60-80%.
**57.** *vLLM main innovation?* PagedAttention + continuous batching.
**58.** *KV per token Llama 70B?* ~0.5 MB BF16.
**59.** *GQA-8 cache savings?* 8×.
**60.** *RadixAttention generalization?* Arbitrary shared subtrees, not just prefix.

---

## Self-grading

If you can't answer 1-15, you don't know KV cache basics. If you can't answer 16-35, you'll struggle on PagedAttention / serving system questions. If you can't answer 36-50, infra interview questions on LLM serving will go past you.

Aim for 40+/60 cold.
