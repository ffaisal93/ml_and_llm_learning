# AI Infrastructure Engineer — Production Playbook

> A production-flavored counterpart to the research-scientist chapters in this repo. Built around the 8 areas a production AI infrastructure engineer is expected to know cold (per Md Ismail Sojal's checklist + industry hiring patterns):
>
> 1. GPU / VRAM fundamentals, quantization & batching
> 2. vLLM / TensorRT-LLM / inference optimization
> 3. KV caching, speculative decoding, token throughput
> 4. Distributed training basics (DDP / FSDP / DeepSpeed)
> 5. Model serving & autoscaling
> 6. Vector DB retrieval pipelines
> 7. Prompt caching & cost optimization
> 8. Observability for LLM apps
>
> The research-scientist focus elsewhere in this repo covers the *what* and *why*. This chapter is the *how to actually run it in production* layer — what AI infra engineers at OpenAI, Anthropic, Cohere, Together, Fireworks, Anyscale, Modal, Baseten, Replicate, etc. ship and operate.
>
> Pair with [`61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md`](../61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md) (algorithmic depth), [`06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`](../06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md) (inference internals), `63_paged_attention_and_llm_serving/` (paged attention deep dive).

---

## Table of contents

1. The job — what an AI Infra Engineer actually does
2. GPU / VRAM fundamentals (hardware-level mental model)
3. Quantization in production
4. Batching strategies (static / dynamic / continuous / chunked)
5. Inference engines: vLLM, TensorRT-LLM, TGI, SGLang, LMDeploy
6. KV caching in production (PagedAttention, prefix caching, eviction)
7. Speculative decoding in production
8. Token-throughput metrics and SLOs (TTFT, TPOT, TPS, ITL)
9. Distributed training infrastructure (DDP, FSDP, DeepSpeed, Megatron, slurm/k8s/Ray)
10. Model serving and autoscaling (Triton, KServe, BentoML, Ray Serve, Modal, Baseten)
11. Vector DB retrieval pipelines (Pinecone, Weaviate, Qdrant, Milvus, pgvector)
12. Prompt caching and cost optimization
13. Observability for LLM apps (LangSmith, Langfuse, Helicone, Arize)
14. Capacity planning and cost modeling
15. Reliability — checkpointing, fault tolerance, blue-green
16. Security — secrets, network, prompt-injection at the infra layer
17. The full production stack — how it fits together
18. Senior signals
19. References
20. Interview grill — 100 questions

---

## 1. The job

An AI Infrastructure Engineer makes LLMs **fast, cheap, reliable, and scalable** in production. Not training new models from scratch (that's research). Not building product features (that's product). The role is the operational layer between research artifacts (model weights) and product traffic.

Day-to-day responsibilities:
- Take a checkpoint, deploy it on the right hardware with the right inference engine.
- Hit latency SLOs (p50, p95, p99) at projected QPS.
- Hit cost targets (\$/1M tokens, \$/request, \$/active user).
- Ensure reliability (uptime, failover, rolling deploys).
- Set up observability: traces, metrics, logs, eval.
- Capacity-plan for growth.
- Optimize for KV cache, batching, quantization, speculative decoding.
- Operate multi-tenant or multi-model serving fleets.
- Wire up vector DBs and retrieval for RAG products.
- Manage cost (prompt caching, model routing, batch APIs).

The interview probes whether you've **shipped this stack** (and which parts you've personally touched), not whether you've published papers about it.

> **Saying it out loud.** My job is to take a model somebody else trained and make it fast, cheap, and reliable enough to put in front of real traffic. That means picking the hardware and the inference engine, hitting a p99 latency target at the projected request rate, and hitting a cost-per-million-tokens number the business can actually charge against. I'm not doing research and I'm not building product features — I'm the layer in between, where a checkpoint becomes a service. The thing that separates people who've done this from people who've read about it is that we start from the constraint, not the design: tell me the latency SLO and the cost ceiling and I'll tell you the quantization, the batching, and the GPU count that fit inside them.

---

## 2. GPU / VRAM Fundamentals

The hardware mental model. You can't reason about inference without it.

### 2.1 GPU architecture (NVIDIA-centric, since 95% of LLM serving runs on NVIDIA)

- **Streaming Multiprocessors (SMs).** Independent compute units. H100 has 132 SMs.
- **CUDA cores.** Per-SM execution units for FP32/FP16. Not the bottleneck for LLMs.
- **Tensor Cores.** Specialized matrix-multiply units. Operate on 4x4 or 16x16 fragments. The actual workhorse for LLM inference.
- **Memory hierarchy:**
  - **Registers** (~256 KB per SM, fastest, per-thread)
  - **Shared memory / L1** (~256 KB per SM, ~19 TB/s bandwidth — what FlashAttention exploits)
  - **L2 cache** (~50 MB across chip, ~3 TB/s)
  - **HBM (High Bandwidth Memory) / VRAM** (40-141 GB on H100/H200, ~3.35 TB/s on H100 SXM)
  - **CPU DRAM** (slow, ~30 GB/s via PCIe — avoid touching during decode)

### 2.2 The two regimes

LLM inference has two compute regimes:

- **Compute-bound (prefill).** Long input → many tokens to score in parallel → tensor cores saturate. Throughput limited by FLOPS (~989 TFLOPS BF16 on H100).
- **Memory-bandwidth-bound (decode).** One token at a time → must load all weights from HBM → throughput limited by HBM bandwidth (~3.35 TB/s on H100 SXM).

**Hook.** "Prefill compute-bound; decode memory-bandwidth-bound. Most production cost is decode."

### 2.3 Frontier GPU specs (2025-2026)

*Prices and hourly rates below are point-in-time figures as of late 2025 / early 2026, not current quotes. GPU list prices, cloud on-demand rates, and spot pricing move fast — treat every number here as an order-of-magnitude anchor for arithmetic, and say "as of when I last looked" if you quote one in an interview.*

| GPU | VRAM | HBM BW | FP16 TFLOPS | TDP | List \$ |
|---|---|---|---|---|---|
| A100 80GB SXM | 80 GB | 2 TB/s | 312 | 400W | ~\$15K |
| H100 80GB SXM | 80 GB | 3.35 TB/s | 989 | 700W | ~\$30K |
| H100 NVL 94GB | 94 GB | 3.9 TB/s | 1979 | 350W | ~\$40K |
| H200 141GB | 141 GB | 4.8 TB/s | 989 | 700W | ~\$40K |
| B100/B200 | 192 GB | 8 TB/s | ~2500 (FP4 is 9k) | 1000-1200W | ~\$50K+ |
| GB200 NVL72 (rack) | 72 × 192 GB | massive | massive | massive | ~\$3M+ |

Cloud renting (rough order of magnitude):
- A100: \$1.5-3 / hour
- H100: \$3-6 / hour
- H200: \$4-8 / hour
- B200: \$7-12 / hour

**Why memory matters most.** A 70B model in BF16 is 140 GB. In FP8, 70 GB. In INT4, 35 GB. The choice between A100 (80GB), H100 (80GB), H200 (141GB), B200 (192GB) is mostly about **what fits**.

### 2.4 NVLink / NVSwitch / InfiniBand

**In plain language.** GPUs in a big training or serving job have to constantly hand each other partial results, so how fast they can talk determines how you're allowed to split the model. NVLink is the very fast wiring between GPUs sitting in the same physical server; InfiniBand is the merely-fast wiring between servers in a rack or cluster — roughly an order of magnitude slower. That gap is the whole design rule: the way of splitting a model that chats the most (tensor parallelism) has to stay inside one server on NVLink, and the ways that chat less (data and pipeline parallelism) are what you stretch across servers.

- **NVLink.** GPU-to-GPU within a node. H100: 900 GB/s bidirectional (NVLink 4). Used for tensor parallelism (which is NVLink-bound).
- **NVSwitch.** Switch fabric making all GPUs in a server (8 typically) NVLink-connected.
- **InfiniBand (NDR).** Cross-node networking. ~400 Gbps. Slower than NVLink, used for data parallelism / pipeline parallelism across nodes.

**Implication.** Tensor parallel within a node (TP=8 on a DGX H100). Data/Pipeline parallel across nodes.

### 2.5 What an interview-ready answer looks like

> "For 70B BF16, that's 140 GB of weights + KV cache. Doesn't fit on one 80GB H100; needs TP=2 or H200 (141GB). For decode at long context, KV cache might add 10-30 GB on top, pushing toward TP=2 or 4 even on H200. Decode is memory-bandwidth-bound at ~3.35 TB/s on H100, so per-GPU per-second I can scan ~3.35 TB / 140 GB = ~24 token/s before TP. With TP=2, weights split, BW per token halves → ~48 token/s. PagedAttention + continuous batching gets total throughput up by sharing prefill and overlapping. With INT4 quantization, weights drop to 35 GB, single H100 fits, and per-GPU TPS gets to ~95."

That's a senior answer.

> **Saying it out loud.** The one thing to internalize about a GPU is that it has way more arithmetic than memory bandwidth, so most of the time you're not waiting on math, you're waiting on bytes moving out of HBM. That's why prefill and decode behave like different machines: prefill scores a whole prompt at once and saturates the tensor cores, while decode produces one token at a time and has to re-read every weight from memory to do it. So capacity planning is really a memory question — a 70B model in BF16 is 140 GB of weights, which doesn't fit on an 80 GB H100 at all, and the KV cache piles on top of that. The number to have ready: H100 SXM is about 3.35 TB/s of HBM bandwidth, so scanning 140 GB of weights caps you at roughly 24 tokens per second per GPU before any parallelism or quantization.

---

## 3. Quantization in Production

(Short recap — full coverage in [`06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`](../06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md).)

### 3.1 The standard production menu (2025-2026)

| Format | Bytes/param | Quality loss | Hardware | Tools |
|---|---|---|---|---|
| BF16 | 2 | None (baseline) | A100+ | Default |
| FP8 (E4M3 / E5M2) | 1 | <0.5% on most tasks | H100+ | TensorRT-LLM, vLLM |
| INT8 (W8A8) | 1 (W) | ~0.5-1% | All | SmoothQuant, TensorRT |
| INT8 weight-only (W8A16) | 1 (W) | ~0.3% | All | LLM.int8() |
| INT4 weight-only (W4A16) | 0.5 (W) | ~1-3% | All | GPTQ, AWQ |
| FP4 | 0.5 | ~1-2% | B200 only | TensorRT, vLLM |
| INT4 KV cache | 0.5 | <1% | All | vLLM `kv_cache_dtype="fp8"` |

**The default 2025 production setup:** weights INT8 or FP8, KV cache FP8, activations BF16, TensorRT-LLM or vLLM as engine.

### 3.2 Calibration and post-training quantization

- **Calibration set.** ~128 samples from production traffic distribution. Capture per-layer activation min/max.
- **Per-channel** (weight-by-output-channel) is standard; smoother than per-tensor.
- **Per-group** (groups of 128 weights) further reduces outlier impact for INT4.
- **SmoothQuant.** Migrate quantization difficulty from activations to weights via a per-channel scaling. Critical for INT8 W8A8.
- **GPTQ, AWQ.** PTQ algorithms for INT4 weight-only that minimize layer-wise output error.

### 3.3 Quality validation

Always validate on:
- Held-out perplexity (relative to BF16 baseline).
- Task benchmarks (MMLU, HumanEval, MT-Bench).
- Production-traffic eval (sample 1k requests, run BF16 and quantized, judge for divergence).

A common production policy: **don't ship if perplexity gap > 1% or task benchmark gap > 2%.**

> **Saying it out loud.** Quantization is storing the weights in fewer bits so you read fewer bytes per token — and since decode is memory-bandwidth-bound, fewer bytes is directly more speed, not just less memory. FP8 is the safe default on Hopper and newer: about half the bytes of BF16, hardware support so it's genuinely faster, and quality loss you usually can't measure. INT4 with AWQ or GPTQ halves it again and is what you reach for when you're trying to fit a big model on one GPU. The tradeoff to name is that quality loss is not uniform — it concentrates in long-context, multi-turn, and code generation, and standard perplexity checks will happily miss it. So the rule is that you never ship a quantized model on a perplexity number alone; you run the task evals you actually care about, side by side.

---

## 4. Batching Strategies

The single biggest throughput lever in production.

### 4.1 Static batching

Wait for B requests, run them as one batch with right-padding. Simple, but:
- Fast requests wait for slow ones (head-of-line blocking).
- Wasted compute on padding.
- No good for variable-length workloads.

Used only in low-throughput dev or batch-API contexts.

### 4.2 Dynamic batching

Group requests arriving within a time window (e.g., 50ms) into a batch. Better than static but still has padding waste and sync issues.

### 4.3 Continuous batching (a.k.a. in-flight batching, iteration-level scheduling)

The breakthrough. From Orca (2022) and used by vLLM, TGI, TensorRT-LLM today.

**Idea.** At each decode iteration, the scheduler can:
- Add new requests (prefill) to the running batch.
- Remove finished requests immediately (don't wait for the slowest).
- Mix prefill and decode in the same iteration (with chunked prefill).

**Result.** GPU utilization stays high; tail latency from slow requests doesn't block fast ones; throughput often 5-10× static batching.

### 4.4 Chunked prefill

Long prompt prefill can take seconds, blocking decode. **Chunked prefill** splits a long prefill into chunks (e.g., 1k tokens each); each iteration does part of the prefill plus decode for other requests.

vLLM's `--enable-chunked-prefill` flag is now default. Critical for low-TTFT serving with mixed long+short prompts.

### 4.5 Disaggregated serving (Mooncake, DistServe)

Separate prefill and decode onto different GPU pools. Prefill is compute-bound (use B200/H100); decode is memory-bound (could use cheaper but high-BW GPU). Send KV cache between pools over high-speed network.

State-of-the-art for very-large-scale deployments. Used by Anthropic, OpenAI, Together.

### 4.6 Hook ladder

"Static < dynamic < continuous + chunked prefill < disaggregated."

> **Saying it out loud.** Batching is how you stop wasting the GPU, because loading the weights costs the same whether you're serving one request or sixty-four. The naive version is static batching, where you wait for a group, run them together, and everyone leaves when the slowest one finishes — which means short requests sit around burning GPU time doing nothing. Continuous batching fixes that by scheduling at the level of individual decode steps: the moment a sequence finishes, a waiting request drops into its slot. That's typically a five-to-ten-times throughput win and it's the single biggest lever in a serving stack. The tradeoff to name is that bigger batches raise time-to-first-token because of queuing, which is exactly what chunked prefill and disaggregated prefill/decode exist to unwind.

---

## 5. Inference Engines

Pick one. Stick with it. Master its config.

### 5.1 vLLM

- **What.** Open-source, originally Berkeley/UCB, now wide community.
- **Strengths.** PagedAttention (best KV cache management), continuous batching, chunked prefill, prefix caching, broad model support, easy quantization (GPTQ, AWQ, FP8), Python-first API.
- **Weaknesses.** No NVIDIA-specific kernel-level squeeze; H100 FP8 throughput slightly behind TRT-LLM.
- **Best for.** OSS fleet, fast iteration, multi-model.

### 5.2 TensorRT-LLM

- **What.** NVIDIA's optimized inference engine.
- **Strengths.** Best raw throughput on NVIDIA HW. Custom kernels per model. FP8 / FP4 first-class. Tight integration with Triton Inference Server.
- **Weaknesses.** Build per model (compile-time graph optimization). Less flexible. Tied to NVIDIA stack.
- **Best for.** Stable production at huge scale where 10-30% throughput matters.

### 5.3 TGI (Text Generation Inference, HuggingFace)

- **What.** HF's serving engine.
- **Strengths.** Tight HF Hub integration, Rust-based front-end (low overhead).
- **Weaknesses.** Has lagged vLLM/TRT-LLM on bleeding-edge features.
- **Best for.** HuggingFace-shop ecosystems.

### 5.4 SGLang

- **What.** UC Berkeley project, late 2024.
- **Strengths.** Excellent for **structured outputs** (JSON, regex, function calling) and **complex tool-call workflows**. RadixAttention prefix caching.
- **Best for.** Agentic or RAG workloads where prefix caching dominates.

### 5.5 LMDeploy

- **What.** Shanghai AI Lab.
- **Strengths.** Strong on Chinese deployments, good INT4 support.
- **Best for.** Chinese-market deployments.

### 5.6 DeepSpeed-FastGen / MII

- Microsoft. Less prominent now; vLLM has overtaken.

### 5.7 Decision matrix

| You want | Pick |
|---|---|
| Easiest open-source | vLLM |
| Max throughput on NVIDIA | TensorRT-LLM |
| HF ecosystem | TGI |
| Heavy structured outputs / RAG | SGLang |
| Multi-region multi-cloud | vLLM (more portable) |
| Custom kernels for niche model | TRT-LLM |

**Hook.** "vLLM is the OSS default; TRT-LLM if you need the last 10-20% throughput."

> **Saying it out loud.** These are all doing the same three things — paged KV, continuous batching, fused kernels — and the choice is mostly about who you are, not what's fastest. vLLM is the default: open, huge community, PagedAttention is theirs, and it runs on NVIDIA and AMD. TensorRT-LLM is faster on NVIDIA if you're willing to pay for it in build complexity and lock-in, because you compile a model-specific engine. SGLang wins on structured and agentic workloads because of RadixAttention prefix sharing, and TGI is the pick when you're already inside the HuggingFace ecosystem. The senior move is naming the version and the flags — "vLLM with chunked prefill on and fp8 KV cache" — because that's the part you can only know from having actually run it.

---

## 6. KV Caching in Production

(Algorithm covered in `06_llm_inference/`. Production focus here.)

### 6.1 PagedAttention (vLLM)

- KV cache split into fixed-size **blocks** (typically 16 tokens).
- A **block table** maps logical sequence positions → physical block locations.
- Like virtual memory: avoid fragmentation, share blocks across requests.

### 6.2 Prefix caching

- Hash-based cache of (prefix tokens → KV blocks).
- New request: find longest prefix match, reuse blocks, compute only suffix.
- vLLM `--enable-prefix-caching`. SGLang RadixAttention.
- **Massive** wins on chat workloads (system prompt + history reused across turns).

### 6.3 KV cache eviction policies

When VRAM is full and you can't admit a new request:
- **Drop oldest unfinished sequence** (LRU on requests).
- **Recompute from prompt** (preserve correctness, expensive).
- **Swap to CPU memory** (vLLM `--swap-space` flag) — pull back when needed.

### 6.4 KV quantization

`--kv-cache-dtype fp8` in vLLM. Halves KV memory at <0.5% quality loss. Almost always a win.

### 6.5 Cross-request KV sharing

For tenant deployments where many users share a system prompt: prefix caching captures this. For LoRA-multi-tenant: shared base KV + LoRA-specific deltas (S-LoRA).

### 6.6 The KV math you should be able to do

```
KV cache size per token = 2 (K, V) * num_layers * num_kv_heads * head_dim * bytes_per_elem
```

**Llama 3 70B (BF16):** 2 × 80 × 8 × 128 × 2 = **327 KB / token**. At 32K context: 10.5 GB / sequence. At batch=8: 84 GB just for KV cache. **This is why GQA matters.**

> **Saying it out loud.** The KV cache is the model's memory of the tokens so far, and in production it's usually what runs you out of VRAM rather than the weights, because it grows with context length times batch size and it's per user. The old way to allocate it was one contiguous slab per request, sized to the maximum possible length — which wasted most of it, since almost no request hits the max. PagedAttention borrows the operating system's trick: chop the cache into fixed 16-token blocks and keep a page table, so you allocate as you go and blocks from different requests can share pages. That's how vLLM gets a two-to-four-times throughput improvement without touching the model. The failure mode to name is fragmentation-driven preemption — under memory pressure the scheduler evicts a running sequence and has to recompute its prefill, so your p99 latency spikes while average throughput still looks fine.

---

## 7. Speculative Decoding in Production

(Algorithm covered in `06_llm_inference/`. Production focus here.)

### 7.1 What you actually deploy

- **Vanilla speculative.** Tiny same-family draft (e.g., 1B → 70B). Memory cost: ~1.5GB extra per replica.
- **Self-speculative (Medusa, EAGLE).** No separate draft; extra decoding heads. No extra weights. Used by together.ai, fireworks.
- **N-gram speculative (Chain Speculation).** Draft from prompt itself for repetitive content (code, structured outputs). Free.

### 7.2 Production gotchas

- **Acceptance rate matters more than spec count.** Tune draft / speculation length to maximize accepted-tokens-per-second.
- **Batching interaction.** Speculative decoding throughput drops as batch size grows (the verifier's parallel benefit shrinks). Sweet spot: low-to-medium batch sizes.
- **KV cache duplication.** Both target and draft need their own cache for the same sequence. Memory cost.
- **Quality.** 100% lossless if implemented correctly (target verifies). Watch for subtle samplers (temperature, top-p) that can break verification.

### 7.3 vLLM / TRT-LLM speculative settings

- vLLM: `--speculative-model EAGLE-Llama-3-8B --num-speculative-tokens 5`
- TRT-LLM: built-in for Medusa, EAGLE.

> **Saying it out loud.** Speculative decoding exploits the fact that decode is memory-bound: since you're already paying to read all the weights for one token, you may as well check several at once. A cheap draft model guesses the next few tokens, the big model verifies them all in a single forward pass, and you keep the longest prefix that matches what the big model would have produced anyway. It's mathematically lossless — the output distribution is identical — and it typically buys one-and-a-half to three times faster decoding. The tradeoff people get wrong: the win comes from spare compute, so it evaporates at high batch sizes where you're already compute-saturated. At large batch it can actually make you slower, which is why production systems gate it on current load rather than leaving it on.

---

## 8. Token-throughput Metrics and SLOs

The four numbers you live and die by.

| Metric | What | Typical SLO |
|---|---|---|
| **TTFT** (Time To First Token) | Latency from request to first decoded token. Dominated by prefill. | p50 < 0.5s, p99 < 2s |
| **TPOT** / **ITL** (Time Per Output Token / Inter-Token Latency) | After first token, latency per next token. | p50 < 30 ms (chat), < 50 ms (long context) |
| **TPS** (Tokens Per Second) | Total output tokens / second. | Per request: > 30 tps (good UX). Total throughput: depends on capacity. |
| **Throughput (req/s)** | Concurrent users × completion rate. | Site-dependent. |

### 8.1 Tradeoffs

- **Smaller batch** → lower TTFT, higher TPOT (less amortized weight loading).
- **Larger batch** → higher TTFT (queuing), lower per-request TPS (sharing GPU).
- **Disaggregated prefill/decode** → both improve.
- **Chunked prefill** → TTFT consistent at long context.

### 8.2 SLO design

1. Pick TTFT target (e.g., p95 < 1s).
2. Pick TPOT target (e.g., p95 < 50ms).
3. Pick concurrency target (e.g., 100 concurrent decodes).
4. Provision GPUs to meet all three.
5. Monitor and autoscale on the most-violated metric.

### 8.3 Useful telemetry per request

- Tokens in / out.
- TTFT, TPOT, total latency.
- Engine queue time vs actual compute time.
- Cache hit rate (prefix).
- Speculative acceptance rate.
- VRAM utilization.

> **Saying it out loud.** There are two latencies users feel and they trade against each other. Time-to-first-token is how long they stare at nothing, and it's dominated by prefill and by queuing. Time-per-output-token is how fast the text streams once it starts, and it's dominated by memory bandwidth and by how many other requests are sharing the GPU. Small batches give you great TTFT and mediocre throughput; large batches give you great throughput and users waiting in a queue. The numbers worth having: p99 TTFT under about two seconds and around thirty milliseconds per output token, because thirty milliseconds is roughly faster than people read. And the discipline that scores is quoting p99 rather than averages — the average hides exactly the preemption and queuing events that make a service feel broken.

---

## 9. Distributed Training Infrastructure

**In plain language: the three ways to split a job.** *Data parallelism* gives every GPU a full copy of the model and a different slice of the batch; they average their gradients at the end of each step. It's the simplest and it's what you use until the model stops fitting on one GPU. *Tensor parallelism* cuts each individual layer's matrices into pieces so several GPUs compute one layer together — that requires talking to each other multiple times per layer, so it lives inside a single NVLink-connected node, usually eight GPUs. *Pipeline parallelism* gives each GPU a different consecutive stack of layers, like stations on an assembly line, so the chatter is only at the handoffs and it can cross nodes. Real frontier runs use all three at once plus ZeRO/FSDP sharding of the optimizer state, and the named cost of pipelining is the "bubble" — idle GPUs waiting for the first micro-batch to work its way down the line.

The training side. Detailed coverage in [`61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md`](../61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md) — production focus here.

### 9.1 The library landscape

- **PyTorch DDP.** Default for single-node multi-GPU and small multi-node.
- **PyTorch FSDP.** ZeRO-3 for production. Default at Meta / Anthropic for training.
- **DeepSpeed.** Microsoft library. ZeRO 1/2/3, ZeRO-Infinity (CPU/NVMe offload), pipeline parallelism, MoE support.
- **Megatron-LM.** NVIDIA library. Best-in-class tensor parallelism + pipeline. Used by Bloom, MT-NLG, many internal models.
- **Megatron-DeepSpeed.** Combines both. NVIDIA + Microsoft hybrid.
- **NeMo Framework.** NVIDIA wrapper around Megatron + Triton + others. Production-grade.
- **MosaicML Composer / LLM Foundry.** Now Databricks. Optimized Llama-style training.

### 9.2 The job-launch stack

- **Slurm.** HPC-style scheduler. Most common at academic + many cloud LLM teams.
- **Kubernetes.** With KubeFlow, Volcano, or Run:ai. More common at startups.
- **Ray.** Python-native distributed framework. Anyscale's product. Increasingly common for training + serving in one stack.
- **AWS / GCP / Azure managed services.** SageMaker HyperPod, Vertex AI, Azure ML.

### 9.3 Reliability essentials

- **Frequent async checkpointing.** Every 30-60 minutes; async write to remote storage so training doesn't pause.
- **Fast checkpoint restore.** Sharded parallel reads. < 5 min target on a 70B+ model.
- **Hardware fault tolerance.** A 1000-GPU run sees daily failures. Use libraries (Megatron, NeMo) with built-in retry-and-restart-from-checkpoint.
- **Loss-spike detection.** Auto-rollback to last good checkpoint if loss spikes > N×.
- **Slow-worker / straggler detection.** Replace lagging GPUs.
- **Network fabric monitoring.** InfiniBand link flaps cause silent perf drops.

### 9.4 Cost / capacity planning

- Training a 7B model from scratch: ~50-200 H100-days. ~\$4-15K cloud.
- 70B from scratch: ~5K-15K H100-days. \$400K-\$1.5M.
- 400B+: 100K+ H100-days. \$10M+.
- Fine-tuning: 10-100× cheaper than from-scratch.

> **Saying it out loud.** Above a few hundred GPUs, training stops being an ML problem and becomes a reliability problem. A thousand-GPU run sees hardware failures daily, so the real engineering is asynchronous checkpointing every thirty to sixty minutes, sharded restore that gets you back in under five minutes, and automatic rollback when the loss spikes. You combine the three parallelism strategies according to the network: tensor parallel inside a node on NVLink, pipeline and data parallel across nodes on InfiniBand. The failure mode that actually costs money isn't a crash — it's the straggler, one slow GPU or a flapping InfiniBand link that silently drags the whole synchronous step down, so every other GPU in the job idles at its cost per hour while nothing errors out.

---

## 10. Model Serving and Autoscaling

Where rubber meets road.

### 10.1 The platform layer

- **NVIDIA Triton Inference Server.** Production gold standard. Multi-model, multi-framework, dynamic batching, model ensembles.
- **KServe (formerly KFServing).** Kubernetes-native. Standard CRDs for InferenceService.
- **BentoML.** Python-first model packaging + serving.
- **Ray Serve.** Ray's serving layer. Good for complex pipelines (multi-step inference, RAG).
- **Modal, Baseten, Replicate.** Serverless GPU services. Pay per second of GPU use. Good for variable workloads.
- **Together AI / Fireworks / Anyscale Endpoints.** Hosted inference for popular OSS models. Cheaper than self-hosting at moderate scale.

### 10.2 Multi-model serving

- **Single replica per model.** Wasteful at low QPS.
- **Multi-model on one GPU.** Multiple checkpoints in VRAM; dispatch by request. Tradeoff with KV cache.
- **LoRA multi-tenancy.** One base model + many LoRA adapters. S-LoRA (Punica), vLLM `--enable-lora`. Massive cost win for fine-tune-per-customer products.
- **Model swap on demand.** Pull weights from S3 when needed; cold start cost. Used for long-tail models.

### 10.3 Autoscaling

The hard part of GPU autoscaling: **GPUs take 60-300s to come up (provision + model load)**, far slower than CPU autoscaling. Strategies:

- **Provision to peak.** Expensive but reliable.
- **Predictive autoscaling.** Provision for forecasted demand (e.g., based on time of day).
- **Warm pool.** Keep N spare GPUs ready (idle cost).
- **Burst to spot/on-demand mix.** Baseline reserved, peak on-demand.
- **Serverless GPU (Modal, Baseten, Replicate, Cloudflare Workers AI).** Sub-second cold starts via shared base model + per-request adapter.

### 10.4 Routing

- **Latency-aware routing.** Send request to least-loaded replica (queue depth, in-flight tokens).
- **Affinity routing.** Send to replica with hot prefix cache for this user (common with LangChain + vLLM).
- **Model routing.** Route easy queries to cheap small model, hard queries to expensive big one (RouteLLM, Martian).

### 10.5 Deployment patterns

- **Blue-green.** Two parallel deployments; cut over instantly.
- **Canary.** New version gets 1-5% of traffic; monitor; ramp.
- **Shadow.** Mirror traffic to new version; compare outputs offline; no impact.
- **A/B for quality.** Random assignment; collect quality signals.

> **Saying it out loud.** Autoscaling GPUs is much harder than autoscaling web servers, because a cold start means pulling tens of gigabytes of weights and warming an engine — that's minutes, not seconds. So you don't scale on CPU utilization; you scale on queue depth or time-to-first-token, and you scale up much earlier than feels necessary. Scale-to-zero is only honest for genuinely bursty internal workloads, because the first user after a scale-down eats the whole cold start. The tradeoff to name is idle cost versus tail latency: you're paying for warm GPUs that are doing nothing precisely so that your p99 doesn't include a model load, and the right amount of that waste is a business decision, not an engineering one.

---

## 11. Vector DB Retrieval Pipelines

The retrieval side of RAG products. Detailed RAG coverage in `39_rag_retrieval_augmented_generation/`. Infrastructure focus here.

### 11.1 The vector DB landscape

| DB | Type | Strengths |
|---|---|---|
| **Pinecone** | Managed | Easy, serverless, popular. Pricier. |
| **Weaviate** | Self-hosted / managed | GraphQL, hybrid search built-in. |
| **Qdrant** | Self-hosted / managed | Rust, fast, payload filters. |
| **Milvus** | Self-hosted | Scales to billions of vectors. Open source. |
| **pgvector** (Postgres) | Self-hosted | If you already have Postgres. Limited at scale (>10M). |
| **Chroma** | Self-hosted | Easy local dev. Not for prod scale. |
| **Vespa** | Self-hosted | Full-text + vector + ranking, used by Yahoo. |
| **Elasticsearch / OpenSearch** | Self-hosted / managed | Lexical + dense hybrid. |
| **LanceDB** | Embedded | Single-binary, fast for moderate scale. |
| **Turbopuffer** | Managed | Cost-optimized for cold storage. |

### 11.2 Index types

- **Flat.** Brute-force exact search. Up to ~100K vectors.
- **HNSW (Hierarchical Navigable Small World).** Graph-based. Default for most DBs. Trades memory for accuracy.
- **IVF (Inverted File).** Cluster, search nearest clusters. More memory-efficient than HNSW.
- **IVF-PQ (Product Quantization).** IVF + compressed vectors. Best memory efficiency. Some recall loss.
- **DiskANN.** Disk-based ANN. Billion-vector scale on a single machine.

**Hook.** "HNSW for moderate scale, IVF-PQ for billion-scale, flat only for tiny indexes."

### 11.3 Retrieval pipeline architecture

```
Query
  ↓
Embedding model (e.g., text-embedding-3-small, BGE, Cohere embed)
  ↓
Vector DB → top-K candidates (often K=50-100)
  ↓                     ↘
Lexical (BM25) → top-K   Hybrid retrieval (RRF / weighted)
  ↓                     ↙
Reranker (Cohere, BGE-reranker, ColBERT)
  ↓ top-N (often N=5-20)
LLM generator with retrieved context
```

### 11.4 Production gotchas

- **Embedding model versioning.** Re-embed everything when you change models. Coordinate across services.
- **Sharding.** Vectors by tenant or by topic. Per-tenant indexes for isolation.
- **Replication.** Read replicas for high QPS.
- **Index rebuilds.** Most DBs need offline rebuild on schema change. Plan for double the storage during rebuild.
- **Latency budget.** Retrieval is on the critical path → 50-100ms p95 for retrieval+rerank.
- **Hybrid retrieval.** Almost always wins over pure dense. RRF or per-query weighting.

### 11.5 Embedding models (production menu)

- **OpenAI text-embedding-3-small/large.** Easy, paid.
- **Cohere embed-multilingual-v3.** Strong multilingual.
- **BGE (BAAI General Embedding).** Open weights, strong performance.
- **E5 (Microsoft).** Open weights.
- **Jina embeddings.** Multilingual, multimodal.
- **NV-Embed.** NVIDIA's massive embedding model.
- **Domain-specific finetunes** of any of the above.

> **Saying it out loud.** A vector database is an approximate nearest-neighbor index with an operations story bolted on, and the operations story is usually why you pick one. Exact search is fine up to a few million vectors; past that you use HNSW, which is fast and accurate but expensive in RAM, or IVF-PQ, which compresses hard and trades away recall. The part that bites teams is that the recall knob — efSearch, or nprobe — silently degrades retrieval quality with no error and no alert, so somebody tunes it down for latency and the product gets worse in a way that never shows up in a dashboard. The failure mode to name is embedding-model drift: change the embedding model and every vector in the index is now in a different space, so you must reindex the entire corpus, and a partial reindex gives you garbage results from the mixed portion.

---

## 12. Prompt Caching and Cost Optimization

Where production money lives.

### 12.1 Prefix / prompt caching

- **Lexical prefix caching.** Hash-based cache; system prompt + chat history reused identically. vLLM, SGLang, Anthropic, OpenAI offer this. **Often 50-90% cost reduction** on chat workloads.
- **Semantic caching.** Look up similar past queries; if a near-match exists with a previous answer, return cached answer. Risk: false positives.
- **OpenAI / Anthropic Cached Tokens API.** First-party prompt caching with 50-90% discount on cache hits.

### 12.2 Model routing

- **Cheap model first.** Try a small model; if confidence low, escalate.
- **Task-based routing.** Code → coding model; chat → chat model; etc.
- **RouteLLM, Martian, NotDiamond.** Productized routing layers. ~50-90% cost cut at minimal quality loss.

### 12.3 Batch APIs

OpenAI Batch API, Anthropic Batches: half-price for non-realtime workloads. Use for:
- Bulk embedding generation.
- Offline data labeling.
- Synthetic data generation.
- Eval runs.

### 12.4 Output limiting

- Set max_tokens aggressively.
- Stop sequences.
- Structured outputs (JSON schema) are typically shorter than free-form.

### 12.5 Tenant cost attribution

- Token counts per request, per tenant.
- Aggregated dashboards.
- Per-tenant budgets / rate limits.
- Cost-plus billing for B2B SaaS.

### 12.6 Cost-optimization checklist

1. Enable prompt caching everywhere.
2. Route to smallest viable model.
3. Use batch API for non-realtime.
4. Quantize aggressively (FP8 weights, FP8 KV cache).
5. Use speculative decoding.
6. Set tight output token limits.
7. Cache embeddings (don't re-embed identical content).
8. Cache final responses for FAQ-style queries.
9. Track cost per (user, route, day).
10. Set alarms on outlier-cost requests.

> **Saying it out loud.** The cheapest token is the one you never compute, and in chat and agent workloads a huge fraction of every request is an identical prefix — the same system prompt, the same tool definitions, the same conversation so far. Prefix caching keeps that prefix's KV cache around so you skip re-running prefill on it, and on real chat traffic that's commonly a fifty-to-ninety percent cost reduction, which dwarfs anything you'd get from squeezing the model. After that it's routing cheap queries to a small model and pushing anything non-interactive onto a batch API. The tradeoff worth naming is semantic caching, where you serve a stored answer for a merely-similar question — it saves the most and it's the one that will eventually return a confidently wrong answer to somebody, so it needs a tight similarity threshold and a way to see how often it fires.

---

## 13. Observability for LLM Apps

This is the half nobody teaches but interviewers care about.

### 13.1 The core trace

For a chat / agent request, you want a trace that captures:

```
Request (user_id, conversation_id, request_id)
├── LLM call 1 (model, tokens_in, tokens_out, ttft, tpot, cache_hit)
│   ├── Prompt (with PII redacted)
│   └── Response
├── Tool call 1 (tool_name, args, latency, status)
│   └── Tool response
├── LLM call 2 ...
├── Retrieval call (query, top-K results, hybrid weights)
└── Final response
```

### 13.2 The platform menu

- **LangSmith (LangChain).** Most popular for LangChain users. Trace + eval + dataset management.
- **Langfuse.** Open source LangSmith alternative. Self-host or cloud.
- **Helicone.** Drop-in proxy that logs everything. Easy adoption.
- **Arize Phoenix.** Open source. Strong eval features.
- **Weights & Biases Weave.** From W&B. Trace + eval.
- **Datadog LLM Observability.** Enterprise.
- **Honeycomb.** Generic distributed tracing; works for LLM with custom spans.
- **OpenTelemetry GenAI semantic conventions.** Vendor-neutral standard. Use this for portability.

### 13.3 Online + offline eval

- **Online (production traffic).** Sample N% of requests, run eval pipeline (LLM-judge, programmatic checks). Alert on regression.
- **Offline (golden set).** 500-5000 hand-curated examples. Run on every model swap.
- **A/B (canary).** Compare new model on live traffic with quality metrics.

### 13.4 Drift detection

- **Input drift.** User-prompt distribution shift. Detect via embedding drift on prompts.
- **Output drift.** Response distribution shift. Length, refusal rate, formatting.
- **Latency drift.** Sudden p95 jumps.
- **Cost drift.** \$/request creeping up.

### 13.5 Logging — privacy and compliance

- **PII redaction** before logging. Use Presidio, AWS Comprehend, or in-house regex+NER.
- **Retention policy.** GDPR, HIPAA, SOC2.
- **Per-tenant isolation.** Don't mix logs across customers.
- **Encryption at rest.**
- **Access audit logs.** Who looked at what, when.

### 13.6 Production alarm set

- TTFT p95 > SLO.
- TPOT p95 > SLO.
- Error rate > 0.5%.
- Cache hit rate dropped > 10%.
- Cost per request > 1.5× baseline.
- Quality eval score dropped > 2 points.
- Refusal rate spiked > 20%.
- VRAM utilization > 95% for > 5 min.

> **Saying it out loud.** LLM observability is different from normal service monitoring because the failure you care about most returns a 200. The model answers fluently and the answer is wrong, and no latency graph will ever show you that. So the trace has to carry the semantic layer — the prompt, the retrieved chunks, the tool calls, the tokens in and out, the cost — not just spans and status codes. Then you run a small online judge on a sample of live traffic and a fixed offline eval set on every deploy. The alarm set that matters is error rate, p99 latency, cost per request, and a quality score that can actually page someone, because if quality has no alarm, quality regressions get discovered by customers.

---

## 14. Capacity Planning and Cost Modeling

The senior task: estimate hardware for a planned product.

### 14.1 The estimation flow

1. **QPS forecast.** Daily active users × avg requests per user / 86400. Add headroom (2-3×).
2. **Avg input + output tokens** per request.
3. **Throughput per GPU** = experimental measurement on the inference engine. Run benchmarks at expected batch size and context length.
4. **GPUs needed** = QPS × avg tokens-out / TPS-per-GPU. Add 30-50% buffer.
5. **VRAM check.** Model + KV cache (peak concurrency × max context × KV/token) ≤ available VRAM.
6. **Cost** = GPUs × \$/hour × hours. Add storage, network egress, observability tools.

### 14.2 A worked example

Building a coding assistant for 100K DAU. Each user makes 10 requests/day, avg 2k input + 500 output tokens.

- QPS = 100K × 10 / 86400 ≈ 12 req/s. Peak ~30 req/s.
- Total output tokens/s at peak = 30 × 500 = 15,000 tps.
- On Llama 3 70B FP8 with vLLM on H100, measured throughput ≈ 8000 tps per GPU at batch=64.
- GPUs needed ≈ 15,000 / 8,000 × 1.5 (buffer) ≈ 3 GPUs.
- TP=2 for memory → 6 GPUs total = 1 DGX node.
- Cost: ~\$5/hour/GPU × 6 = \$30/hour = ~\$22K/month.
- Per user: \$22K / 100K = \$0.22/user/month. Need to charge ≥ \$1/user/month for healthy margin.

That's the calculation.

### 14.3 Common interview question

> "We have a chatbot product with 1M MAU, 10% DAU. Estimate GPU costs for a Llama 3 70B deployment."

Walk the flow above. Show your work. State assumptions (avg session length, avg QPS per user, etc.).

> **Saying it out loud.** I'd do this out loud as arithmetic, because the interviewer is scoring the method, not the answer. Daily actives times requests per user divided by 86,400 gives average QPS; multiply by two or three for peak. Peak QPS times average output tokens gives the tokens per second I have to produce. Divide that by measured throughput per GPU — measured, not from a spec sheet — add fifty percent buffer, then check that the weights plus the peak KV cache actually fit in VRAM, which often forces more GPUs than throughput alone did. Then multiply by the hourly rate. The move that makes it a senior answer is stating every assumption as you go and finishing with the per-user number, because a hundred thousand users at twenty-two thousand dollars a month is twenty-two cents a head, and that's the number that tells you whether the product can exist.

---

## 15. Reliability — Checkpointing, Fault Tolerance, Blue-Green

The survival skills.

### 15.1 Inference reliability

- **Multi-replica per model.** N+1 redundancy. Loss of one replica doesn't break service.
- **Multi-AZ.** Replicas across availability zones.
- **Circuit breakers.** If model is timing out → fail fast, don't queue.
- **Graceful degradation.** Big model down → fall back to smaller model.
- **Request retries with backoff** at the client.
- **Health checks.** Liveness + readiness probes. Auto-restart on failure.

### 15.2 Training reliability

(Already covered in §9.3.) Async checkpointing, retry-on-failure, slow-worker replacement, loss-spike auto-rollback.

### 15.3 Blue-Green / Canary deploys for inference

- **Blue-green.** Deploy new version on parallel cluster; flip load balancer; keep old as instant rollback for ~24h.
- **Canary.** 1% → 5% → 25% → 100% over hours; monitor error rate, latency, quality at each step.
- **Shadow.** New version sees 100% mirrored traffic, returns ignored. Compare outputs offline. Useful before any user-facing rollout.

### 15.4 Disaster recovery

- **Model artifact backups.** Multiple regions, with checksums.
- **Config-as-code.** All deploy config in git. Reproducible deploys.
- **Runbook.** Documented procedures: model swap, region failover, full service restart.
- **Game days.** Practice failure scenarios.

> **Saying it out loud.** Reliability for inference is mostly ordinary distributed-systems hygiene applied to unusually expensive replicas: N-plus-one across availability zones, circuit breakers so a timing-out model fails fast instead of building a queue, and a smaller fallback model so a degraded answer beats no answer. The interesting part is the rollout, because you can't unit-test a model's quality. So you shadow first — mirror real traffic to the new version and throw the responses away — then canary at one percent, five, twenty-five, watching error rate and latency and a quality score at each step. The thing to name is that rollback has to be instant, which means keeping the old version warm for a day rather than tearing it down the moment the new one is live.

---

## 16. Security at the Infrastructure Layer

(Cross-reference: [`65_llm_security/LLM_SECURITY_DEEP_DIVE.md`](../65_llm_security/LLM_SECURITY_DEEP_DIVE.md) covers prompt injection, jailbreaks, lethal trifecta. Here, infra-layer concerns.)

- **Secrets management.** Vault, AWS KMS, GCP Secret Manager. Never bake API keys into images.
- **Network isolation.** Private VPC, no public LLM endpoints unless authenticated.
- **API gateway.** Rate limiting, auth (OAuth/JWT), IP allowlists.
- **Per-tenant isolation.** Don't leak data across tenants in shared cache, retrieval, logs.
- **Output sanitization.** Strip prompt-injection payloads from rendered output (HTML escape, markdown sanitize).
- **Egress filtering.** Tool-using agents → allowlist destinations, deny private IPs / cloud metadata.
- **Sandbox for code execution.** gVisor / Firecracker / nsjail per request. Read-only fs except scratch. Time limits. No network unless allowlisted.
- **Audit logs.** Who deployed what, when. SOC 2 / ISO 27001.
- **Vulnerability scanning.** Container scanning, dependency scanning. SBOM.
- **Compliance.** GDPR, HIPAA, SOC 2, FedRAMP — depending on customer base.

> **Saying it out loud.** At the infra layer I stop thinking about clever prompts and start thinking about blast radius: assume the model will eventually be talked into doing something bad, and make sure it can't reach anything that matters when it does. Concretely that's egress filtering on anything with tools — allowlist the destinations and explicitly deny private IP ranges and the cloud metadata endpoint, because that's how a prompt injection turns into stolen instance credentials. Code execution goes in a per-request sandbox like gVisor or Firecracker with a read-only filesystem, a time limit, and no network. And per-tenant isolation has to extend into the caches and the retrieval index, not just the database — shared prefix caches and shared vector indexes are the two places tenant data leaks across without anyone writing an obviously insecure line of code.

---

## 17. The Full Production Stack — How It Fits Together

The senior interview question: *"Walk me through the full architecture of a production LLM product."*

```
                 ┌─────────────────┐
                 │ Client / SDK    │
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  API Gateway    │  ← Auth, rate limit, request validation
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  Router         │  ← Model routing, A/B, feature flags
                 └────────┬────────┘
                 ┌────────▼────────┐  ┌──────────────┐
                 │  Pre-process    │←→│ Embedding    │
                 │  (PII redact,   │  │ service      │
                 │   prompt build) │  └──────┬───────┘
                 └────────┬────────┘         │
                          │            ┌─────▼────┐
                          │            │ Vector   │
                          │            │ DB       │
                          │            └──────────┘
                 ┌────────▼────────┐
                 │  LLM Engine     │  ← vLLM / TRT-LLM
                 │  (multi-replica)│
                 └────────┬────────┘
                 ┌────────▼────────┐
                 │  Tool / Agent   │
                 │  orchestration  │
                 └────────┬────────┘
                 ┌────────▼────────┐
                 │  Post-process   │  ← Sanitize, cite, format
                 └────────┬────────┘
                          │
                 ┌────────▼────────┐
                 │  Response       │
                 └─────────────────┘

  Cross-cutting:
  - Observability (LangSmith / Langfuse / Helicone)
  - Cost tracking
  - Online eval sampling
  - Audit logs
  - Cache (prompt prefix + final response)
```

**Components:**
- API gateway (Kong, Tyk, AWS API Gateway).
- Auth (Auth0, Cognito, internal OAuth).
- Rate limiting per (tenant, endpoint, time).
- Router (custom, RouteLLM, Martian).
- Pre-process: PII detection (Presidio), prompt assembly (Jinja).
- Inference engine cluster (Triton, KServe).
- Vector DB (Pinecone, Qdrant, etc.).
- Reranker (Cohere reranker, BGE, ColBERT).
- Cache (Redis, in-engine prefix cache).
- Observability stack (OpenTelemetry → Datadog / Langfuse / Honeycomb).
- Eval pipeline (offline golden + online sample → LangSmith / custom).
- Logging (S3, Snowflake) with PII redaction.
- Multi-region / multi-AZ deployment.
- Blue-green or canary deployer (Argo Rollouts, Spinnaker).

That's the full picture.

> **Saying it out loud.** I'd walk it front to back and keep saying why each box exists. Client hits a gateway that does auth, rate limiting, and validation, so nothing unauthenticated ever reaches a GPU. Behind it an orchestration layer decides whether this request needs retrieval, which model tier it deserves, and whether the prefix is already cached. Then the inference layer — vLLM or TRT-LLM with paged KV and continuous batching, multiple replicas across zones behind a router that's prefix-aware so cache hits stay on the same replica. Retrieval hangs off the side with a vector index and a reranker, and observability wraps the whole thing so every request has a trace with tokens, cost, and latency attached. The framing that scores: the gateway protects the GPUs, the caches protect the budget, and the replicas protect the SLO.

---

## 18. Senior Signals

What separates "knows the words" from "has shipped this."

- **You start with constraints.** SLO targets first, then design.
- **You name the inference engine and version.** "vLLM 0.6 with --enable-chunked-prefill --kv-cache-dtype fp8 on H100s."
- **You quantify with KV math.** "70B BF16 KV cache is 327 KB/token; 32K context × batch 8 = 84 GB."
- **You distinguish prefill from decode regimes** and their bottlenecks.
- **You name PagedAttention, continuous batching, chunked prefill** by name and explain why each matters.
- **You distinguish vLLM, TRT-LLM, TGI, SGLang** — what each is best at.
- **You name observability platforms** beyond "we log it" (LangSmith, Langfuse, OpenTelemetry).
- **You think about cost** at the per-request and per-tenant level.
- **You bring up disaggregated serving** for very-large-scale.
- **You discuss multi-tenancy** (S-LoRA, prefix caching for shared system prompts).
- **You're cautious about quantization** (validate quality, don't blindly enable).
- **You design for failure** (multi-replica, blue-green, circuit breakers, fallback model).
- **You quantify GPU economics** (rent vs reserved, spot, H100 vs H200 vs B200).
- **You separate experiment / staging / prod** environments and config.

> **Saying it out loud.** The tell for seniority in this interview isn't knowing more acronyms, it's the order you say things in. Start from constraints — what's the latency SLO, what's the cost ceiling, what's the traffic shape — and let the design fall out of them, rather than proposing an architecture and defending it. Name specific software with versions and flags, because that's the part you can only get from having run it at three in the morning. Quantify with KV math rather than adjectives. And be visibly cautious about the two things that quietly break: quantization quality, which needs real task evals and not perplexity, and anything you turned on for throughput that costs you p99.

---

## 19. References

### Inference engines
- vLLM — github.com/vllm-project/vllm
- TensorRT-LLM — github.com/NVIDIA/TensorRT-LLM
- TGI (HuggingFace) — github.com/huggingface/text-generation-inference
- SGLang — github.com/sgl-project/sglang
- LMDeploy — github.com/InternLM/lmdeploy

### Serving platforms
- NVIDIA Triton — github.com/triton-inference-server/server
- KServe — kserve.github.io
- Ray Serve — docs.ray.io/en/latest/serve
- BentoML — bentoml.com

### Vector DBs
- Pinecone, Weaviate, Qdrant, Milvus (each has docs)
- pgvector — github.com/pgvector/pgvector
- DiskANN — github.com/microsoft/DiskANN

### Observability
- LangSmith — smith.langchain.com
- Langfuse — langfuse.com
- Helicone — helicone.ai
- Arize Phoenix — github.com/Arize-ai/phoenix
- OpenTelemetry GenAI — opentelemetry.io/docs/specs/semconv/gen-ai/

### Foundational papers
- Orca (continuous batching) — Yu et al. 2022.
- PagedAttention — Kwon et al., vLLM, 2023.
- Speculative Decoding — Leviathan et al. 2023, Chen et al. 2023.
- SmoothQuant — Xiao et al. 2022.
- AWQ — Lin et al. 2023.
- GPTQ — Frantar et al. 2022.
- S-LoRA — Sheng et al. 2023.
- Mooncake (disaggregated) — Qin et al. 2024.
- DistServe — Zhong et al. 2024.

### Tutorials / blogs
- vLLM blog — blog.vllm.ai
- NVIDIA Triton tutorials
- Anyscale blog (Ray + serving)
- Together AI engineering blog
- Fireworks blog
- Lilian Weng — *Inference Optimization* (2023).
- Sebastian Raschka — *Building LLMs from Scratch*.

### Cross-references in this repo
- [`06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`](../06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md)
- [`61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md`](../61_large_scale_llm_systems/EFFICIENT_TRAINING_INFERENCE_PLAYBOOK.md)
- `63_paged_attention_and_llm_serving/`
- [`41_mixture_of_experts/MOE_DEEP_DIVE.md`](../41_mixture_of_experts/MOE_DEEP_DIVE.md)
- [`39_rag_retrieval_augmented_generation/RAG_DEEP_DIVE.md`](../39_rag_retrieval_augmented_generation/RAG_DEEP_DIVE.md)
- [`65_llm_security/LLM_SECURITY_DEEP_DIVE.md`](../65_llm_security/LLM_SECURITY_DEEP_DIVE.md)

---

## 20. Interview Grill — 100 questions

### A. GPU / VRAM (Q1–10)
1. What's the difference between SRAM, L2, HBM on a GPU?
2. Compare A100 / H100 / H200 / B200 on VRAM and HBM bandwidth.
3. What is NVLink and when does it matter?
4. Why is decode memory-bandwidth-bound but prefill compute-bound?
5. Why does TP usually stay within a node?
6. What's the FP16 TFLOPS of an H100 SXM?
7. How much VRAM does a 70B BF16 model take? FP8? INT4?
8. What's the rough \$ / hour for an H100 on cloud?
9. What's NVSwitch?
10. When would you pick H200 over H100?

> **Saying it out loud.** *(The one they'll actually ask: why decode is memory-bandwidth-bound and prefill isn't.)* Prefill has the whole prompt in hand, so it can score thousands of tokens in parallel against one read of the weights — the arithmetic per byte loaded is high and the tensor cores stay busy. Decode has exactly one token, so it re-reads every weight and the entire KV cache out of HBM to produce that one token, and the arithmetic per byte is near zero. So the GPU is sitting there waiting on memory with most of its FLOPs unused. The number that makes it concrete: an H100 has about 3.35 TB/s of bandwidth, so a 140 GB BF16 70B model caps out around 24 tokens per second per GPU no matter how much compute you throw at it.

### B. Quantization (Q11–18)
11. What are FP8 E4M3 and E5M2?
12. SmoothQuant — what problem does it solve?
13. AWQ vs GPTQ — when each?
14. Per-tensor vs per-channel vs per-group quantization?
15. INT8 W8A8 vs INT8 W8A16?
16. KV cache quantization — quality risk?
17. How do you validate quantized model quality?
18. When would you NOT quantize?

> **Saying it out loud.** *(The one they'll actually ask: how do you validate a quantized model.)* Not with perplexity — that's the trap. Perplexity is an average over easy tokens and it barely moves even when the model has gotten meaningfully worse. What I'd do is run the actual task evals side by side against the unquantized baseline, weighted toward the places degradation concentrates: long context, multi-turn, structured output, and code. Then a human or judge comparison on a sample of real production prompts. The tradeoff to name is that FP8 on Hopper is basically free quality-wise and INT4 is not — INT4 buys you the model fitting on one GPU, and you pay for it in exactly the hard cases your perplexity check didn't cover.

### C. Batching (Q19–25)
19. Compare static / dynamic / continuous batching.
20. What's chunked prefill?
21. Why is continuous batching 5-10× faster than static?
22. What's disaggregated serving?
23. How does batch size affect TTFT vs TPOT?
24. What's iteration-level scheduling?
25. Where does Orca fit historically?

> **Saying it out loud.** *(The one they'll actually ask: why continuous batching is five to ten times faster than static.)* Because in static batching the whole batch is held hostage by its longest sequence. If sixty-three requests finish in fifty tokens and one runs to two thousand, those sixty-three slots sit idle for the rest of the batch — you've reserved GPU time and produced nothing with it. Continuous batching schedules per decode iteration instead of per batch, so the instant a sequence emits its stop token, a queued request takes the slot. The GPU stays full. The tradeoff to name is that the extra admissions can push time-to-first-token up for everyone, which is why chunked prefill exists — it interleaves prefill work in slices so a long prompt can't stall the decode loop.

### D. Inference engines (Q26–34)
26. What's vLLM's killer feature?
27. When do you pick TRT-LLM over vLLM?
28. What does SGLang excel at?
29. What's TGI?
30. Compare vLLM vs TRT-LLM in 30 seconds.
31. What does PagedAttention solve?
32. What's RadixAttention (SGLang)?
33. What's the typical inference engine config for a 70B production deploy?
34. What inference engine would you pick for an agentic workload with heavy structured output?

> **Saying it out loud.** *(The one they'll actually ask: when do you pick TensorRT-LLM over vLLM.)* When you're locked to NVIDIA anyway, the model and shapes are stable, and you've squeezed everything else and still need the last twenty or thirty percent. TRT-LLM compiles a model-specific engine, which is where the speed comes from and also where the pain comes from — every model change, every precision change, and often every NVIDIA release means rebuilding, and the build is slow and fussy. vLLM is the default for basically everyone else: it iterates fast, supports new architectures within days, and runs on AMD too. The tradeoff in one line is engineering velocity versus peak throughput, and most teams should not be trading away velocity.

### E. KV cache (Q35–42)
35. KV cache size formula?
36. How does GQA shrink KV cache?
37. How does MLA shrink KV cache?
38. What's prefix caching and when does it help most?
39. What KV eviction policies exist?
40. CPU swap-space — when to use?
41. KV quantization to FP8 — quality cost?
42. How does PagedAttention compare to flat allocation?

> **Saying it out loud.** *(The one they'll actually ask: do the KV cache math on the spot.)* The formula is two — for K and V — times layers, times KV heads, times head dimension, times tokens, times bytes per element, and then times batch size. Walk it out loud and plug numbers in. For a 70B with GQA in FP16, that lands around a couple hundred kilobytes per token, so a 32K context at batch 8 is tens of gigabytes sitting on top of your weights. That's the calculation that decides your hardware, and it's why the KV heads term matters so much — GQA cutting 64 KV heads to 8 is an eight-times reduction in that whole number. Quantizing the cache to FP8 halves it again, at some quality cost on long context.

### F. Speculative decoding (Q43–48)
43. Sketch speculative decoding.
44. What's Medusa?
45. What's EAGLE?
46. Why does speculative decoding speedup decrease at large batch?
47. How do you tune speculative tokens?
48. Memory cost of running speculative?

> **Saying it out loud.** *(The one they'll actually ask: when does speculative decoding stop helping.)* At high batch size. The whole trick is spending idle compute to avoid a memory read, and at large batch you no longer have idle compute — you've already amortized the weight loading across many sequences, so you're compute-bound and the draft model's forward passes are pure added cost. It can genuinely make you slower under load. The other thing to name is that the win depends entirely on acceptance rate: if the draft model agrees with the target seventy or eighty percent of the time you get a real speedup, and if it's a poorly matched draft at thirty percent you're burning compute to be rejected. So production systems gate it on current load and monitor acceptance rate as a first-class metric.

### G. Throughput / SLO (Q49–55)
49. Define TTFT, TPOT, TPS, ITL.
50. Typical chat-product TTFT and TPOT SLOs?
51. Tradeoff: small batch vs large batch?
52. What's a good chat TPS for user UX?
53. How do you tune for low TTFT?
54. How do you tune for high throughput?
55. Why monitor cache hit rate?

> **Saying it out loud.** *(The one they'll actually ask: how batch size affects TTFT versus TPOT.)* They pull in opposite directions, and that's the whole design tension in a serving stack. Bigger batches amortize the weight read across more sequences, so per-token decode latency goes down and total throughput goes up — but requests wait longer in the queue before admission, so time-to-first-token goes up. Smaller batches do the reverse: snappy first token, worse throughput, more cost per token. So the SLO decides the batch size, not the other way around. The escape hatch to name is disaggregated serving — put prefill and decode on separate GPU pools so each can be tuned for its own bottleneck, and both numbers improve at the cost of a lot more system complexity.

### H. Distributed training (Q56–63)
56. Difference between DDP and FSDP?
57. ZeRO 1 vs 2 vs 3?
58. When use DeepSpeed vs Megatron vs FSDP?
59. What's slurm vs k8s for training?
60. Async vs sync checkpoint?
61. How do you handle a 1000-GPU run failure?
62. Why is loss-spike detection important?
63. Cost of training a 70B from scratch — order of magnitude?

> **Saying it out loud.** *(The one they'll actually ask: why does tensor parallelism stay inside a node.)* Because tensor parallelism splits an individual layer across GPUs, which means they have to all-reduce partial results multiple times per layer — that's enormous, latency-sensitive communication happening constantly. NVLink inside a server gives you hundreds of gigabytes per second; InfiniBand between servers is roughly an order of magnitude less. Stretch tensor parallelism across nodes and communication, not compute, becomes the whole run. So the standard layout is tensor parallel up to eight inside a DGX node, pipeline parallel across nodes because it only communicates at stage boundaries, and data parallel on top. The named cost of the pipeline dimension is the bubble — idle stages at the start and end of each batch, which you shrink with more micro-batches.

### I. Serving / autoscaling (Q64–71)
64. Compare Triton, KServe, Ray Serve, BentoML.
65. Why is GPU autoscaling slow?
66. What's a warm pool?
67. What's S-LoRA / multi-tenant LoRA?
68. Compare blue-green / canary / shadow deploy.
69. Latency-aware vs affinity routing?
70. When is serverless GPU appropriate?
71. How do you handle cold start?

> **Saying it out loud.** *(The one they'll actually ask: how do you autoscale GPU inference.)* Not on CPU utilization — that metric means nothing here. You scale on queue depth or on time-to-first-token, because those are what the user feels, and you scale up aggressively early because a cold start is minutes: you're pulling tens of gigabytes of weights and warming an engine before that replica serves anything. Scale down slowly and conservatively for the same reason. The tradeoff to name is idle GPU cost versus tail latency — keeping warm capacity is deliberately paying for nothing so that your p99 doesn't contain a model load, and scale-to-zero is only defensible for internal or genuinely bursty workloads where somebody waiting two minutes is acceptable.

### J. Vector DBs (Q72–78)
72. HNSW vs IVF-PQ — tradeoffs?
73. When does pgvector stop being enough?
74. What's hybrid retrieval?
75. What's RRF?
76. How do you handle embedding-model versioning?
77. Latency budget for retrieval+rerank?
78. What's a reranker and which would you use?

> **Saying it out loud.** *(The one they'll actually ask: HNSW versus IVF-PQ.)* HNSW builds a layered navigable graph and gives you excellent recall at low latency, but it holds the full vectors plus the graph in RAM, so it's the expensive option and it's awkward to update in bulk. IVF-PQ partitions the space and then compresses each vector down with product quantization, so memory drops by something like an order of magnitude and you pay in recall. Rule of thumb: HNSW under roughly ten million vectors where quality matters, IVF-PQ above that or when memory is the binding constraint. The failure mode to name is that both have a recall knob — efSearch and nprobe — that silently trades away retrieval quality for latency with no error and no alert when somebody tunes it down.

### K. Cost optimization (Q79–86)
79. What's prompt caching?
80. What's the OpenAI / Anthropic cached-tokens discount?
81. When use Batch API?
82. What's model routing? Tools (RouteLLM, Martian)?
83. Why is per-tenant cost attribution important?
84. Five ways to cut LLM cost.
85. When is quantization not worth it?
86. What's semantic caching, and what's the risk?

> **Saying it out loud.** *(The one they'll actually ask: the chatbot bill is too high, what do you do first.)* Prompt caching, before anything else. In chat and agent traffic an enormous share of every request is an identical prefix — system prompt, tool schemas, prior turns — and caching its KV means you skip prefill on it entirely, which is commonly a fifty-to-ninety percent cost cut for a config change rather than a model change. Then route: send the easy majority of queries to a small model and reserve the big one for the hard tail. Then move anything non-interactive to a batch API for roughly half price. Quantization and better batching come after, because they're more work for less money. The tradeoff to name on routing is that classifying "easy" wrong is a quality regression your cost dashboard will never show you.

### L. Observability (Q87–93)
87. Compare LangSmith / Langfuse / Helicone.
88. What does an LLM trace look like?
89. What's OpenTelemetry GenAI?
90. Online eval vs offline eval?
91. How do you detect drift?
92. What metrics trigger alarms?
93. PII redaction strategies?

> **Saying it out loud.** *(The one they'll actually ask: how do you detect quality regressions in production.)* You need something other than the request status, because the characteristic LLM failure returns a perfectly healthy 200 with a wrong answer. So: a fixed offline eval suite that runs on every deploy and blocks it, plus an online judge scoring a sample of live traffic continuously, plus drift monitors on the input distribution — prompt length, language, topic mix, retrieval scores — because inputs usually shift before outputs visibly break. And then the quality score needs a real alarm attached to it, at the same tier as latency. The named failure mode is silent degradation: nothing errors, no dashboard turns red, and you find out from a customer three weeks later.

### M. Capacity / reliability / security (Q94–100)
94. Estimate GPUs for 100K DAU coding-assistant on Llama 3 70B.
95. What's a circuit breaker?
96. Multi-AZ deployment essentials?
97. How do you handle a model rollback?
98. Secrets management?
99. Egress filtering for tool-using agents?
100. SBOM and why does it matter?

> **Saying it out loud.** *(The one they'll actually ask: estimate the GPUs for this product.)* Say the arithmetic out loud and state every assumption as you make one. Daily actives times requests each over 86,400 seconds gives average QPS; times two or three for peak; times average output tokens gives the tokens per second you must produce; divide by measured per-GPU throughput on your actual engine; add fifty percent buffer; then separately verify weights plus peak KV cache fit in VRAM, which frequently forces more GPUs than throughput did. Multiply by the hourly rate — and flag that hourly rates move, so quote them as of a date. Then close with the per-user cost, because that's the number that decides whether the product has a business model at all.

---

## 21. Drill plan

- **Day 1:** §2-4 (GPU, quantization, batching). Drill A, B, C.
- **Day 2:** §5-8 (engines, KV, speculative, SLO). Drill D, E, F, G.
- **Day 3:** §9-10 (training infra, serving). Drill H, I.
- **Day 4:** §11-12 (vector DBs, cost). Drill J, K.
- **Day 5:** §13-16 (observability, capacity, reliability, security). Drill L, M.
- **Day 6:** §17 (full stack architecture diagram drilled). Whiteboard a production stack from memory.
- **Day 7:** Mixed mock — interviewer picks any of 100 questions; you answer in <60 seconds.

Single sentence to remember: **AI infra engineering = pick engine + quantization + batching for the SLO budget; KV math determines hardware; PagedAttention + continuous batching + prompt caching are the throughput trinity; multi-replica + canary + observability are the reliability trinity.**
