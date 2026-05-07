# Efficient LLM Training & Inference — Interview Playbook

> A unified interview-prep digest covering the full set of optimization techniques for training and deploying large language models efficiently. Organized for fast recall and oral delivery.
> Inspired by Gauri Gupta's interview-prep notes (NeoSigma); expanded with depth and cross-references to other chapters in this repo.

The five blocks: **memory · compute · inference · training-parallelism · communication primitives**. Frontier-lab and big-tech ML systems interviews probe across all five. This chapter gives you a unified mental map, a 30-second oral pitch per topic, and links to deeper coverage elsewhere in the repo.

---

## Table of contents

1. The mental model — why optimize, what fails first
2. Memory optimization (Flash Attention, MQA/GQA, Activation Checkpointing)
3. Compute optimization (Sequence Packing, Efficient Transformers)
4. Inference optimization (KV cache, stateful caching, speculative decoding, quantization)
5. Training optimization (mixed precision, parallelism strategies)
6. Communication primitives — the building blocks
7. Putting it together — recipe for a 70B+ training run
8. Cross-reference map (where each topic is covered in detail in this repo)
9. Interview pitch ladder (30-second / 2-minute / 5-minute per topic)
10. Interview grill — 70 active-recall questions

---

## 1. The mental model

When asked "how do you scale to N billion parameters," the right answer is structured around **what runs out first**:

- **Parameter memory** ($P$ bytes per param × dtype): just storing the weights.
- **Optimizer state memory** (Adam needs ~12 bytes/param at fp32: weights, momentum, variance — much more than weights).
- **Activation memory** (per layer, per micro-batch, per sequence position).
- **KV cache memory** (at inference, dominated by sequence length × num layers × heads × dim).
- **Compute** (FLOPs: prefill $\sim n^2$ in attention; decode dominated by memory bandwidth).
- **Communication bandwidth** (gradient sync at training; KV transfer at inference; cross-device tensor traffic).

The art is: pick the technique that addresses the *current* bottleneck without creating a new one.

> **One-line summary.** Training scales by pipelining + sharding (4 axes: data, tensor, pipeline, expert); inference scales by KV cache + batching + quantization + speculative decoding.

---

## 2. Memory Optimization

### 2.1 Flash Attention

**Problem.** Standard attention is $O(n^2)$ memory and time in sequence length because of the explicit attention matrix.

**Idea.**
- **Tiling.** Decompose Q, K, V into blocks that fit in fast on-chip SRAM. Compute attention block-by-block, never materializing the full $n \times n$ matrix in HBM.
- **Recomputation.** Store only **softmax normalization factors** (which scale linearly with $n$) instead of the full softmax output. On the backward pass, recompute attention from these factors.
- **Online softmax** (Milakov & Gimelshein 2018) is the algorithmic key — softmax can be computed in a single pass with a running max and a running denominator.

**Result.** Linear memory in $n$; ~2-4× wall-clock speedup; identical numerical output (no approximation).

**Variants.** FlashAttention-2 (better parallelism), FlashAttention-3 (Hopper async + FP8).

**Hook.** "Tile Q/K/V into SRAM; store softmax norm factors not softmax outputs; recompute on backward."

**Deep dive.** See `05_attention_mechanisms/ATTENTION_DEEP_DIVE.md`.

### 2.2 Multi-Query / Grouped-Query Attention (MQA/GQA)

**Problem.** KV cache memory at inference scales with `num_heads × seq_len × d_head × num_layers`. For long context, the KV cache dominates.

**MQA (Shazeer 2019).** All Q heads share a *single* K and V head. Memory cut by `num_heads × `, but quality degrades on hard tasks.

**GQA (Ainslie 2023).** Group multiple Q heads to share a single K/V head. With `kv_heads = 8` (groups of 4 in a 32-head model), you get ~`4×` KV-cache savings with negligible quality loss. **The current default in Llama 3, Qwen 2.5, etc.**

**MLA (Multi-head Latent Attention, DeepSeek 2024).** Project K, V, Q into a low-rank latent space; attend in latent space; project back. ~10× KV cache savings versus MHA at near-equal quality.

**Hook.** "Share K/V across heads (MQA), groups (GQA), or via low-rank latent (MLA) — KV cache shrinks proportionally."

**Deep dive.** `05_attention_mechanisms/ATTENTION_DEEP_DIVE.md`, `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`.

### 2.3 Activation Checkpointing (Gradient Checkpointing)

**Problem.** During backprop you need activations from the forward pass. Storing them all blows memory.

**Idea.** Save activations only at *checkpoint* layers (e.g., every $\sqrt{L}$ layers). On backward, recompute the activations for layers between checkpoints.

**Tradeoff.** $O(\sqrt{L})$ activation memory instead of $O(L)$, at the cost of one extra forward pass per backward. Typical extra compute: ~33%.

**Selective checkpointing.** Skip cheap-to-recompute layers (e.g., LayerNorm) and checkpoint only expensive ones (attention, MLP). Most modern frameworks do this automatically.

**Hook.** "Save every √L layers; recompute the rest on backward."

---

## 3. Compute Optimization

### 3.1 Sequence Packing

**Problem.** Pad-to-max means a batch with one long sequence wastes most of its tokens on padding.

**Idea.** Concatenate multiple variable-length sequences into a single fixed-length stream. Use a **document mask** (block-diagonal attention mask) so attention can't cross document boundaries.

**Result.** Near-100% useful-token utilization. Throughput improvement scales with how skewed your length distribution is — often 2-4×.

**Hook.** "Pack sequences end-to-end; use document mask to keep attention within boundaries."

### 3.2 Efficient Transformer Variants (sub-quadratic attention)

For very long context. The main families:

- **Local / sliding window** (Longformer, Mistral, BigBird-local). Attention restricted to a window of size $w$ around each token. $O(n w)$ memory and time.
- **Global tokens** (BigBird, Longformer-global). A few tokens attend to / from everywhere; rest local. Captures global structure cheaply.
- **Random attention** (BigBird). Each token attends to $r$ random tokens. Theoretical claim: combination of local + global + random approximates full attention.
- **Low-rank approximations** (Linformer, Performer). Project K, V to low-rank space.
- **Dilated / hierarchical** (LongNet). Dilation factor grows with depth; lower layers attend locally, upper layers attend across the whole sequence.
- **State-space models** (Mamba, S4). Replace attention with a recurrent state-space convolution; linear in $n$.

**Hook ladder.** Sliding window → global tokens → low-rank projection → SSMs.

**Deep dive.** `42_state_space_models/SSM_DEEP_DIVE.md`, `14_advanced_positional_embeddings/POSITIONAL_DEEP_DIVE.md`.

---

## 4. Inference Optimization

Inference is where money lives in production. These techniques are the difference between profitable and unprofitable LLM products.

### 4.1 KV Cache

**Problem.** Autoregressive decoding recomputes attention over all previous tokens at every step → quadratic decode cost.

**Idea.** Cache K and V tensors for all previously-generated tokens. At each new step, compute only the new K and V; concatenate with cache; attention runs over (new Q) × (full cached K, V).

**Result.** Decode becomes linear in sequence length, not quadratic. ~10-100× speedup on long sequences.

**Memory cost.** `2 (K and V) × num_layers × num_kv_heads × d_head × seq_len × bytes_per_element`. For Llama 70B at 8K context, this is ~3 GB per request. *KV cache, not weights, dominates GPU memory at long context.*

**Advanced KV optimizations:**

- **GQA / MQA / MLA.** Section 2.2 — shrink the heads.
- **Cross-layer KV sharing.** Tie KV cache across consecutive layers; ~2× savings.
- **Interleaved local/global attention.** Local-only for most layers; full attention every 4-6th layer. ~5× savings.
- **PagedAttention (vLLM).** Treat KV cache like virtual memory: variable-size logical pages, indirection-table lookup. Fragments fit; eviction is page-granular. The reason vLLM beats every framework on KV-bound serving.
- **KV cache quantization.** 4-bit or even 2-bit KV cache; 4-8× memory savings with negligible quality loss for moderate-context tasks.
- **KV eviction (StreamingLLM, H2O).** Drop low-attention or middle-of-context tokens from cache. Trades quality for memory at very long context.

**Hook.** "Cache K and V; new step computes only new K, V then attends to the full cache."

**Deep dive.** `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`, `63_paged_attention_and_llm_serving/`.

### 4.2 Stateful / Prefix Caching

**Problem.** Multi-turn conversations re-process the entire context every turn. The system prompt + chat history is identical across many requests.

**Idea.** Cache KV across requests, keyed by **rolling hash of the prefix**. On a new query, find the longest prefix match in cache; load that KV; compute only from the divergence point.

**Implementation.** Tree-structured cache with LRU eviction. Used by Anthropic Claude, OpenAI ChatGPT, vLLM (`enable_prefix_caching`).

**Win.** Often 5-10× speedup on chat workloads where system prompts are 1k+ tokens.

**Hook.** "Hash prefixes; tree cache; LRU evict."

### 4.3 Speculative Decoding

**Problem.** Autoregressive decoding produces one token per forward pass — bandwidth-bound.

**Idea.** A small **draft model** generates K candidate tokens cheaply. The big **target model** runs ONE forward pass that scores all K positions in parallel. Accept the longest correct prefix.

**Math.** Expected speedup ≈ `(α / α_d) × (1 + α + α² + ... + α^k)` where α is per-token acceptance rate. With α=0.7, k=4: ~2.5× speedup.

**Variants.**
- **Vanilla speculative** (Leviathan 2023, Chen 2023): tiny draft model + big target.
- **Medusa, EAGLE, EAGLE-2/3.** Self-speculative — additional heads on the same model predict 2/3/N tokens ahead.
- **Lookahead decoding.** Algorithmic acceleration without a draft model.
- **Server-side speculative** (Mooncake, DistServe): draft and target run on different machines.

**Hook.** "Draft generates K, target verifies in one pass, accept longest correct prefix."

**Deep dive.** `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`.

### 4.4 Quantization

**Problem.** FP32/BF16 weights and activations are memory-heavy and bandwidth-heavy.

**Schemes.**

- **Symmetric vs asymmetric.** Symmetric maps `[-α, α] → [-127, 127]`; asymmetric uses a zero-point.
- **Min/max calibration.** Use observed min/max as quantization range. Outlier-sensitive.
- **MSE.** Choose range minimizing reconstruction MSE. More robust.
- **Cross-entropy.** For softmax outputs, preserve relative ordering of largest values. Argmin of CE between original and quantized softmax.

**Categories.**

- **PTQ (Post-Training Quantization).** Quantize after training, no retraining. Cheap. INT8 typically lossless; INT4 requires care. Modern: GPTQ, AWQ, SmoothQuant — all PTQ variants.
- **QAT (Quantization-Aware Training).** Simulate quantization during training so the model becomes robust to it. Backprop uses the **straight-through estimator (STE)**: gradient passes through the quantizer as if it were identity within range.
- **Mixed-precision.** Use higher precision (FP16/INT8) for sensitive layers (typically attention output projections), lower precision (INT4/INT2) elsewhere. Best memory-quality trade-off.
- **FP8.** Hopper / Blackwell native FP8 (e1m4, e2m3). Used in pretraining and inference at the frontier.
- **FP4 / NF4.** 4-bit float types. NF4 is non-uniform optimized for weight distribution.
- **KV cache quantization.** Even smaller — 4-bit, 2-bit. Big inference savings.

**Frontier 2024-2026.** FP4/FP6 on Blackwell. Models *trained* in FP8 (DeepSeek-V3, frontier). Outlier handling via SmoothQuant / per-channel / per-group quantization.

**Hook.** "PTQ = quantize after; QAT = simulate during; STE = gradient as identity within range."

**Deep dive.** `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md`.

---

## 5. Training Optimization — Parallelism Strategies

The four axes of parallelism: **Data, Tensor, Pipeline, Expert**. Real frontier training combines all four.

### 5.1 Mixed Precision Training

**FP32 vs BF16 vs FP16:**
- **FP32:** 8 exponent bits, 23 mantissa.
- **FP16:** 5 exponent, 10 mantissa. Dynamic range too narrow for stable training (gradients underflow).
- **BF16:** 8 exponent (same as FP32), 7 mantissa. Same range as FP32, less precision. The default for modern training.

**Loss scaling.** When using FP16, scale the loss by `2^k` before backward (e.g., `2^16`); scale gradients back down before the optimizer step. Prevents underflow on small gradients. BF16 doesn't need this thanks to its FP32 exponent range.

**Master weights in FP32.** Optimizer keeps a master copy in FP32; forward/backward use BF16. Final-precision rounding errors don't compound.

**Hook.** "BF16 forward/backward, FP32 master weights, FP32 optimizer state."

### 5.2 Data Parallelism

**DataParallel (legacy).** Single process, multiple threads (Python GIL contention!). Replicate model on every GPU; split batch; average gradients via NCCL. Use **DDP** instead.

**DistributedDataParallel (DDP).** One process per GPU. Each replicates the model; processes a different shard of the batch; gradient sync via **Ring All-Reduce**. The standard.

**Synchronization patterns:**
- **Bulk Synchronous Parallel (BSP).** Sync at every minibatch. Standard. Wait for slowest worker.
- **Asynchronous (ASP).** No wait, but stale gradients hurt convergence. Rare in modern training.

**ZeRO (Zero Redundancy Optimizer, Rajbhandari 2019).** Recognizes that data parallelism replicates *everything* across GPUs (params, gradients, optimizer state). With Adam at FP32 master + BF16 model + BF16 grad: optimizer state ≈ 12 bytes/param, model+grad ≈ 4 bytes/param → optimizer state is the biggest! Three stages:
- **ZeRO-1: shard optimizer state.** 4× memory reduction. Same comm volume as DDP. **Always use this.**
- **ZeRO-2: + shard gradients.** 8× memory reduction. Same comm.
- **ZeRO-3: + shard parameters.** Linear in data-parallel degree (e.g., 64× with 64 GPUs). +50% communication. Used when needed for very large models.

**FSDP (Fully Sharded Data Parallel).** PyTorch's productionized ZeRO-3. Wraps modules; gathers params before forward / backward; releases after. Default for Llama-3+ scale training.

**Hook.** "ZeRO-1 = optimizer state shards; ZeRO-2 + gradient shards; ZeRO-3 + param shards = FSDP."

**Deep dive.** `61_large_scale_llm_systems/`, `62_frontier_training_playbook/`.

### 5.3 Pipeline Parallelism

**Naive model parallel.** Split model across layers, one chunk per GPU. Problem: only one GPU works at a time → bubble.

**GPipe (Huang 2018).** Split mini-batch into M micro-batches. Each GPU works on a different micro-batch in a staggered schedule. Bubble = `(d − 1) / (m + d − 1)` for `d` stages, `m` micro-batches.

**PipeDream (Narayanan 2018).** 1F1B (one-forward-one-backward) schedule. Each worker alternates forward / backward, so backward can start early. Issue: micro-batches may use different model versions → instability. Mitigations:
- **Weight stashing.** Keep multiple model versions per worker.
- **Vertical sync.** Version flows with activation/gradient.
- **PipeDream-flush.** Periodic global sync (like GPipe).
- **PipeDream-2BW.** Only 2 versions, "double-buffered weights."

**Zero Bubble Pipeline (Qi 2023).** Split backward into B-for-input (must run sequentially) and W-for-weights (can run later). Reorder:
- **ZB-H1.** B starts earlier; W passes fill end-bubble.
- **ZB-H2.** Add F passes during warmup; reorder W to eliminate all bubbles.

**DeepSeek DualPipe (V3).** Bidirectional pipeline: feed micro-batches from *both ends* simultaneously. Overlap computation and communication within F+B chunk pairs. Significant comm hide.

**Llama 3 pipeline tweaks.** Reduce one transformer layer from first and last stages (those stages also handle embedding and loss computation). Variable micro-batches per batch. Embedding layer alone on first stage; output projection + loss alone on last.

**Hook ladder.** Naive → GPipe (bubble) → 1F1B → PipeDream-flush → ZB-H1/H2 → DualPipe.

**Deep dive.** `61_large_scale_llm_systems/`, `62_frontier_training_playbook/`.

### 5.4 Tensor Parallelism

**Idea.** Split a single matrix multiply across devices.

**Column-wise.** Split weight columns. Each device computes `X @ A_i`. End: all-gather to concatenate. Used for the up-projection of MLP, Q/K/V projections.

**Row-wise.** Split weight rows AND input columns. Each device computes `X_i @ A_i`. End: all-reduce to sum. Used for the down-projection of MLP, attention output projection.

**Megatron pattern (canonical).** For transformer block:
- Q/K/V: column-wise split (each device has a subset of heads).
- Attention output: row-wise split (devices already have head outputs; row-wise + all-reduce produces final).
- MLP up: column-wise split.
- MLP down: row-wise split.
- Result: only 2 all-reduces per transformer block (one per attention, one per MLP). All-reduces happen after activation, when tensors are smallest.

**Sequence parallelism.** Megatron extension: also split LayerNorm and dropout along sequence dim. Saves activation memory.

**TP degree.** Limited to single node (NVLink) because all-reduce is bandwidth-hungry. Typical: 4 or 8 within a node.

**Hook.** "Column → all-gather; Row → all-reduce. Megatron does column + row in pairs to minimize all-reduces."

**Deep dive.** `61_large_scale_llm_systems/`, `04_transformers/TRANSFORMERS_DEEP_DIVE.md`.

### 5.5 Context Parallelism (a.k.a. Sequence Parallelism, Ring Attention)

**Idea.** Split the *sequence* dimension across GPUs. Each GPU handles a chunk of tokens. For attention, each GPU's queries need keys/values from the full sequence — solved via **Ring Attention**: KV chunks circulate through GPUs in a ring, each GPU does a partial attention update each step.

**Use case.** Very long context (100K+ tokens). When sequence dimension is the dominant memory cost.

**Variants.**
- Ring Attention (Liu 2023).
- DeepSpeed-Ulysses (Microsoft).
- FlashAttention 3 + Ring.

**Hook.** "Sequence shards across GPUs; KV ring-circulates for full attention."

### 5.6 Expert Parallelism (MoE)

**Idea.** Replace dense FFN with a set of experts (e.g., 8 or 64 small FFNs). A gating function routes each token to **top-K experts** (K=1 in Switch Transformer, K=2 in GShard). Only K experts run per token → constant compute even as expert count grows.

**Sharding.** Each expert is on a different GPU. Token-to-expert assignment requires **All-to-All** communication: tokens routed to their destination experts; outputs routed back.

**Routing strategies:**
- **Top-1 (Switch Transformer).** Cheapest but quality-limited.
- **Top-2 (GShard, GLaM).** Standard. Combine outputs by gating weight.
- **Hash routing.** Deterministic, no learned router (PR-MoE).
- **Expert-Choice routing (GShard 2022).** Each expert picks tokens (capacity-bounded). Avoids load-imbalance issues.

**Load balancing — the hard problem.** Naïve top-K routing gives some experts way more tokens than others (popularity skew). Mitigations:
- **Auxiliary load-balance loss.** Penalize uneven distribution. (Standard MoE.)
- **Communication balance loss.** Penalize uneven all-to-all volumes per device.
- **Auxiliary-free balancing (DeepSeek-V3).** Add a learnable bias to each expert's score; bump down over-loaded experts, bump up under-loaded ones — no extra loss term.
- **Capacity factor.** Hard cap on tokens per expert. Excess tokens are *dropped* (priority dropping) or routed to next-best expert.

**Frontier MoE models (2024-2026):**
- Mixtral 8×7B / 8×22B (Mistral).
- DBRX (Databricks).
- DeepSeek-V3 (671B params, 37B activated, 256 experts top-8 routing + auxiliary-free balancing + DualPipe).
- GPT-4 (rumored 16-expert MoE).

**Hook.** "Sparse activation (top-K experts); All-to-All routes tokens; balance loss prevents popularity skew."

**Deep dive.** `41_mixture_of_experts/MOE_DEEP_DIVE.md`.

### 5.7 The full parallelism stack — putting it together

Modern frontier training combines all four axes. **3D parallelism** = data + tensor + pipeline. **4D** adds expert. Typical 70B-1T training config:

- **TP = 8** within node (NVLink bandwidth).
- **PP = 4-16** across nodes.
- **DP = 16-256** outermost dimension (one DDP group across the TP×PP chunks).
- **EP = 8-64** for MoE models.
- **FSDP / ZeRO-1** layered on DP for sharded optimizer state.

**Communication cost analysis.** TP all-reduces happen most frequently → highest BW link. PP point-to-point activations are smaller and rarer → lower BW link. DP all-reduce of gradients is once per step → can use slower link. Engineers map these to NVLink (TP), NVSwitch (TP/PP), InfiniBand (DP across nodes).

---

## 6. Communication Primitives

Memorize these — every distributed-training interview asks about at least one.

| Primitive | What it does | Used for |
|---|---|---|
| **Broadcast** | One process → all others (same data). | Distributing weights at init. |
| **Scatter** | One → all (different chunks). | Splitting batch in DataParallel. |
| **Gather** | All → one (collect chunks). | Collecting per-replica metrics. |
| **Reduce** | All → one (sum / max / etc.). | Aggregate metric to chief. |
| **All-Gather** | All → all (everyone has all chunks). | Final step of TP column-parallel. Final step of ZeRO-3 forward. |
| **Reduce-Scatter** | All → all (each process keeps its share of the reduction). | First step of optimized All-Reduce. |
| **All-Reduce** | All → all (everyone has the reduction). | Gradient sync in DDP. Final TP row-parallel step. |
| **All-to-All** | All → all (each process sends a different chunk to each). | Expert routing in MoE. |

**Key identity.** `All-Reduce = Reduce-Scatter + All-Gather`. This decomposition is what makes Ring All-Reduce optimal.

**Ring All-Reduce.** Each of N GPUs sends/receives data to/from neighbors in a ring. Two phases of N-1 steps each. Total comm volume per GPU: `2 × (N − 1) × X / N ≈ 2X` (independent of N). The reason DDP scales to thousands of GPUs.

**NCCL.** NVIDIA's library implementing all these on GPUs with NVLink/InfiniBand awareness. The default backend.

**Hook.** "All-Reduce = Reduce-Scatter + All-Gather; Ring All-Reduce is `2(N-1)X/N` per GPU."

---

## 7. Putting it together — recipe for a 70B+ training run

The interview question: *"Walk me through how you'd train a 70B model from scratch."*

**Hardware.** 64-512 H100 GPUs, NVLink within node (8 GPUs), InfiniBand across nodes.

**Parallelism.**
- TP = 8 (within node, exploits NVLink).
- PP = 4 (across 4 nodes per pipeline group).
- FSDP / ZeRO-3 = the rest of the cluster as data parallel.
- 3D parallelism: TP × PP × DP = world size.

**Numerics.**
- BF16 forward/backward.
- FP32 master weights and optimizer state.
- FlashAttention-3 for attention (FP8 on Hopper; mixed BF16/FP8 on Blackwell).

**Activation memory.**
- Activation checkpointing (selective: skip cheap ops).
- Sequence packing.

**Optimizer.**
- AdamW with cosine schedule + warmup.
- Loss scaling not needed (BF16).
- Gradient clipping at 1.0.

**Throughput tricks.**
- Overlap gradient all-reduce with backward compute.
- Overlap pipeline communication with compute (DualPipe-style).
- Selective recomputation (skip cheap ops).
- FP8 for the GEMMs (Blackwell).

**Reliability.**
- Frequent checkpointing (every ~30 min).
- Async checkpoint to remote storage.
- Slow-worker / dead-worker detection and replacement.
- Loss spike monitoring with auto-restart from last good checkpoint.

**At inference.**
- KV cache with PagedAttention (vLLM).
- Continuous batching.
- Speculative decoding with a 1-2B draft model.
- INT8 or FP8 weights.
- Prefix caching for chat workloads.
- Tensor parallel across 4-8 GPUs for serving.

That's the full senior answer.

---

## 8. Cross-reference map

Where each topic is covered in detail elsewhere in this repo:

| Topic | Detailed file |
|---|---|
| Flash Attention | `05_attention_mechanisms/ATTENTION_DEEP_DIVE.md` |
| MQA / GQA / MLA | `05_attention_mechanisms/ATTENTION_DEEP_DIVE.md` |
| Activation checkpointing | `62_frontier_training_playbook/` |
| Efficient transformers / SSMs | `42_state_space_models/SSM_DEEP_DIVE.md` |
| KV cache | `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` |
| Stateful caching | `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` |
| Speculative decoding | `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` |
| Quantization | `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` |
| Mixed precision | `10_optimizers/`, `62_frontier_training_playbook/` |
| ZeRO / FSDP / DDP | `61_large_scale_llm_systems/` |
| Pipeline parallelism (GPipe / 1F1B / Zero Bubble / DualPipe) | `61_large_scale_llm_systems/`, `62_frontier_training_playbook/` |
| Tensor parallelism (Megatron) | `61_large_scale_llm_systems/` |
| Context / Ring attention | `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` |
| MoE | `41_mixture_of_experts/MOE_DEEP_DIVE.md` |
| PagedAttention | `63_paged_attention_and_llm_serving/` |
| Frontier 2024-2026 (FP8/FP4, MLA, DualPipe) | `06_llm_inference/LLM_INFERENCE_DEEP_DIVE.md` §14 |

---

## 9. Interview pitch ladder

For each topic, three answer lengths.

### Flash Attention

- **30 sec.** "Standard attention is quadratic memory because of the n×n attention matrix. Flash Attention tiles Q/K/V into SRAM-fitting blocks, processes attention block-by-block using online softmax, and on backward recomputes from saved norm factors. Linear memory, ~2-4× speedup."
- **2 min.** Add: tiling math, online softmax recurrence, why HBM bandwidth was the bottleneck, FlashAttention-2/3 improvements (parallelism, FP8).
- **5 min.** Add: implementation details (CUDA kernel structure), comparison with sparse / linearized attention, Hopper async pipelining.

### ZeRO

- **30 sec.** "Plain DDP replicates everything across GPUs. Most memory is optimizer state — Adam needs ~12 bytes/param at FP32. ZeRO-1 shards optimizer state across DP workers (4× savings), ZeRO-2 also shards gradients (8×), ZeRO-3 also shards parameters (linear in DP degree, +50% comm). FSDP is PyTorch's ZeRO-3."
- **2 min.** Add: what gets gathered when (forward param all-gather, backward grad reduce-scatter), why you don't always use ZeRO-3 (comm cost), interaction with TP/PP.
- **5 min.** Add: hybrid ZeRO (which layers ZeRO-3 on, others ZeRO-1), CPU offload variants, NVMe offload for inference.

### Speculative Decoding

- **30 sec.** "Decode is bandwidth-bound, not compute-bound. A small draft model proposes K tokens; the big target model verifies all K in a single forward pass. Accept the longest correct prefix. With α≈0.7 acceptance and K=4, ~2.5× speedup."
- **2 min.** Add: math of expected speedup, draft selection (small same-family LM), self-speculative variants (Medusa, EAGLE).
- **5 min.** Add: production tradeoffs (memory for draft model, KV cache duplication, batching interaction).

### Pipeline Parallelism

- **30 sec.** "Split layers across GPUs. Naive has bubbles (only one GPU works). GPipe splits batch into micro-batches and pipelines them, bubble shrinks to (d-1)/(m+d-1). 1F1B (PipeDream) interleaves forward/backward to start backward earlier. Zero Bubble splits backward into B-input and B-weight to fill remaining bubbles. DualPipe (DeepSeek-V3) feeds from both ends simultaneously."
- **2 min.** Add: weight stashing + version consistency, memory imbalance handling (Llama 3 last-stage trick), interaction with TP within stage.
- **5 min.** Add: full schedule diagram, comm overlap with compute, embedding/loss layer placement.

(Same drill for MoE, KV cache, Tensor Parallelism, Quantization.)

---

## 10. Interview Grill — 70 questions

### Memory (Q1–14)

1. Why is attention $O(n^2)$ in memory?
2. What problem does Flash Attention solve and how?
3. Difference between tiling and recomputation in Flash Attention?
4. What is online softmax and why does it matter?
5. Compare FlashAttention vs FlashAttention-2 vs FlashAttention-3.
6. What does MQA do? Quality cost?
7. What's GQA and when do you choose it over MQA?
8. What's MLA (Multi-head Latent Attention) and which model uses it?
9. Activation checkpointing — what's the compute/memory tradeoff?
10. What's selective activation checkpointing?
11. Why is KV cache memory often larger than weight memory at long context?
12. Three ways to shrink KV cache.
13. PagedAttention — what does it solve?
14. What's KV cache quantization and how aggressive can you go?

### Compute (Q15–22)

15. Why does sequence packing improve throughput?
16. What's a document mask?
17. Compare BigBird, Longformer, Linformer, LongNet.
18. Why might SSMs (Mamba) replace attention for some workloads?
19. What's the compute-bound vs bandwidth-bound regime in inference?
20. Prefill vs decode — which is bandwidth-bound?
21. Why does continuous batching help?
22. What's chunked prefill?

### Inference (Q23–34)

23. Walk through KV caching step-by-step.
24. What's stateful prefix caching?
25. Sketch speculative decoding.
26. What's the speedup formula for speculative decoding?
27. Compare Medusa, EAGLE, vanilla speculative decoding.
28. PTQ vs QAT — when each?
29. What's the straight-through estimator?
30. Compare GPTQ, AWQ, SmoothQuant.
31. What's mixed-precision quantization?
32. Why do INT4 weights but BF16 activations work?
33. What's NF4?
34. FP8 vs FP16 vs BF16 — which for what?

### Training — mixed precision (Q35–38)

35. BF16 vs FP16 — what's different and why does it matter for training stability?
36. What's loss scaling and when do you need it?
37. What are master weights?
38. Where in the training loop do you keep FP32?

### Training — data parallelism (Q39–46)

39. Why is DataParallel inferior to DDP?
40. Walk through Ring All-Reduce.
41. What's the per-GPU comm volume of Ring All-Reduce as a function of model size?
42. What's the difference between BSP and ASP?
43. Why does ZeRO-1 always make sense?
44. ZeRO-2 vs ZeRO-3 — when to use which?
45. What's FSDP?
46. Why does ZeRO-3 cost +50% comm?

### Training — pipeline parallelism (Q47–52)

47. What's the GPipe bubble formula?
48. Walk through 1F1B.
49. What's weight stashing in PipeDream?
50. Walk through Zero Bubble.
51. What does DualPipe do differently?
52. Why does Llama 3 reduce one transformer layer from first and last stages?

### Training — tensor parallelism (Q53–58)

53. Column-wise vs row-wise tensor parallelism — communication primitive each ends with?
54. How does Megatron-LM combine column + row to minimize all-reduces?
55. Why is TP usually limited to within a single node?
56. What's sequence parallelism and how does it extend TP?
57. What's Ring Attention?
58. Compare context parallelism vs tensor parallelism.

### Training — MoE (Q59–66)

59. What's a Mixture of Experts?
60. Top-1 vs Top-2 routing — tradeoffs?
61. What's expert-choice routing?
62. What's the load-balancing problem?
63. What's auxiliary load-balance loss?
64. What's auxiliary-free load balancing (DeepSeek-V3)?
65. What's capacity factor and priority dropping?
66. What's All-to-All comm and why is it the MoE bottleneck?

### Communication primitives (Q67–70)

67. List 8 standard collective primitives.
68. What's the All-Reduce = Reduce-Scatter + All-Gather identity?
69. Why does Ring All-Reduce scale to thousands of GPUs?
70. What's NCCL?

---

## 11. Drill plan

- **Day 1–2:** Read sections 1–4 (memory + compute + inference). Quiz yourself on Q1–34.
- **Day 3–4:** Read section 5 (training parallelism). Quiz Q35–66.
- **Day 5:** Read sections 6–7 (comm primitives + recipe). Quiz Q67–70.
- **Day 6–7:** Memorize the 30-second pitches in §9. Practice the "design a 70B training run" answer.
- **Recall test.** Pick three random topics; write the 30-second pitch from memory.

---

## Acknowledgement

This chapter was sparked by Gauri Gupta's interview-prep notes (NeoSigma, 2025), shared via X. The structure of memory → compute → inference → training-parallelism → comm follows her organization; the depth, cross-references, and interview-grill format are this repo's additions.

Single sentence to remember: **scaling = pick what runs out first; combine 4 axes of parallelism (data, tensor, pipeline, expert); shrink memory with FlashAttn + GQA + checkpointing + ZeRO; speed up inference with KV cache + paged attention + speculative decoding + quantization.**
