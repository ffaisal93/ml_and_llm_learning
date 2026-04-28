# Large-Scale LLM Systems — Deep Dive

> Frontier-lab interview prep. Pair with `INTERVIEW_GRILL.md`.

This is the systems content that separates "I trained a model in a notebook" from "I trained Llama-70B on 1000 GPUs." Frontier labs and big-tech infra teams probe this hard. The interview test isn't whether you've done it — it's whether you understand the math that determines what *can* be done.

---

## 1. Memory math for training

For a model with $P$ parameters in mixed-precision (BF16/FP16) training with Adam:

| Component | Memory (bytes per param) | Total |
|---|---|---|
| Weights (BF16) | 2 | $2P$ |
| Gradients (BF16) | 2 | $2P$ |
| Optimizer state — momentum (FP32) | 4 | $4P$ |
| Optimizer state — variance (FP32) | 4 | $4P$ |
| Master weights (FP32, for stability) | 4 | $4P$ |
| **Total** | **16** | **$16P$ bytes** |

Plus activations (depends on architecture, sequence length, micro-batch).

**Example: 70B params**
- Weights + grads + Adam states + master weights ≈ $16 \cdot 70 \times 10^9 = 1{,}120$ GB.
- Single H100 has 80 GB.
- Need to spread across many GPUs even before counting activations.

**Activation memory** for a transformer layer per token: $O(\mathrm{batch} \cdot \mathrm{seq} \cdot d_{\mathrm{model}})$ for the residual stream, plus larger intermediate tensors in attention/FFN. A 70B model with 8K context and batch 1 can have hundreds of GB of activations alone.

### Activation checkpointing

Recompute activations during backward instead of storing. Trades ~33% extra forward time for ~70% activation memory reduction. Standard for large model training.

### Mixed precision recap

- **Weights/grads in BF16** (or FP16 with loss scaling).
- **Master weights + optimizer state in FP32** for numerical stability.
- **BF16 vs FP16**: BF16 has same exponent range as FP32 (no overflow); FP16 needs loss scaling. BF16 is the modern default.
- **FP8** training is the new frontier (Hopper/Blackwell). Even more memory savings, requires more careful scale management.

---

## 2. Data parallelism

Each GPU gets a full copy of the model; different GPUs process different micro-batches; gradients all-reduced.

### Standard DP / DDP

Naive replication. Each rank holds full weights, full grads, full optimizer state. Limit: model must fit on a single GPU.

### ZeRO (Zero Redundancy Optimizer)

Microsoft DeepSpeed's idea: partition model state across DP ranks. Three stages:

- **ZeRO-1**: partition optimizer state. ~4× memory reduction.
- **ZeRO-2**: ZeRO-1 + partition gradients. ~8× reduction.
- **ZeRO-3**: ZeRO-2 + partition weights. ~$N$× reduction (where $N$ is DP world size).

ZeRO-3 = full sharding. PyTorch's FSDP (Fully Sharded Data Parallel) is essentially ZeRO-3.

### How FSDP/ZeRO-3 works in practice

- Forward: `all_gather` weights for the layer being computed; do compute; free the gathered weights.
- Backward: same for grads, plus `reduce_scatter` to send each shard back to its owner.
- Optimizer step: each rank updates only its shard.

Communication cost: $O(P)$ per step in total bytes — same big-O as DDP, but spread across more collective ops. Latency-sensitive.

### Communication patterns
- **all-reduce**: every rank ends up with the sum across all ranks. Bandwidth-bounded by ring algorithm: $2(N-1)/N \cdot P$ bytes per rank.
- **reduce-scatter**: like all-reduce, but each rank only gets its shard.
- **all-gather**: each rank shares its shard with everyone.

In ZeRO-3: an optimizer step costs ~3× the data of DDP per step (gather, gather, scatter), but allows training models that couldn't fit otherwise.

---

## 3. Tensor parallelism (TP)

Split individual matrix multiplications across GPUs. Megatron-LM's invention.

### Column / row parallelism

For $Y = XW$ where $W \in \mathbb{R}^{d \times h}$:
- **Column parallelism**: split $W$ along columns: $W = [W_1, W_2]$. Each GPU computes $XW_i$. Outputs gathered.
- **Row parallelism**: split $W$ along rows: $W = [W_1; W_2]$. Each GPU computes $X_i W_i$. Outputs summed.

### Megatron transformer pattern

For each transformer layer:
- **Attention**: $Q, K, V$ projections column-parallel (split heads across GPUs). Output projection row-parallel.
- **FFN**: first linear column-parallel; activation; second linear row-parallel.

Result: communication only at layer boundaries (one all-reduce per attention block, one per FFN block). Inside each, compute is local.

### TP scaling limits

- TP requires all-reduce per layer → very latency-sensitive.
- Best within a node (NVLink, ~600 GB/s on H100). Bad across nodes (Infiniband, ~50 GB/s).
- Typical: TP = 8 within a node, then DP/PP across nodes.

### Sequence parallelism

Extension of TP: in operations *not* parallelized in the matmul (LayerNorm, dropout), split along sequence dimension to save activation memory. Adds extra all-gather/scatter but reduces memory.

---

## 4. Pipeline parallelism (PP)

Split model across GPUs by *layer*. GPU 0 holds layers 0-7, GPU 1 holds 8-15, etc.

### Naive pipeline

Forward through all layers in order; backward through all layers. Most GPUs idle most of the time → "pipeline bubble."

### Microbatching (GPipe)

Split mini-batch into $m$ micro-batches; pipeline them. Each GPU processes one micro-batch at a time, hands forward to next stage. Reduces bubble to $\approx (P-1)/(P-1+m)$ where $P$ is number of stages.

### 1F1B (PipeDream / Megatron)

Interleave forward and backward of different micro-batches to keep all GPUs busy after warmup. Lower memory footprint than GPipe (don't store all micro-batch activations).

### PP cost
- Communication: send activations forward, gradients backward — small per-step.
- Bubble: needs many micro-batches to amortize. Trade-off: more micro-batches → less bubble, more memory.
- Imbalance: layers must split evenly in compute (uniform layer sizes).

---

## 5. 3D parallelism

Combine all three: TP + PP + DP. Standard for $\geq$ 100B-param training.

**Example: 175B model on 1024 GPUs**
- TP = 8 (within a node)
- PP = 16 (across nodes)
- DP = 8 (replicate the TP×PP setup 8 times)

Total = $8 \times 16 \times 8 = 1024$. Each rank holds $\frac{175 \mathrm{B}}{8 \times 16} = 1.4\mathrm{B}$ params, fits comfortably.

**Communication topology**: TP wants high-bandwidth (NVLink, intra-node). PP wants moderate bandwidth. DP wants low-frequency, large messages.

### Adding sequence parallelism / context parallelism

For very long contexts, parallelize *across sequence positions*:
- **Sequence parallel**: split input along sequence dim where TP doesn't help (LayerNorm, dropout).
- **Context parallel** (Ring Attention, Megatron's CP): split the *attention matrix* across GPUs along sequence; ring-pass keys/values.

Critical for million-token context.

---

## 6. Expert parallelism (for MoE)

Place different experts on different GPUs. Tokens routed via all-to-all.

For a 1T-param MoE with 64 experts:
- 64-way expert parallelism: each GPU holds one expert.
- Per token, only the chosen experts compute → activated params per token are small.

**Cost**: all-to-all communication twice per layer (dispatch tokens to experts, gather outputs back). All-to-all is the most expensive collective; scales poorly across nodes.

### Load balancing

If all tokens route to one expert, you have a bottleneck. **Auxiliary loss** penalizes imbalance. **Capacity factor** caps tokens per expert per batch (drop overflow). DeepSeek-V3 uses an auxiliary-loss-free balancing approach.

---

## 7. Compute / communication overlap

Modern training frameworks overlap compute with communication to hide latency.

- **DDP**: gradient all-reduce overlaps with backward of earlier layers.
- **FSDP**: weight all-gather for layer $\ell+1$ overlaps with compute of layer $\ell$.
- **TP**: hard to overlap due to dependencies inside a layer.
- **PP**: forward of micro-batch $k$ overlaps with backward of micro-batch $k-1$.

The fraction of time spent in communication vs compute is the key efficiency metric. Frontier labs target >50% MFU (Model FLOPs Utilization) — meaning compute is busy >50% of wall-clock time.

---

## 8. Common training failures at scale

### Loss spikes / divergence
- Cause: large gradient, small token group, optimizer state mismatch, BF16/FP8 numerical instability.
- Fixes: gradient clipping (typically 1.0), warmup (longer for large batch), reduce LR, restart from checkpoint, BF16 over FP16, occasional FP32 reductions.

### NaNs
- Cause: overflow in attention softmax (FP16 specifically), division by zero in normalization, numerical underflow in LayerNorm denominator $\sqrt{\sigma^2 + \epsilon}$ when variance is tiny in low precision.
- Fixes: BF16, attention computed in higher precision, increase $\epsilon$ in LayerNorm.

### Hangs
- Cause: collective op deadlock (one rank waiting at NCCL while another is hung in CPU code), ECC errors on a single GPU.
- Fixes: NCCL timeout settings; per-rank watchdog; sticky bad-GPU detection.

### Stragglers
- Cause: one slow GPU holding up the whole job.
- Fixes: run health checks before training; redundant replicas; fault tolerance with checkpointing.

### Checkpoint failures
- Cause: writing 100s of GB to network FS during a fragile time window.
- Fixes: async/local checkpointing; multi-tier storage; resharding for restart on different topology.

---

## 9. Common interview gotchas

| Question | Common wrong answer | Right answer |
|---|---|---|
| Memory for 70B model? | "A lot" | $16P$ for state ≈ 1.1 TB; plus activations |
| FP16 vs BF16? | Same | BF16 has FP32's exponent range — no need for loss scaling |
| Why is FSDP good? | Just shards | Shards weights+grads+opt state; recovers DDP-equivalent training with $N$× memory savings |
| Why TP only inside a node? | Tradition | All-reduce per layer is latency-sensitive; needs NVLink-class bandwidth |
| Pipeline bubble — fix? | None | Microbatching ($m$ micro-batches): bubble $\approx (P-1)/(P-1+m)$ |
| MoE comm cost? | Same as dense | All-to-all twice per layer; usually the bottleneck |
| MFU > 100%? | Possible | No — FLOPs utilization, capped at 1 |

---

## 10. Eight most-asked interview questions

1. **Compute the memory needed to train a 70B model with Adam in BF16.** ($16P$ for state ≈ 1.1 TB; plus activations.)
2. **What's the difference between ZeRO-1, 2, 3?** (Partitions opt state, +grads, +weights respectively.)
3. **Walk through tensor parallelism for a transformer FFN.** (Column-parallel first linear, row-parallel second; one all-reduce per FFN block.)
4. **Why is TP usually limited to within a node?** (Latency-sensitive all-reduce; needs NVLink bandwidth.)
5. **Pipeline bubble — what is it and how do you reduce it?** ($m$ micro-batches → bubble fraction $\to 0$ as $m$ grows.)
6. **Combine TP + PP + DP — when do you use each?** (TP intra-node, PP across few nodes, DP across many; product = total GPUs.)
7. **Communication patterns: all-reduce, all-gather, reduce-scatter — when each?** (DP grad sync, FSDP weight gather, FSDP grad scatter respectively.)
8. **Why activation checkpointing?** (Activations dwarf weights for long contexts; recompute saves ~70% memory at ~33% extra compute.)

---

## 11. Drill plan

- Compute training memory for: 7B, 13B, 70B, 175B. With + without activation checkpointing.
- Recite: column vs row parallelism; pipeline bubble formula; ZeRO stages.
- Sketch the all-to-all pattern for MoE expert parallelism.
- For each common training failure (loss spike, NaN, hang, straggler, checkpoint), recite cause + fix.
- Be able to talk through: "design a system to train Llama-70B on 1024 H100s." 5 minutes.

---

## 12. Further reading

- Rajbhandari et al. (2020), *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.*
- Shoeybi et al. (2019), *Megatron-LM: Training Multi-Billion Parameter Language Models.*
- Narayanan et al. (2021), *Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM.* — 3D parallelism.
- Korthikanti et al. (2022), *Reducing Activation Recomputation in Large Transformer Models.* — sequence parallelism.
- Liu et al. (2023), *Ring Attention with Blockwise Transformers for Near-Infinite Context.*
- Chowdhery et al. (2022), *PaLM* — frontier system at the time.
- Anthropic engineering blog — when published, contains relevant Constitutional AI / scaling content.
