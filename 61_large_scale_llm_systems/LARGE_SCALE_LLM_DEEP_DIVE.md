# Large-Scale LLM Systems — Deep Dive

> Frontier-lab interview prep. Pair with [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

This is the systems content that separates "I trained a model in a notebook" from "I trained Llama-70B on 1000 GPUs." Frontier labs and big-tech infra teams probe this hard. The interview test isn't whether you've done it — it's whether you understand the math that determines what *can* be done.

---

## 1. Memory math for training

**In plain language.** This section is doing one calculation: how many bytes does it take to hold a model *while you're training it*, as opposed to just running it. The answer is much bigger than people expect, because the optimizer keeps several extra copies of every parameter, and the table below is just an inventory of those copies.

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

> **Saying it out loud.** Sixteen bytes per parameter is the number that governs everything. Two for the BF16 weights, two for the BF16 gradients, and then twelve for Adam — momentum, variance and an FP32 master copy of the weights, all in full precision because that's where rounding errors actually hurt. So a 70B model needs about 1.1 terabytes of state before you've stored a single activation, which is fourteen H100s just to hold it still. And activations aren't a footnote: at 8K context a 70B model's activations run into the hundreds of gigabytes on their own, which is why checkpointing isn't optional at this scale.

### Activation checkpointing

Recompute activations during backward instead of storing. Trades ~33% extra forward time for ~70% activation memory reduction. Standard for large model training.

> **Saying it out loud.** Activation checkpointing buys memory with compute at a very favourable rate. Normally the forward pass keeps every intermediate tensor because backprop needs them; instead you keep a checkpoint every so often and recompute the rest on the way back. That's about 70% of your activation memory returned for roughly 33% more compute — worth it, because you were memory-bound, not compute-bound. The side effect worth naming is that it makes your MFU look worse than your HFU, since the recomputed FLOPs are work the hardware really did but the model doesn't benefit from twice.

### Mixed precision recap

- **Weights/grads in BF16** (or FP16 with loss scaling).
- **Master weights + optimizer state in FP32** for numerical stability.
- **BF16 vs FP16**: BF16 has same exponent range as FP32 (no overflow); FP16 needs loss scaling. BF16 is the modern default.
- **FP8** training is the new frontier (Hopper/Blackwell). Even more memory savings, requires more careful scale management.

> **Saying it out loud.** The whole mixed-precision setup is about exponent range, not precision. FP16 has five exponent bits, so small gradients underflow to zero and you have to scale the loss up before backward and back down before the step. BF16 keeps FP32's eight exponent bits and gives up mantissa instead, so it covers the same range and needs no loss scaling — which is why it's the default now. You still keep FP32 master weights and optimizer moments, because a tiny update added to a BF16 weight just rounds away and the model quietly stops learning. FP8 is the next step down, and it works, but it demands per-tensor scale management and a stale scale shows up as a loss spike out of nowhere.

---

## 2. Data parallelism

**In plain language.** Data parallelism is the simple one: every GPU has the entire model, they each chew through different training examples, and at the end of the step they average their gradients so the copies stay identical. Everything interesting in this section is about the waste in that word "identical" — if all N GPUs hold the same bytes, N minus 1 of those copies are dead weight.

Each GPU gets a full copy of the model; different GPUs process different micro-batches; gradients all-reduced.

### Standard DP / DDP

Naive replication. Each rank holds full weights, full grads, full optimizer state. Limit: model must fit on a single GPU.

> **Saying it out loud.** Plain DDP replicates absolutely everything — weights, gradients and optimizer state — on every GPU, and only the data differs. That's beautifully simple and it's why gradient sync is a single all-reduce per step, but it means the entire 16-bytes-per-parameter footprint has to fit on one card. On an 80-gigabyte H100 that's about 5 billion parameters, so anything bigger than a 7B model with real activations simply won't start. That's the failure mode: not slow, just an out-of-memory error before step one.

### ZeRO (Zero Redundancy Optimizer)

**In plain language.** ZeRO's observation is that data parallelism stores N identical copies of everything, so let each GPU own only its own 1/N and fetch the rest on demand. The three stages are just how far you're willing to take that: shard the optimizer's bookkeeping, then the gradients, then the weights themselves. Each stage frees more memory and, at the last one, costs more network traffic.

Microsoft DeepSpeed's idea: partition model state across DP ranks. Three stages:

- **ZeRO-1**: partition optimizer state. ~4× memory reduction.
- **ZeRO-2**: ZeRO-1 + partition gradients. ~8× reduction.
- **ZeRO-3**: ZeRO-2 + partition weights. ~$N$× reduction (where $N$ is DP world size).

ZeRO-3 = full sharding. PyTorch's FSDP (Fully Sharded Data Parallel) is essentially ZeRO-3.

> **Saying it out loud.** The three stages map to the three things you're storing. Stage 1 shards the optimizer state, which is 12 of your 16 bytes per parameter, so you get about 4x memory back — and crucially at zero extra communication cost, because the all-reduce you were already doing decomposes into exactly the reduce-scatter and all-gather that stage 1 needs. Stage 2 adds the gradients for about 8x, still free. Stage 3 adds the parameters, which scales your savings with the GPU count but costs roughly 50% more traffic, since every layer's weights now have to be gathered before forward and again on backward. So the rule is stage 1 always, stage 3 only when it's the difference between fitting and not.

### How FSDP/ZeRO-3 works in practice

- Forward: `all_gather` weights for the layer being computed; do compute; free the gathered weights.
- Backward: same for grads, plus `reduce_scatter` to send each shard back to its owner.
- Optimizer step: each rank updates only its shard.

Communication cost: $O(P)$ per step in total bytes — same big-O as DDP, but spread across more collective ops. Latency-sensitive.

> **Saying it out loud.** In practice FSDP works layer by layer: all-gather the shards of the layer you're about to run, do the compute, immediately free the gathered copy, move on. So your peak memory is your own shard of everything plus one full layer, rather than the whole model. The thing that makes it fast is prefetching — while layer L computes, the gather for layer L plus 1 is already in flight, so the extra communication hides behind the math. When that overlap fails, and it fails when layers are small or the interconnect is slow, FSDP turns into a sequence of stalls and MFU drops off a cliff. That's the tuning knob people actually fight with: how coarsely you wrap modules.

### Communication patterns
- **all-reduce**: every rank ends up with the sum across all ranks. Bandwidth-bounded by ring algorithm: $2(N-1)/N \cdot P$ bytes per rank.
- **reduce-scatter**: like all-reduce, but each rank only gets its shard.
- **all-gather**: each rank shares its shard with everyone.

In ZeRO-3: an optimizer step costs ~3× the data of DDP per step (gather, gather, scatter), but allows training models that couldn't fit otherwise.

> **Saying it out loud.** Three collectives cover most of distributed training, and you can derive them from each other. All-reduce means everybody ends up with the sum; all-gather means everybody ends up with everybody's data, no arithmetic; reduce-scatter means the data is summed but each rank keeps only its slice. The identity that matters is that all-reduce equals reduce-scatter followed by all-gather — which is exactly how Ring All-Reduce is implemented, and why each GPU only ever moves about twice the payload regardless of how many GPUs there are. That constant is why data parallelism scales to thousands of GPUs, and it's what lets ZeRO pull the two halves apart and use them independently.

---

## 3. Tensor parallelism (TP)

**In plain language.** Tensor parallelism cuts the model the other way from pipeline parallelism: rather than giving each GPU different layers, you give every GPU a slice of the *same* layer's weight matrix and they cooperate on each matrix multiply. There are exactly two ways to slice a matrix — by columns or by rows — and which one you pick decides what message the GPUs have to exchange when they're done.

Split individual matrix multiplications across GPUs. Megatron-LM's invention.

### Column / row parallelism

For $Y = XW$ where $W \in \mathbb{R}^{d \times h}$:
- **Column parallelism**: split $W$ along columns: $W = [W_1, W_2]$. Each GPU computes $XW_i$. Outputs gathered.
- **Row parallelism**: split $W$ along rows: $W = [W_1; W_2]$. Each GPU computes $X_i W_i$. Outputs summed.

> **Saying it out loud.** There are only two ways to cut a weight matrix and each ends in a different collective. Cut by columns and every GPU sees the full input but produces only some of the output features, so you finish with an all-gather to concatenate. Cut by rows and every GPU sees part of the input and produces a partial sum of the *whole* output, so you finish with an all-reduce to add them up. That's the entire vocabulary, and everything Megatron does is about arranging those two cuts so the communication lands in the cheapest possible place.

### Megatron transformer pattern

For each transformer layer:
- **Attention**: $Q, K, V$ projections column-parallel (split heads across GPUs). Output projection row-parallel.
- **FFN**: first linear column-parallel; activation; second linear row-parallel.

Result: communication only at layer boundaries (one all-reduce per attention block, one per FFN block). Inside each, compute is local.

> **Saying it out loud.** The Megatron pattern is column then row, and the elegance is that the intermediate never has to move. In the FFN, the first matrix is column-split so each GPU owns a disjoint set of hidden neurons; the nonlinearity is elementwise so it applies locally; the second matrix is row-split, consuming exactly those neurons and producing partial sums — one all-reduce, after the projection back down to d_model where the tensor is smallest. Attention is the same shape: Q, K, V column-split by head so each GPU runs whole heads independently, then the output projection row-split. Net result is two all-reduces per transformer layer, which is the number to quote.

### TP scaling limits

- TP requires all-reduce per layer → very latency-sensitive.
- Best within a node (NVLink 4, ~900 GB/s per GPU on H100). Bad across nodes (Infiniband, ~50 GB/s).
- Typical: TP = 8 within a node, then DP/PP across nodes.

> **Saying it out loud.** TP is pinned inside a node because those all-reduces sit on the critical path twice per layer and there's nothing to hide them behind. For an 80-layer model that's 160 blocking collectives per forward pass, and each one stalls until the slowest peer replies — so you need NVLink-class bandwidth, roughly 900 GB/s per GPU on H100, not the order-of-magnitude-less you get from InfiniBand between nodes. Push TP across nodes and MFU collapses without any change to the model. The other practical constraint is divisibility: the TP degree has to divide your head count and FFN width cleanly, which is another reason everyone lands on 8.

### Sequence parallelism

Extension of TP: in operations *not* parallelized in the matmul (LayerNorm, dropout), split along sequence dimension to save activation memory. Adds extra all-gather/scatter but reduces memory.

> **Saying it out loud.** Sequence parallelism is the cheap add-on that mops up what tensor parallelism left behind. Megatron's original scheme leaves the LayerNorm, dropout and residual adds fully replicated on every GPU — they're almost free in FLOPs, but their activations are the full batch-by-sequence-by-d tensor, so replication wastes real memory. Splitting those along the sequence dimension recovers it, at the cost of an all-gather going into each TP region and a reduce-scatter coming out. Just be careful with the name: this is not context parallelism, which shards the tokens for attention itself.

---

## 4. Pipeline parallelism (PP)

**In plain language.** Pipeline parallelism cuts the model by depth, like stations on an assembly line — GPU 0 runs the first ten layers, hands the result to GPU 1, and so on. The upside is that only a small activation tensor crosses between stages, so it works over slow links; the downside is that an assembly line with one item on it leaves everyone but one station idle, and that idle time has a name: the bubble.

Split model across GPUs by *layer*. GPU 0 holds layers 0-7, GPU 1 holds 8-15, etc.

### Naive pipeline

Forward through all layers in order; backward through all layers. Most GPUs idle most of the time → "pipeline bubble."

### Microbatching (GPipe)

Split mini-batch into $m$ micro-batches; pipeline them. Each GPU processes one micro-batch at a time, hands forward to next stage. Reduces bubble to $\approx (P-1)/(P-1+m)$ where $P$ is number of stages.

> **Saying it out loud.** Micro-batching is what turns a serial handoff into a real pipeline. Without it, stage 0 does the whole batch and then waits out the entire rest of the forward and backward; with it, stage 0 finishes micro-batch one, passes it on, and immediately starts micro-batch two, so all the stages overlap in time. The bubble fraction is P minus 1 over P minus 1 plus m, so with 4 stages and 8 micro-batches you're still wasting 27% of the cluster, but at 32 you're down to 9% — the rule of thumb is m at least 4 times P. The price is memory: GPipe holds the activations for every in-flight micro-batch until its backward runs.

### 1F1B (PipeDream / Megatron)

Interleave forward and backward of different micro-batches to keep all GPUs busy after warmup. Lower memory footprint than GPipe (don't store all micro-batch activations).

> **Saying it out loud.** 1F1B alternates one forward with one backward instead of doing all the forwards first. The bubble is exactly the same size as GPipe's — that's the part people get wrong — but the memory is much better, because each micro-batch's activations get freed as soon as its backward runs, so peak in-flight activations drops from m micro-batches to roughly the number of stages. The complication it introduces is weight versioning: a backward may run against weights that have since been updated, which is stale-gradient territory, and that's why PipeDream needs weight stashing or a periodic flush.

### PP cost
- Communication: send activations forward, gradients backward — small per-step.
- Bubble: needs many micro-batches to amortize. Trade-off: more micro-batches → less bubble, more memory.
- Imbalance: layers must split evenly in compute (uniform layer sizes).

> **Saying it out loud.** The reason pipeline is the axis you stretch across nodes is that it barely communicates: one hidden-state tensor per micro-batch boundary, point-to-point, not a collective — orders of magnitude less than tensor parallelism's per-layer all-reduce. What you pay instead is bubble and balance. More micro-batches shrink the bubble but cost activation memory, and every stage has to take about the same time, because a pipeline runs at the speed of its slowest stage. The classic imbalance is at the ends: stage 0 also carries the embedding and the last stage carries the output projection and the loss, which for a big vocabulary is genuinely expensive.

---

## 5. 3D parallelism

Combine all three: TP + PP + DP. Standard for $\geq$ 100B-param training.

**Example: 175B model on 1024 GPUs**
- TP = 8 (within a node)
- PP = 16 (across nodes)
- DP = 8 (replicate the TP×PP setup 8 times)

Total = $8 \times 16 \times 8 = 1024$. Each rank holds $\frac{175 \mathrm{B}}{8 \times 16} = 1.4\mathrm{B}$ params, fits comfortably.

**Communication topology**: TP wants high-bandwidth (NVLink, intra-node). PP wants moderate bandwidth. DP wants low-frequency, large messages.

> **Saying it out loud.** 3D parallelism is really about matching each axis's communication appetite to the right link. TP talks twice per layer, so it goes inside the node on NVLink — that's why it's 8. PP sends one small activation per micro-batch, so it tolerates InfiniBand across nodes. DP syncs once per optimizer step, big message but infrequent, so it goes outermost. And the numbers multiply: 8 times 16 times 8 is 1024 GPUs, with each rank holding 175 billion over 128, about 1.4 billion parameters, which is a comfortable 22 gigabytes of Adam state. Get the mapping backwards — TP across nodes — and you can halve your MFU without touching the model.

### Adding sequence parallelism / context parallelism

For very long contexts, parallelize *across sequence positions*:
- **Sequence parallel**: split input along sequence dim where TP doesn't help (LayerNorm, dropout).
- **Context parallel** (Ring Attention, Megatron's CP): split the *attention matrix* across GPUs along sequence; ring-pass keys/values.

Critical for million-token context.

> **Saying it out loud.** Context parallelism is the fourth axis, and it's the only one that helps when a *single example* is too long. TP shrinks the model per GPU and PP shrinks the layers per GPU, but neither touches the fact that activation memory scales with sequence length on every device — at a million tokens that's what kills you. So you shard the tokens themselves, which is fine for the MLP because it's per-token, and for attention you circulate the KV blocks around a ring, accumulating with online softmax so the result is still exact. The tradeoff is a ring of communication overlapped with compute: if the interconnect can't keep up, every GPU sits waiting for the next KV block.

---

## 6. Expert parallelism (for MoE)

Place different experts on different GPUs. Tokens routed via all-to-all.

For a 1T-param MoE with 64 experts:
- 64-way expert parallelism: each GPU holds one expert.
- Per token, only the chosen experts compute → activated params per token are small.

**Cost**: all-to-all communication twice per layer (dispatch tokens to experts, gather outputs back). All-to-all is the most expensive collective; scales poorly across nodes.

> **Saying it out loud.** MoE lets you buy parameters without buying compute per token. You replace the one dense FFN with, say, 64 experts on 64 GPUs, and a small router sends each token to only one or two of them — so a trillion-parameter model might activate only tens of billions per token. What you've done, though, is turn a local matmul into a network operation: every layer needs two all-to-alls, one to ship tokens to their experts and one to bring the results back. All-to-all is the worst-scaling collective there is — every rank sends a different message to every other rank, with no ring or tree trick available — so on MoE models the network, not the GPU, is usually the bottleneck.

### Load balancing

If all tokens route to one expert, you have a bottleneck. **Auxiliary loss** penalizes imbalance. **Capacity factor** caps tokens per expert per batch (drop overflow). DeepSeek-V3 uses an auxiliary-loss-free balancing approach.

> **Saying it out loud.** Load balancing is the hard part of MoE because routing is a popularity contest by nature — nothing stops the router from sending most tokens to a handful of experts, and then those GPUs are the whole pipeline's critical path while the rest idle. The classic fix is an auxiliary loss that penalises uneven routing, but it's a second objective fighting your language-modelling loss and its coefficient needs tuning. The capacity factor is the hard backstop: cap how many tokens an expert will take, and drop the overflow — which means a badly balanced model is silently skipping computation on some tokens, and that's the failure mode to name. DeepSeek-V3's trick avoids the extra loss entirely by nudging a per-expert routing bias down when an expert is overloaded and up when it's starved.

---

## 7. Compute / communication overlap

Modern training frameworks overlap compute with communication to hide latency.

- **DDP**: gradient all-reduce overlaps with backward of earlier layers.
- **FSDP**: weight all-gather for layer $\ell+1$ overlaps with compute of layer $\ell$.
- **TP**: hard to overlap due to dependencies inside a layer.
- **PP**: forward of micro-batch $k$ overlaps with backward of micro-batch $k-1$.

The fraction of time spent in communication vs compute is the key efficiency metric. Frontier labs target >50% MFU (Model FLOPs Utilization) — meaning compute is busy >50% of wall-clock time.

> **Saying it out loud.** Overlap is what separates a 25% MFU run from a 50% one. The pattern is always the same: start the communication as early as you possibly can and keep the GPU doing math while it's in flight. DDP buckets gradients and fires the all-reduce for a layer while the backward of earlier layers is still running; FSDP prefetches the next layer's weight gather during the current layer's compute; pipeline overlaps micro-batch k's forward with micro-batch k minus 1's backward. Tensor parallelism is the one that resists, because the all-reduce is a true data dependency inside the layer — which is exactly why TP has to live on the fastest link. And the number to quote is MFU: frontier runs target above 50%, and MFU is capped at 1 by construction, so anyone reporting more has used the wrong peak.

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

> **Saying it out loud.** The thing that surprises people is that at 1000 GPUs, failure is the normal state, not the exception, and each failure mode has a signature. Loss spikes look like numerics meeting a bad batch — fix with gradient clipping at 1.0, longer warmup, BF16 over FP16, and a rollback if it doesn't recover. NaNs in FP16 almost always come from attention softmax overflow, since logits above about 65,000 become infinity. Hangs are the nasty one: NCCL collectives are barriers, so one dead rank blocks everyone silently and burns money until the timeout fires. Stragglers are worse than they sound — one card running 10% slow taxes the entire cluster 10%, because synchronous SGD waits for everyone. And checkpointing is the insurance policy that makes the rest survivable, which is why it's asynchronous, sharded, and written locally before being uploaded.

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

> **Saying it out loud.** This table is really a list of the places people give an answer that sounds right and isn't. The memory one is the biggest: don't say "a lot," say 16 bytes a parameter, so 1.1 terabytes for 70B, plus activations. Don't say FP16 and BF16 are the same — BF16 trades mantissa for FP32's exponent range, which is why it doesn't need loss scaling. And the MFU one is a nice trap: MFU is a ratio of achieved to peak FLOPs, so it's capped at 1, and if you ever compute more than 100% you've used the sparsity peak or the wrong FLOP formula. In every case the pattern is the same — give the number and the mechanism, not the adjective.

---

## 10. Eight most-asked interview questions

1. **Compute the memory needed to train a 70B model with Adam in BF16.** ($16P$ for state ≈ 1.1 TB; plus activations.)

> **Saying it out loud.** Sixteen bytes a parameter, so about 1.1 terabytes — and I'd say the breakdown out loud so they know it's derived, not memorised: 2 for BF16 weights, 2 for BF16 gradients, 4 each for Adam's momentum and variance in FP32, and 4 for the FP32 master weights. Then I'd immediately flag that this is state only. Activations at 8K context add hundreds of gigabytes more, which is why the practical answer to "how many H100s" is not fourteen but sixty-four.
2. **What's the difference between ZeRO-1, 2, 3?** (Partitions opt state, +grads, +weights respectively.)

> **Saying it out loud.** Stage 1 shards the optimizer state, stage 2 also shards the gradients, stage 3 also shards the parameters — and each one includes the one before it. Roughly 4x, 8x, and then linear in your data-parallel degree. The part that makes the answer sound senior is the communication story: stages 1 and 2 are free, because the all-reduce you were doing anyway decomposes into reduce-scatter plus all-gather, whereas stage 3 costs about 50% more traffic since weights must be gathered before every forward and again on backward. So: stage 1 always, stage 3 when you'd otherwise not fit.
3. **Walk through tensor parallelism for a transformer FFN.** (Column-parallel first linear, row-parallel second; one all-reduce per FFN block.)

> **Saying it out loud.** Column-parallel the first linear, row-parallel the second, and you pay exactly one all-reduce per FFN block. The reason that ordering works is that the column split gives each GPU a disjoint set of hidden neurons, the activation function is elementwise so it needs no communication, and the row split then consumes exactly those neurons and produces partial sums to add up. And the all-reduce lands after the projection back down to d_model, which is where the tensor is four times smaller than the hidden layer — communicating at the narrow point is the whole design principle.
4. **Why is TP usually limited to within a node?** (Latency-sensitive all-reduce; needs NVLink bandwidth.)

> **Saying it out loud.** Because the all-reduce happens twice per layer, on the critical path, with nothing to hide it behind. In an 80-layer model that's 160 blocking collectives per forward, and each one waits on the slowest peer — so you need NVLink-class bandwidth, roughly 900 GB/s per GPU on H100, rather than the roughly 50 gigabytes a second you'd get from InfiniBand between nodes. That's an order of magnitude, and it shows up directly as MFU. So TP goes inside the node, typically 8-wide, and every other axis goes outside it.
5. **Pipeline bubble — what is it and how do you reduce it?** ($m$ micro-batches → bubble fraction $\to 0$ as $m$ grows.)

> **Saying it out loud.** The bubble is the idle time while the pipeline fills and drains — during warmup only the first stage has work, during cooldown only the last. You reduce it by splitting the batch into more micro-batches so the stages overlap: the fraction is P minus 1 over P minus 1 plus m, so at 4 stages you go from 27% waste at 8 micro-batches to 9% at 32. The tradeoff is that in-flight micro-batches cost activation memory, which is why 1F1B exists — same bubble, but it runs each backward as soon as it can so activations get freed earlier. Zero Bubble goes further by splitting the backward into the ordered input-gradient half and the deferrable weight-gradient half.
6. **Combine TP + PP + DP — when do you use each?** (TP intra-node, PP across few nodes, DP across many; product = total GPUs.)

> **Saying it out loud.** I'd answer this by interconnect, because that's what actually decides it. TP inside the node where NVLink is, because it all-reduces twice per layer. PP across a handful of nodes, because it only sends a small activation per micro-batch and tolerates a slower link. DP outermost across the whole cluster, because it syncs once per optimizer step. Their product is your world size — 8 times 16 times 8 is 1024 — and you layer ZeRO-1 on the DP dimension for free memory. The failure mode when you get the ordering wrong is dramatic and invisible in the code: your model trains correctly and your MFU is half what it should be.
7. **Communication patterns: all-reduce, all-gather, reduce-scatter — when each?** (DP grad sync, FSDP weight gather, FSDP grad scatter respectively.)

> **Saying it out loud.** All-reduce is for gradient sync in data parallelism and for the row-parallel step in tensor parallelism — everyone needs the same summed result. All-gather is for collecting shards when everyone needs the whole thing, which is FSDP fetching weights before a layer and the column-parallel concatenation. Reduce-scatter is for when the sum needs to be split back up, which is FSDP sending each gradient shard home to its owner. And the identity to close on is that all-reduce equals reduce-scatter plus all-gather — which is exactly how Ring All-Reduce is implemented, and why per-GPU traffic is about 2P bytes no matter how many GPUs you have.
8. **Why activation checkpointing?** (Activations dwarf weights for long contexts; recompute saves ~70% memory at ~33% extra compute.)

> **Saying it out loud.** Because activations dwarf everything else at long context, and unlike weights you can regenerate them. A 70B model at 8K context can carry hundreds of gigabytes of activations, so you throw most of them away and recompute from checkpoints on the backward pass — roughly 70% of that memory back for about 33% extra compute. The reason that trade is good is that you were memory-bound, not compute-bound, so the spare FLOPs were going to waste anyway. Modern practice is selective: don't recompute LayerNorm, it's cheap to store; do recompute the attention and MLP intermediates, which are where the bytes are.

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
