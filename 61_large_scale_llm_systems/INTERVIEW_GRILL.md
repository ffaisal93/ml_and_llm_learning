# Large-Scale LLM Systems — Interview Grill

> 50 questions on memory math, sharding, parallelism, communication, training failures. Drill until you can answer 35+ cold.

---

## A. Memory math

**1. For a model with $P$ params trained with Adam in mixed precision, total state memory?**
$\sim 16P$ bytes: 2 (weights BF16) + 2 (grads BF16) + 4 (momentum FP32) + 4 (variance FP32) + 4 (master FP32).

> **Saying it out loud.** Sixteen bytes per parameter is the number to say, and then break it down so they know you're not reciting. Two bytes for the BF16 weights and two for the BF16 gradients — that's the part people expect. The other twelve is Adam's bookkeeping: four for momentum, four for variance, and four for the FP32 master copy of the weights, all in full precision because the optimizer is where numerical sloppiness kills you. So the punchline is that the optimizer state is three-quarters of your training memory, which is exactly why ZeRO starts by sharding it.

**2. 70B params — total state memory?**
$16 \cdot 70 \times 10^9 \approx 1.1$ TB.

> **Saying it out loud.** About 1.1 terabytes, and I'd get there out loud: 16 bytes a parameter times 70 billion is roughly 1.1 times 10 to the 12. The thing to flag immediately is that this is *state only* — no activations, no fragmentation, no communication buffers — so it's a floor, not an estimate. In practice activations at long context can rival it, which is why activation checkpointing is basically mandatory at this scale.

**3. H100 memory?**
80 GB (HBM3); H200/B200 increase this further.

> **Saying it out loud.** An H100 SXM has 80 gigabytes of HBM3, and the number I'd pair with it is 3.35 terabytes per second of bandwidth, because for inference the bandwidth matters more than the capacity. The H200 is the same compute with 141 gigabytes at 4.8 terabytes a second — same chip, better memory — and Blackwell goes further again. If someone asks about FLOPs, H100 dense BF16 is about 989 teraflops; you'll see 1,979 quoted, but that's the with-sparsity number and quoting it unqualified is a tell.

**4. How many H100s minimum for 70B Adam state?**
$\sim 14$ for state alone (1.1 TB / 80 GB). In practice need many more for activations + comm overhead.

> **Saying it out loud.** Fourteen, and I'd show the division: 1.1 terabytes of state divided by 80 gigabytes a card. But the honest answer is that fourteen is a fantasy number — you also need activations, communication buffers, memory for fragmentation, and headroom so the allocator isn't thrashing. Realistically you're looking at 64 or more GPUs for a comfortable 70B run, and the constraint that actually binds is usually activation memory at your target sequence length, not the parameter state.

**5. Activation memory scaling?**
$O(\mathrm{batch} \times \mathrm{seq} \times d_{\mathrm{model}} \times L)$ for residual stream, plus larger inside attention/FFN. Long context blows this up.

> **Saying it out loud.** Activation memory scales with batch times sequence length times model width times depth — so it's linear in all four, and that's already bad because the KV and attention intermediates inside each layer are several times the residual stream. The reason it bites is that sequence length is the one people push hardest: doubling context doubles activation memory even with FlashAttention keeping attention itself linear. That's the failure mode — the run that fit yesterday at 4K OOMs today at 8K — and the fix is checkpointing plus sequence or context parallelism.

**6. Activation checkpointing — what does it do?**
Recompute activations during backward instead of storing. ~70% activation memory savings at ~33% extra compute.

> **Saying it out loud.** It trades compute for memory: you throw away most of the forward activations and recompute them during the backward pass from the nearest checkpoint. The exchange rate is roughly 70% of your activation memory back for about 33% more compute, which is a good deal because you were memory-bound, not compute-bound. The modern refinement is selective checkpointing — don't bother recomputing LayerNorm, it's cheap to keep, so spend the memory budget where the tensors are big. One thing to name: it makes MFU look worse than HFU, because the recomputed FLOPs are real work that isn't useful work.

**7. BF16 vs FP16?**
BF16 has FP32's exponent range → no overflow → no loss scaling needed. FP16 has more precision but limited range. BF16 default for modern training.

> **Saying it out loud.** BF16 wins on range, not precision — and that's the whole answer. FP16 has five exponent bits, so small gradients underflow to zero and you need loss scaling, where you multiply the loss by something like 2^16 before backward and divide it back out. BF16 has FP32's eight exponent bits and only seven mantissa bits, so it's actually *less* precise but covers the same dynamic range, which means no loss scaling and no overflow in the attention softmax. Fewer mantissa bits turns out not to matter for gradient descent, which is a noisy process anyway — that's why BF16 became the default.

**8. FP8 training — what's new?**
Hopper/Blackwell native. Even less memory + faster matmuls. Requires per-tensor scale management.

> **Saying it out loud.** FP8 halves the bytes again, so on Hopper and Blackwell you get both memory savings and roughly double matmul throughput. The catch is dynamic range: with four or five exponent bits there's no room for error, so you carry per-tensor — sometimes per-block — scaling factors and update them as training proceeds. The failure mode is a scale that goes stale and either saturates or flushes to zero, which shows up as a loss spike out of nowhere. DeepSeek-V3 is the existence proof that FP8 pretraining works at frontier scale, but people typically keep the sensitive pieces — the optimizer, the master weights, often the attention softmax — in higher precision.

---

## B. Data parallelism

**9. DDP — what gets replicated?**
Full weights, grads, optimizer state per rank. Limit: model must fit on single GPU.

> **Saying it out loud.** Everything, and that's the point of the question. Every rank holds a full copy of the weights, a full copy of the gradients, and a full copy of the optimizer state, and the only thing that differs is the slice of data it sees. So DDP's hard limit is that the whole model plus Adam state has to fit on a single GPU — 16 bytes a parameter means about 5 billion parameters on an 80-gig card. Above that you're not choosing ZeRO for speed, you're choosing it because DDP simply won't start.

**10. ZeRO-1?**
Partition optimizer state across DP ranks. ~4× memory savings.

> **Saying it out loud.** Stage 1 shards the optimizer state — momentum, variance and master weights — across the data-parallel ranks, so each GPU only stores and updates 1/N of it. That's about 4x memory back, since the optimizer is 12 of the 16 bytes per parameter. The reason it's a no-brainer is that the communication volume is identical to plain DDP: you were doing an all-reduce of gradients anyway, and it decomposes into reduce-scatter plus all-gather, which is exactly the pattern ZeRO-1 needs. Free memory, zero cost — always on.

**11. ZeRO-2?**
Partition optimizer state + gradients. ~8× savings.

> **Saying it out loud.** Stage 2 adds gradient sharding on top of stage 1, so each rank only ever holds the gradients for the parameters it owns. That gets you to roughly 8x memory savings, and again the communication volume is the same as DDP because you stop at the reduce-scatter instead of completing the all-reduce. The one thing you give up is gradient accumulation convenience — with sharded gradients you have to be careful about how you accumulate across micro-batches — but frameworks handle it.

**12. ZeRO-3?**
Partition optimizer state + grads + weights. $\sim N$× savings.

> **Saying it out loud.** Stage 3 also shards the parameters themselves, so no rank holds the full model at any time. Memory now scales with N, your data-parallel degree — 64 GPUs, roughly 64x savings — which is what makes models that couldn't otherwise fit trainable at all. The price is about 1.5x the communication of DDP, because you now have to all-gather each layer's weights before its forward, gather again for backward, and reduce-scatter the gradients. So the rule of thumb is stage 1 always, stage 3 only when you genuinely can't fit — you're paying network bandwidth for memory.

**13. FSDP = ?**
PyTorch's implementation of ZeRO-3. Fully Sharded Data Parallel.

> **Saying it out loud.** FSDP is PyTorch's native ZeRO-3 — same idea, different lineage: DeepSpeed built ZeRO, PyTorch productionised the stage-3 behaviour as Fully Sharded Data Parallel. It works by wrapping modules; before a wrapped unit runs it all-gathers its parameters, runs, then immediately frees them again. The practical knob is the wrapping granularity: wrap too coarsely and you gather too much at once and OOM, wrap too finely and you get lots of small collectives and lose the overlap. That's the tuning tradeoff people actually hit in production.

**14. ZeRO-3 forward pass?**
All-gather weights for current layer, compute, free. Repeat per layer.

> **Saying it out loud.** Layer by layer: before you compute layer L, you all-gather its parameter shards so every rank temporarily has the full weight matrix, you do the matmul, and then you immediately free the gathered copy. Peak memory is therefore your shard of everything plus one full layer, not the whole model. The performance trick that makes it viable is prefetching — you start the all-gather for layer L plus 1 while layer L is still computing, so the communication hides behind compute. If that overlap fails, FSDP degrades into a sequence of stalls and your MFU falls off a cliff.

**15. ZeRO-3 backward pass?**
All-gather weights, compute backward, reduce-scatter gradients to owners.

> **Saying it out loud.** Backward is the mirror image plus one extra step. You all-gather the layer's weights again — you threw them away in forward, so you either re-gather or you kept them, which is the memory-versus-bandwidth knob — compute the gradient, and then reduce-scatter so each rank ends up with the reduced gradient only for the shard it owns. Then each rank updates only its own slice of parameters with its own slice of optimizer state. That reduce-scatter instead of all-reduce is the key: it's the same bandwidth as half an all-reduce, and it delivers exactly the piece each rank needs.

**16. ZeRO-3 vs DDP communication?**
ZeRO-3 has ~1.5× the communication volume of DDP per step (Rajbhandari 2020): all-gather + reduce-scatter is bandwidth-equivalent to one all-reduce, plus an extra all-gather in forward. The trade-off buys $N\times$ memory savings, enabling models that wouldn't otherwise fit.

> **Saying it out loud.** About 1.5x DDP, and the derivation is worth saying because it shows you understand collectives. An all-reduce is a reduce-scatter followed by an all-gather, so DDP's gradient sync costs two units. ZeRO-3 does a reduce-scatter on gradients and an all-gather on weights in backward — that's the same two units — plus an extra all-gather of weights in the forward pass, so three units total. Fifty percent more traffic in exchange for N-times memory savings; whether that's a good trade depends entirely on whether your interconnect has headroom.

**17. all-reduce ring algorithm cost?**
$2(N-1)/N \cdot P$ bytes per rank — bandwidth-optimal.

> **Saying it out loud.** Two times N minus 1 over N, times the payload — which for large N is just about 2P per rank, and crucially it does *not* grow with the number of GPUs. The reason is the ring structure: each GPU only talks to its two neighbours, and the algorithm runs N minus 1 reduce-scatter steps followed by N minus 1 all-gather steps, each moving 1/N of the data. That's why it's bandwidth-optimal and why DDP scales to thousands of GPUs. What does grow with N is latency — 2 times N minus 1 hops — which is why very large rings get replaced by tree or hierarchical algorithms for small messages.

**18. all-gather vs reduce-scatter?**
all-gather: each rank ends up with everyone's data. reduce-scatter: each rank ends up with one shard of the sum.

> **Saying it out loud.** All-gather means everyone ends up holding everyone's data — no arithmetic, just concatenation, and the output is N times bigger than each input. Reduce-scatter does the arithmetic but splits the result: the values get summed across ranks, and then each rank keeps only its own slice of that sum. The reason to know both cold is the identity that all-reduce equals reduce-scatter plus all-gather, which is what Ring All-Reduce implements and what lets ZeRO peel the two halves apart and use them separately.

---

## C. Tensor parallelism

**19. Megatron column parallelism for $Y = XW$?**
Split $W$ along columns: $W = [W_1, W_2]$. Each GPU computes $X W_i$. Concatenate or all-gather outputs.

> **Saying it out loud.** Column parallelism cuts the weight matrix vertically — GPU 0 owns the first half of the output columns, GPU 1 the second half. Each GPU multiplies the *full* input X by its slice of W, so each produces a slice of the output, and you finish with an all-gather to stitch them together. The reason this is the right choice for the first MLP matrix and for Q/K/V is that the output slices are independent along the feature dimension, so a per-head or per-neuron nonlinearity can be applied locally before you ever communicate.

**20. Row parallelism?**
Split $W$ along rows: $W = [W_1; W_2]$. Each GPU computes $X_i W_i$. All-reduce to sum partial outputs.

> **Saying it out loud.** Row parallelism cuts horizontally: each GPU owns a slice of W's rows and a matching slice of the input's columns, so each computes a partial sum over the full output shape. Since every GPU has a piece of the same answer, you have to all-reduce to add them. That's why it pairs with column parallelism — the column split produces exactly the sharded input that row parallelism wants, so you can chain them without any communication in between and pay a single all-reduce at the end.

**21. Megatron transformer FFN parallelism?**
First linear column-parallel, second linear row-parallel. One all-reduce per FFN block.

> **Saying it out loud.** Column-parallel first, row-parallel second — and the elegance is that the intermediate never has to be communicated. The first linear expands to 4d and is column-split, so each GPU holds a disjoint set of hidden neurons; the GELU or SwiGLU is elementwise so it applies locally; then the second linear is row-split, consuming exactly those hidden neurons and producing partial sums. One all-reduce per FFN block, and it happens after the projection back down to d, which is where the tensor is smallest. That's the whole design principle: communicate as rarely and as narrowly as you can.

**22. Megatron attention parallelism?**
Q, K, V projections column-parallel (split by heads). Output projection row-parallel. One all-reduce per attention block.

> **Saying it out loud.** Same pattern, and the head structure makes it natural. Q, K and V are column-parallel split by head, so each GPU owns a complete subset of attention heads and can run the full softmax and the value multiply for its heads with no communication at all — attention is embarrassingly parallel across heads. Then the output projection is row-parallel, because each GPU has a slice of the concatenated head outputs, and one all-reduce sums the partial results. So a transformer layer costs exactly two all-reduces, one per attention and one per MLP.

**23. Why TP only intra-node?**
All-reduce per layer is latency-bound. Needs NVLink-class bandwidth (~600 GB/s). Cross-node Infiniband (~50 GB/s) too slow.

> **Saying it out loud.** Because those all-reduces sit on the critical path twice per layer, so it's latency you can't hide behind anything. In an 80-layer model that's 160 blocking collectives per forward pass, and every one of them stalls the GPU until the slowest peer replies. NVLink inside a node gives you hundreds of gigabytes a second — on H100 it's about 900 GB/s per GPU — while cross-node InfiniBand is an order of magnitude less. Push TP across nodes and your MFU collapses; that's why the universal answer is TP inside the node, everything else outside it.

**24. Typical TP degree?**
8 (matches 8 GPUs per node).

> **Saying it out loud.** Eight, because that's how many GPUs are on an NVLink island in a standard DGX or HGX node. It's not a deep algorithmic result — it's the hardware boundary. You'd go to TP 4 if you wanted two pipeline stages per node or your model's head count doesn't divide by 8, and the constraint to mention is that TP degree has to divide the number of attention heads and the FFN hidden dimension evenly, or you get ragged shards.

**25. Sequence parallelism — what does it parallelize?**
LayerNorm, dropout, residual ops along sequence dim. Reduces activation memory at cost of extra all-gather/scatter.

> **Saying it out loud.** It parallelises the leftovers — the LayerNorm, dropout and residual ops between the tensor-parallel regions, which Megatron's original scheme left fully replicated on every GPU. Those ops are cheap in FLOPs but their activations are the full batch-by-sequence-by-d tensor, so replicating them wastes real memory. Splitting them along the sequence dimension recovers that, at the cost of an extra all-gather going into the TP region and a reduce-scatter coming out. Note the name is overloaded — this is not context parallelism, which actually shards the tokens for attention itself.

---

## D. Pipeline parallelism

**26. PP basic idea?**
Split model by layer: GPU 0 has layers 0-7, GPU 1 has 8-15, etc. Pass activations forward, gradients backward.

> **Saying it out loud.** You cut the model by depth: each GPU owns a contiguous block of layers and passes activations to the next stage on the forward, gradients back on the reverse. The appeal is that the communication is tiny — just the hidden state at the stage boundary, not the weights — so unlike tensor parallelism it survives crossing nodes over InfiniBand. The cost is that a pipeline is only as busy as your scheduling makes it, which is the bubble, and that stage imbalance directly becomes idle time everywhere else.

**27. Pipeline bubble?**
Idle GPU time during warmup/cooldown of pipeline. Bubble fraction $\approx (P-1)/(P-1+m)$ where $P$ = stages, $m$ = micro-batches.

> **Saying it out loud.** The bubble is the idle time at the head and tail of the pipeline while it fills and drains — during warmup only stage 0 has work, and during cooldown only the last stage does. The fraction is P minus 1 over P minus 1 plus m, with P stages and m micro-batches, so with 4 stages and 8 micro-batches you're wasting 27% of your cluster, but at 32 micro-batches it's down to 9%. The rule of thumb is you want m at least 4 times P. The tradeoff is that more micro-batches means more in-flight activations, so you pay for a smaller bubble in memory.

**28. GPipe — what does microbatching do?**
Splits batch into $m$ micro-batches, pipelines them. Reduces bubble. More $m$ → less bubble but more activation memory.

> **Saying it out loud.** Microbatching is what turns a serial handoff into an actual pipeline. Without it, stage 0 processes the whole batch, then sits idle for the rest of the forward and backward; with it, stage 0 finishes micro-batch 1 and hands it on, then immediately starts micro-batch 2, so stages overlap in time. The bubble shrinks as m over P grows, but the memory grows because GPipe holds the activations for every in-flight micro-batch until its backward runs. That memory blow-up is precisely what 1F1B was invented to fix.

**29. 1F1B (PipeDream / Megatron)?**
Interleaved forward and backward of different micro-batches. Steady-state every GPU busy. Lower memory than GPipe.

> **Saying it out loud.** 1F1B means each stage alternates: one forward, then one backward, instead of doing all forwards then all backwards. The bubble is the same size as GPipe's — that's the part people get wrong — but the memory is far better, because you run a micro-batch's backward as soon as you can and free its activations instead of holding all m of them. Peak in-flight activations drops from m micro-batches to roughly the number of pipeline stages. The complication is weight versioning: if backward runs against weights that have moved on, you get stale gradients, which is why PipeDream needs weight stashing or a periodic flush.

**30. PP communication cost?**
Send activations forward, gradients backward — small per step, infrequent.

> **Saying it out loud.** It's the cheapest of all the parallelism axes: you only send the hidden state at the stage boundary — batch by sequence by d_model — once per micro-batch each direction, and it's point-to-point, not a collective. Compare that to tensor parallelism's all-reduce twice per layer and it's orders of magnitude less traffic. That's exactly why pipeline is the axis you stretch across nodes and slow links, and why it's the one you use when your interconnect is the weak part of the cluster.

**31. PP layer balancing — why important?**
All stages must take similar compute, else slowest stage bottlenecks. Embedding/output layers often need special handling.

> **Saying it out loud.** Because a pipeline runs at the speed of its slowest stage, and every stall propagates. If one stage takes 20% longer, every other GPU spends 20% of its time waiting — you don't lose a bit of throughput, you lose it everywhere. The tricky parts are the ends: stage 0 also carries the token embedding and the last stage carries the output projection and the loss, and for a large vocabulary that's a genuinely expensive layer. Llama 3's fix is to give those stages one fewer transformer layer to compensate, which is a nice concrete detail to have ready.

---

## E. 3D parallelism

**32. TP × PP × DP for 175B on 1024 GPUs — typical config?**
TP = 8 (intra-node), PP = 16 (cross-node), DP = 8 (replicate). Each rank holds $175\mathrm{B}/(8 \cdot 16) = 1.4\mathrm{B}$ params.

> **Saying it out loud.** TP 8 inside the node on NVLink, PP 16 across nodes, DP 8 on the outside, and 8 times 16 times 8 gets you exactly 1024. The reason to lay it out that way is bandwidth: TP talks constantly so it gets the fastest link, PP sends small activations rarely so it tolerates InfiniBand, and DP syncs once per step so it can be the outermost and slowest ring. The sanity check is per-rank memory — 175 billion divided by TP times PP, so 128, gives about 1.4 billion parameters per GPU, roughly 22 gigabytes of Adam state, which fits comfortably on 80 gigs with room for activations.

**33. Why this layout?**
TP wants high bandwidth (intra-node). PP wants moderate (small cross-node sends). DP wants infrequent (across many nodes).

> **Saying it out loud.** It's frequency-of-communication matching. Tensor parallelism does a blocking all-reduce twice per layer, so it needs the sub-microsecond latency and near-terabyte bandwidth you only get on NVLink inside a chassis. Pipeline parallelism sends one activation tensor per micro-batch boundary, which is small and infrequent, so InfiniBand is fine. Data parallelism does one big gradient sync per optimizer step, so it's the most latency-tolerant of the three and goes outermost. Get this ordering wrong — TP across nodes, say — and you can lose more than half your MFU without changing a single line of model code.

**34. What does context parallelism do?**
Splits attention computation across sequence positions. Critical for million-token context.

> **Saying it out loud.** Context parallelism shards the sequence itself, so each GPU owns a slice of the tokens rather than a slice of the weights. That's the only axis that helps when a *single* example is too long to fit — at a million tokens, no amount of tensor or pipeline splitting saves you, because the activation memory scales with sequence length on every device. The hard part is attention, since a query on one GPU needs keys and values from all the others, and that's what Ring Attention exists to solve.

**35. Ring attention — basic mechanism?**
Each GPU holds a slice of the sequence. K/V are passed in a ring; each rank computes its query against all K/V over time.

> **Saying it out loud.** Each GPU holds its own slice of queries permanently, and the key-value blocks circulate around a ring — at step one you attend against your own KV, at step two against your neighbour's, and so on. Because attention's softmax can be computed incrementally with a running max and running denominator, you can accumulate a correct, exact result block by block without ever materialising the full attention matrix. It's the same online-softmax idea as FlashAttention, just distributed across GPUs instead of across SRAM tiles. The tradeoff is that the ring has N steps of communication, so if your interconnect can't keep up with your compute the whole thing stalls waiting on KV blocks.

---

## F. Expert parallelism (MoE)

**36. EP — what is it?**
Different experts on different GPUs. Tokens routed via all-to-all to their assigned expert.

> **Saying it out loud.** Expert parallelism puts different experts on different GPUs, because in an MoE the experts are the parameters — a model like DeepSeek-V3 has 256 of them per layer and no single device holds them all. Each token's router picks its top-K experts, and the token has to physically travel to wherever those experts live, which is an all-to-all. The nice property is that compute per token stays constant as you add experts, so you scale capacity without scaling FLOPs; the ugly property is that you've turned a local matmul into a network operation.

**37. MoE communication pattern per layer?**
Two all-to-alls: dispatch tokens to experts, gather outputs back.

> **Saying it out loud.** Two all-to-alls per MoE layer: one to dispatch each token to the device holding its chosen expert, and one to bring the results back so the residual stream reassembles in the right order. That's it structurally, but the volume depends on the routing — with top-2 routing every token gets sent twice, so you're moving roughly 2 times batch times sequence times d_model of data, twice, per layer. That's the number that decides whether your MoE is fast or a disappointment.

**38. Why is all-to-all the bottleneck?**

> **Saying it out loud.** All-to-all is the worst-behaved collective because every rank sends a different message to every other rank — there's no ring trick, no tree, no reuse, and the pattern is irregular because routing decides it at runtime. Across nodes it's bounded by your slowest network hop and it can't be overlapped easily, since the very next operation needs the tokens that are in flight. And it's data-dependent: if routing skews, some links carry far more than others and the whole layer waits for the worst one. That's why frontier MoE work — DeepSeek's DualPipe, for instance — is largely about hiding all-to-all behind compute.
Most expensive collective; scales poorly across nodes; no overlap with compute.

**39. Capacity factor — purpose?**
Caps tokens per expert per batch. Drops overflow. Prevents one expert from being overloaded.

> **Saying it out loud.** Capacity factor is a hard cap on how many tokens any one expert will accept in a batch, usually expressed as a multiple of the average — capacity factor 1.25 means an expert takes 25% more than its fair share and no more. You need it because the all-to-all buffers have to be a fixed size for the collective to work at all, and because one expert receiving everything would serialise the layer. The cost is that overflow tokens get dropped — they skip the expert and pass through on the residual alone — so a badly balanced model is silently losing computation on some of its tokens. That silent dropping is the failure mode to name.

**40. Auxiliary loss in MoE?**
Penalty added to encourage uniform routing across experts. DeepSeek-V3 uses auxiliary-loss-free balancing instead.

> **Saying it out loud.** The auxiliary loss is a small extra term that penalises the router for uneven expert usage — typically the dot product of the fraction of tokens routed to each expert with the mean router probability for that expert, which is minimised when everything is uniform. It works, but it's a second objective fighting your language-modelling loss, so it costs a little quality and needs its coefficient tuned. DeepSeek-V3's alternative is neat: keep a per-expert bias on the routing scores and nudge it down when an expert is overloaded and up when it's starved. Same balance, no gradient interference — that's the tradeoff worth naming.

---

## G. Compute / communication overlap

**41. DDP overlap?**
Gradient all-reduce of layer $\ell$ overlaps with backward of layer $\ell - 1$.

> **Saying it out loud.** As soon as a layer's gradients are computed, you kick off their all-reduce and immediately move on to the next layer's backward — so the network is busy while the GPU keeps doing math. PyTorch does this with gradient buckets: it groups parameters into buckets of a few tens of megabytes and fires the collective when a bucket fills, rather than one tiny collective per tensor. If it works, gradient sync is essentially free; if your bucket size is wrong or you have a lot of small parameters, you get lots of latency-bound collectives and the overlap disappears.

**42. FSDP overlap?**
All-gather of weights for layer $\ell + 1$ overlaps with compute of layer $\ell$.

> **Saying it out loud.** FSDP prefetches: while layer L is computing, it's already all-gathering the parameters for layer L plus 1, so the weights land just in time. That's what makes stage 3 tolerable despite its 50% extra communication — the extra traffic doesn't cost wall-clock time if it hides behind math. It only works when compute per layer is big enough to cover the gather, so it breaks down for small models, short sequences, or a slow interconnect. When it breaks you see it directly in a low MFU with the GPUs sitting idle waiting on NCCL.

**43. MFU — what is it?**
Model FLOPs Utilization. Achieved FLOPs / theoretical peak. Frontier targets > 50%.

> **Saying it out loud.** MFU is achieved useful FLOPs over the hardware's theoretical peak, and it's the honest single number for how well a training run is engineered. You compute it as 6 times parameters times tokens per second, divided by peak — and for BF16 on an H100 that peak is about 989 teraflops dense, not the 1,979 number, which is the sparsity figure. Frontier runs target above 50%, and hitting 40 to 50% on a large cluster is genuinely good work. It's bounded by one by construction, so if you ever compute an MFU above 100% you've used the wrong peak or the wrong FLOP formula.

**44. HFU vs MFU?**
HFU (Hardware FLOPs Utilization) counts all FLOPs done. MFU counts only useful (non-recomputed) FLOPs. MFU < HFU when activation checkpointing is on.

> **Saying it out loud.** HFU counts every FLOP the hardware actually executes; MFU counts only the ones that contribute to the model's forward and backward. The gap between them is almost entirely recomputation from activation checkpointing — those FLOPs are real work the GPU did, but you did them twice. So a run with aggressive checkpointing might show 60% HFU and 45% MFU, and both numbers are true and useful: HFU tells you whether the kernels are efficient, MFU tells you whether your memory strategy is costing you throughput.

---

## H. Failures at scale

**45. Loss spike — common cause + fix?**
Numerical instability (BF16 limits, optimizer state). Fix: gradient clipping (1.0), warmup, lower LR, BF16 over FP16, restart from checkpoint.

> **Saying it out loud.** A loss spike is usually numerics meeting a bad batch. Something makes a few gradients enormous — a weird data shard, an attention logit that blows up, an optimizer moment that's gone stale after a restart — and in low precision that propagates. The standard toolkit is gradient clipping at 1.0, a longer warmup especially at large batch, lowering the learning rate, and BF16 rather than FP16 so you're not fighting range. The operational answer matters as much as the numerical one: you keep frequent checkpoints, you monitor loss automatically, and when a spike doesn't recover on its own within a few hundred steps you roll back and skip the data. Spikes that recover are fine; spikes that don't are a dead run.

**46. NaN in attention — common cause?**
FP16 overflow in softmax for large logits. Fix: BF16 (FP32 exponent range), or compute attention in higher precision.

> **Saying it out loud.** Classic cause is FP16 overflow in the softmax — attention logits grow with the dot product of query and key, and once they exceed roughly 65,000 you get an infinity, the exponential produces inf over inf, and every downstream value becomes NaN. Then it spreads instantly through the residual stream and into your weights on the next step. The fixes are BF16, which has FP32's exponent range and simply doesn't overflow there, or computing the softmax accumulation in FP32 even when everything else is half precision — which is what FlashAttention does anyway. The other candidate is a LayerNorm dividing by a near-zero variance, so bumping epsilon is worth checking too.

**47. NCCL hang — what's happening?**
One rank stuck (e.g., bad GPU, ECC error) while others wait on collective. Fix: timeouts, watchdog, health checks before training.

> **Saying it out loud.** A hang means one rank never arrived at the collective and everyone else is blocked waiting, because NCCL collectives are barriers — all ranks in, or nobody out. The cause is usually mundane: one GPU threw an ECC error, one host is stuck in slow data loading, or one rank hit an exception in Python and died without tearing down the job. It's insidious because nothing errors out; the job just stops making progress and burns money until the NCCL timeout fires. Defences are aggressive timeouts, a per-rank watchdog that reports the last collective it entered, and pre-flight health checks so you find the bad GPU before you start, not two hours in.

**48. Straggler — what's the impact?**
One slow GPU bottlenecks the whole job (synchronous SGD). Fix: pre-flight checks; redundant replicas; periodic re-detection.

> **Saying it out loud.** One slow GPU sets the pace for the entire job, because synchronous SGD means every step ends with a collective that everyone waits on. So a single card running 10% slow — thermal throttling, a degraded NVLink, a noisy neighbour on the host — costs you 10% of your whole cluster, not 10% of one GPU. The fixes are preventative: benchmark every node before the run and eject outliers, keep re-checking during training because degradation happens mid-run, and keep spare nodes so replacement is fast. The number that makes this vivid is that at 1024 GPUs, one bad card is a rounding error in hardware and a full-percentage-point tax on throughput.

**49. Checkpoint write times?**
Naive synchronous checkpointing of 100 GB is slow. Fix: async checkpointing, local node checkpoint + later upload, sharded format.

> **Saying it out loud.** Checkpointing a large run means writing hundreds of gigabytes to shared storage, and doing it synchronously means the whole cluster sits idle while the filesystem catches up. The fix is layered: write to local NVMe first and upload in the background, do it asynchronously by snapshotting the state to host memory and letting a separate thread flush it, and use a sharded format so every rank writes its own piece in parallel rather than funnelling through rank zero. The tension worth naming is checkpoint frequency against overhead — checkpoint too rarely and a failure costs you hours of compute, too often and you spend your run doing I/O. Every 30 minutes or so is the usual compromise.

**50. Restart on different topology?**
Need re-sharded checkpoints. Frameworks like DeepSpeed and TorchTitan support this.

> **Saying it out loud.** You need resharding, because a checkpoint written under one parallelism layout doesn't line up with another. If you saved with TP 8 and PP 16 and you want to restart on a smaller cluster with TP 4, the tensors are physically split differently and someone has to stitch and re-cut them. The clean solution, which DeepSpeed and TorchTitan both implement, is a topology-agnostic distributed checkpoint format that stores logical tensors with their sharding metadata, so the loader reassembles whatever layout you ask for. This matters more than it sounds — hardware failures mid-run routinely force you onto a different node count, and if you can't restart on a different topology you're stuck waiting for the exact same cluster.

---

## Quick fire

**51.** *Adam state per param?* 8 bytes (FP32 momentum + variance).
**52.** *DDP requires what to fit?* Full model on one GPU.
**53.** *FSDP equiv to which ZeRO?* Stage 3.
**54.** *Megatron FFN: first linear?* Column-parallel.
**55.** *Pipeline bubble formula?* $(P-1)/(P-1+m)$.
**56.** *MoE per-layer collectives?* Two all-to-alls.
**57.** *Activation checkpointing trades?* Memory for compute.
**58.** *NCCL ring all-reduce per-rank cost?* $2(N-1)/N \cdot P$ bytes.
**59.** *MFU = ?* Achieved / peak useful FLOPs.
**60.** *BF16 advantage over FP16?* FP32 exponent range.

> **Saying it out loud (quick-fire bank).** These are the reflex facts, and the way to practice is to say each one in under five seconds with no preamble. Adam is 8 bytes per parameter for the moments, 16 all-in with BF16 weights, grads and the FP32 master copy. FSDP is ZeRO-3. Megatron goes column then row so a block costs two all-reduces. The pipeline bubble is P minus 1 over P minus 1 plus m. Ring all-reduce moves about 2P bytes per rank, constant in the number of GPUs. BF16 beats FP16 on exponent range, not precision. If any of those takes you more than a breath, that's the one to drill — in a real interview the terse facts are what buy you the time to reason out loud about the hard parts.

---

## Self-grading

If you can't answer 1-15, you can't talk about training memory. If you can't answer 16-35, you'll fail any infra question on a frontier-lab interview. If you can't answer 36-50, large-scale LLM systems interviews will go past you.

Aim for 40+/60 cold.
