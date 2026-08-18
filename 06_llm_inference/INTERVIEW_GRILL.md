# LLM Inference — Interview Grill

> 50 brutal questions on LLM inference optimization. Drill until you can answer 40+ cold.

---

## A. Foundational: prefill vs decode

**1. What are the two phases of LLM inference and how do they differ?**
Prefill: process the entire prompt in one parallel forward pass. Compute-bound on modern GPUs (high arithmetic intensity, dense matmuls). Decode: generate one token at a time, attending to the growing KV cache. Memory-bandwidth-bound (read full model weights for one token of work). Almost every inference optimization addresses one of these regimes specifically.

> **Saying it out loud.** Inference has two phases and they're bottlenecked by completely different things. Prefill reads your whole prompt in one parallel pass, so it's doing big dense matrix multiplies and it's compute-bound — the GPU is genuinely busy. Decode then produces one token at a time, and to make a single token you have to read every weight in the model out of memory, so you're bandwidth-bound and the tensor cores mostly sit idle. That split is the single most useful frame in this whole area, because every optimisation you can name targets one side or the other.

**2. Why is decode memory-bound rather than compute-bound?**
Each decode step reads all model weights from HBM but performs only enough math for one token's worth of forward pass. Arithmetic intensity is ~1–2 ops/byte; H100's balance point is ~330 ops/byte. Tensor cores sit idle waiting for memory. The fix is batching: more concurrent decodes amortize the weight read.

> **Saying it out loud.** Because the work per byte moved is tiny. To emit one token you read all the model's weights out of HBM and do roughly one multiply-accumulate per weight — that's an arithmetic intensity of about 1 or 2 operations per byte, while an H100 needs something like 300 before the compute units are the limit. So you're off by two orders of magnitude and the GPU is essentially a very expensive memory controller. The fix is batching: if 64 requests decode together, you read the weights once and serve 64 tokens, so intensity scales with batch size.

**3. What's the theoretical lower bound on decode latency?**
$(\text{model-size-bytes} / \text{HBM-bandwidth}) + (\text{KV-cache-per-token} / \text{HBM-bandwidth}) + \text{overhead}$. For 70B fp16 = 140 GB (doesn't fit on a single 80 GB H100, so assume TP$\geq 2$): each GPU reads ~70 GB at ~3 TB/s ≈ 23 ms per token. No software optimization can beat this without reducing bytes per step (quantization, speculation).

> **Saying it out loud.** The floor is model bytes divided by memory bandwidth, because you cannot produce a token without reading every weight once. For a 70B model in fp16 that's 140 gigabytes; split across two H100s at roughly 3 terabytes a second each, that's about 23 milliseconds per token, so around 43 tokens a second per request no matter how good your software is. The only way past it is to move fewer bytes — quantise the weights — or get more tokens per read, which is speculation or batching. That's the number to have ready, because it turns "make it faster" into an arithmetic question.

**4. Why is prefill compute-bound while decode is memory-bound?**
Prefill batches $P$ tokens into a single forward pass, so weights are reused across $P$ queries → arithmetic intensity scales with $P$. Decode has one token at a time → intensity is $\sim 1$. The same hardware behaves entirely differently in the two regimes.

> **Saying it out loud.** It comes down to weight reuse. In prefill you push hundreds or thousands of tokens through the same weight matrix in one go, so each weight you load gets used hundreds of times and arithmetic intensity scales with prompt length. In decode you push one token, so each weight gets used once. Same hardware, same kernels, opposite bottleneck. That's why you tune them separately, and it's the whole motivation behind disaggregated serving.

---

## B. KV cache

**5. Write the KV cache memory formula.**
$\text{KV-size} = 2 \cdot n_{\text{layers}} \cdot d_{\text{model}} \cdot \text{seq-len} \cdot \text{bytes} \cdot \text{batch-size}$. Factor of 2 for $K$ and $V$. $d_{\text{model}} = n_{\text{heads}} \cdot d_{\text{head}}$. With GQA, replace $d_{\text{model}}$ with $n_{\text{kv-heads}} \cdot d_{\text{head}}$.

> **Saying it out loud.** KV cache size is two — one each for keys and values — times layers, times the KV dimension, times sequence length, times batch, times bytes per number. The bit people get wrong is the KV dimension: with grouped-query attention it's the number of *KV* heads times head dim, not the full model dimension, which is where the 8x saving comes from. It's linear in every term, which matters because it means long context and large batch trade directly against each other. If you can write that formula on a whiteboard and plug numbers in, you've answered most KV questions.

**6. Compute the KV cache size for LLaMA-2 70B at 8K context, batch 32, fp16.**
$80 \text{ layers} \times 8192 \text{ } d_{\text{model}} \times 8192 \text{ seq} \times 2 \text{ bytes} \times 2 \text{ } (K, V) \times 32 \text{ batch} \approx 687$ GB without GQA. LLaMA-2 70B uses GQA with 8 KV heads (vs 64 attention heads), so divide by 8: $\sim 86$ GB. Still substantial.

> **Saying it out loud.** Plug it in: 80 layers, model dimension 8192, 8K sequence, 2 bytes for fp16, times 2 for K and V, times 32 for the batch — that's about 687 gigabytes if you used full multi-head attention. Which is absurd, since an H100 has 80. LLaMA-2 70B uses 8 KV heads against 64 attention heads, so you divide by 8 and get roughly 86 gigabytes. Still large — that's more than a whole GPU's worth of memory just for cache, on top of the 140 gigabytes of weights, which is exactly why KV memory rather than compute is what limits your batch size.

**7. What problem does KV cache solve?**
Without it, every decode step recomputes attention keys and values for the entire prefix → $O(n^2)$ total compute over $n$ decode steps. With it, you compute $K/V$ incrementally for the new token only → $O(n)$ total. KV cache makes decode tractable.

> **Saying it out loud.** Without a cache, every new token would recompute the keys and values for the entire prefix, so generating $n$ tokens costs $O(n^2)$ work and you'd redo the same arithmetic over and over. With the cache you compute K and V once per token, store them, and each new step only computes its own — linear total. It's just memoisation, and it's the difference between generation being usable and being hopeless. The tradeoff is the classic one: you traded compute for memory, and memory is now your problem.

**8. What problem does KV cache create?**
Memory. KV cache scales linearly with sequence length and batch size and often dominates GPU memory in serving — limiting how many concurrent users you can fit. Long contexts make this worse.

> **Saying it out loud.** It creates a memory problem that grows with both context length and batch size. In production the cache routinely rivals or exceeds the weights, and unlike weights it isn't shared between requests — every user brings their own. So the practical constraint on how many people you can serve at once is usually KV capacity, not FLOPs. That's why so much of the modern stack — GQA, MLA, paging, KV quantisation, attention sinks — is all attacking the same number.

**9. What's MQA?**
Multi-Query Attention (Shazeer 2019). All attention heads share a single $K$ matrix and a single $V$ matrix. KV cache shrinks by factor $n_{\text{heads}}$. Quality loss is small for many tasks but non-trivial for some.

> **Saying it out loud.** Multi-query attention gives every attention head its own queries but makes them all share one set of keys and values. That shrinks the cache by the number of heads — a 64x cut for a 64-head model, which is enormous. The cost is quality: you've collapsed a lot of representational capacity in the K/V projections, and it shows up on harder tasks and with training instability. That's exactly why GQA exists as the middle ground, and why almost nobody ships pure MQA today.

**10. What's GQA?**
Grouped-Query Attention (Ainslie et al. 2023). Compromise between MHA and MQA: groups of attention heads share $K/V$. LLaMA-2 70B uses 8 KV heads with 64 attention heads (group size 8). KV cache shrinks 8x with minimal quality loss.

> **Saying it out loud.** Grouped-query attention is the compromise that won. Instead of one shared K/V set or one per head, you have a handful — LLaMA-2 70B uses 8 KV heads for 64 attention heads, so each group of 8 query heads shares one. You get an 8x KV cut with quality essentially indistinguishable from full multi-head. The reason it's the default now is that it sits at the good part of the curve: MQA's savings are only marginally better and its quality cost is real.

**11. What's PagedAttention?**
Allocate KV cache in fixed-size blocks (e.g. 16 tokens per block) instead of contiguous per-sequence buffers (Kwon et al. 2023, vLLM). A block table per request maps logical positions to physical blocks. Eliminates internal fragmentation (no need to reserve max-length up front) and external fragmentation. Enables block-level sharing for prefix caching.

> **Saying it out loud.** PagedAttention takes the operating-system idea of virtual memory and applies it to the KV cache. Instead of one contiguous buffer per request sized for the worst case, you allocate fixed-size blocks — 16 tokens each — and keep a block table mapping a request's logical positions to scattered physical blocks. The attention kernel walks the block table instead of assuming contiguity. That kills fragmentation, both the wasted slack from over-reserving and the holes left by finished requests, and it's what made vLLM's throughput numbers possible.

**12. Why does PagedAttention enable 2–4x more concurrent requests?**
Naive allocation reserves max-length per request → most memory is wasted on slack. PagedAttention only allocates blocks as the sequence grows. With variable-length workloads, this typically frees 50–75% of the KV memory previously wasted on padding.

> **Saying it out loud.** Because the naive approach reserves maximum length for every request, and almost nobody uses their maximum. If you allow 8K context and the average request uses 500 tokens, you're wasting more than 90% of the memory you set aside. Paging allocates as the sequence actually grows, so that slack comes back as extra concurrent requests. On typical variable-length traffic that's 50 to 75% of KV memory recovered, and since KV capacity is what caps your batch size, it turns directly into 2 to 4 times the concurrency.

**13. Trade-offs of PagedAttention?**
Slightly more complex attention kernel (must follow block table during attention). Block size hyperparameter (16 typical: too small = overhead, too large = fragmentation returns). Worth it.

> **Saying it out loud.** The costs are real but small. Your attention kernel is more complicated because it has to gather through a block table rather than stride through contiguous memory, so there's a bit of indirection overhead. And block size is a hyperparameter with a tradeoff on each side — too small and you pay table-management overhead per block, too large and internal fragmentation creeps back in, which is why 16 is the usual answer. Given it buys 2 to 4 times the concurrency, it's an easy trade, and effectively every serious serving stack has adopted it.

---

## C. Continuous batching and serving

**14. What's continuous batching?**
Iteration-level scheduling: at every decode step, evict completed requests and admit new ones. Don't wait for the longest request in the batch to finish before processing new requests. Origin: Yu et al. 2022 (Orca paper). Implemented in vLLM, TensorRT-LLM, TGI, SGLang.

> **Saying it out loud.** Continuous batching means you make scheduling decisions every decode step instead of once per batch. The moment a request finishes, its slot is freed and a waiting request takes it, rather than everyone waiting for the longest generation in the batch to complete. It's iteration-level scheduling, from the Orca paper. That one change is the difference between a GPU that's idle most of the time and one that's saturated, and it's in every modern serving stack.

**15. Why does continuous batching matter?**
Without it, GPU utilization for serving is often <30% because short requests finish early but the batch is held until the longest one completes. With continuous batching, GPU stays busy: completed slots are immediately filled. Throughput improvements of 5–20x are common.

> **Saying it out loud.** Because response lengths vary enormously — one request writes 20 tokens and another writes 2,000, and with static batching the short one's slot sits idle for the entire rest of the batch. In practice that leaves utilisation under 30%. Continuous batching refills those slots immediately, and the reported throughput gains are in the 5 to 20 times range, which is a bigger win than almost any kernel optimisation you could do. The tradeoff to name is fairness and tail latency: with everyone contending for slots, you need a scheduling policy or long requests get starved.

**16. What's chunked prefill?**
Split a long prefill into chunks (e.g. 256 tokens at a time) so prefill can be interleaved with decode rather than blocking it for many milliseconds. Improves TPOT for ongoing requests when a new long-prompt request arrives.

> **Saying it out loud.** Chunked prefill splits a long prompt into pieces — a few hundred to a couple of thousand tokens — and processes them across several scheduler steps instead of one giant blocking pass. The problem it solves is head-of-line blocking: if someone submits a 100K-token prompt, a naive scheduler spends hundreds of milliseconds on it while every other user's token generation stalls. Chunking lets you interleave those prefill pieces with everyone else's decode steps. It's a latency-fairness fix, and the small cost is slightly worse prefill efficiency from the smaller batches.

**17. Why is mixing prefill and decode hard?**
They have different computational characteristics. Prefill is compute-bound and benefits from large effective batch sizes (many tokens at once). Decode is memory-bound and works at batch $\approx 1$ token per request. Naive interleaving wastes compute. Modern systems use specialized kernels and scheduling.

> **Saying it out loud.** Because they want opposite things from the hardware. Prefill is compute-bound and wants as many tokens in flight as possible; decode is memory-bound and is one token per request by definition. Put them in the same kernel launch and you either starve the compute or you make the decode users wait. So modern systems use fused kernels that handle both token types in one pass plus a scheduler that budgets how many prefill tokens can ride along with the decode batch each step. And at real scale you stop trying and split them onto different machines, which is disaggregated serving.

---

## D. Speculative decoding

**18. Walk me through speculative decoding.**
Use a small **draft model** to autoregressively propose $k$ tokens. Run the target model in a single forward pass over those $k$ tokens to get target probabilities for each. For each proposed token, accept with probability $\min(1, p_{\text{target}} / p_{\text{draft}})$. If accepted, keep going; if rejected, sample from the residual distribution $(p - q)_+$ and stop. Net effect: you generate (in expectation) more than 1 token per target forward pass.

> **Saying it out loud.** A small cheap model drafts the next few tokens one at a time. Then the big model scores all of those positions in a single forward pass, which is nearly free because you were bandwidth-bound anyway and checking $k$ tokens costs about the same as generating one. You walk the drafts in order and accept each with probability target over draft, capped at 1; at the first rejection you throw away the rest and sample a corrected token from the residual. So per expensive forward pass you get more than one token out, and the whole thing rests on decode being memory-bound rather than compute-bound.

**19. Is speculative decoding exact or approximate?**
**Exact.** The rejection-sampling rule is constructed to make the distribution of accepted tokens exactly the target's distribution. Output samples are statistically indistinguishable from regular target-model decoding. This is a critical correctness property.

> **Saying it out loud.** Exact, and that's the point worth emphasising. It isn't "close enough" — the accept-reject rule plus the residual resampling are constructed so the output distribution is algebraically identical to sampling from the target model alone. The residual step is the part people miss: on rejection you don't just sample from the target, you sample from target-minus-draft clipped at zero and renormalised, and that's what makes the algebra come out right. So it's a pure latency optimisation with zero quality cost, which is why it's safe to turn on in production.

**20. What controls the speedup?**
Acceptance rate $\alpha$ and the draft-to-target compute ratio.

$$
\text{Speedup} \approx \frac{1 + \alpha + \alpha^2 + \cdots + \alpha^k}{1 + (\text{draft-cost} / \text{target-cost}) \cdot k}
$$

High $\alpha$ (close draft to target) and small $\text{draft-cost} / \text{target-cost}$ give big speedups. Typical: 2–3x with a 7B draft for a 70B target.

> **Saying it out loud.** Two numbers: the acceptance rate and the cost ratio between draft and target. If the draft agrees often you get many tokens per expensive pass; if the draft is itself expensive you eat that saving. The rough shape is a geometric series in the acceptance rate on top, divided by one plus draft cost times the number of speculated tokens. The realistic figure is 2 to 3 times with a 7B draft against a 70B target, and there's an optimum for $k$ — speculate too far ahead and you're spending draft compute on tokens that will almost certainly be rejected.

**21. When does speculative decoding NOT help?**

- Throughput-limited servers running at large batch size, where decode is already compute-bound.
- Very low acceptance rates (mismatched draft model).
- Latency-tolerant batch jobs where total compute is the metric (speculation increases total compute).

> **Saying it out loud.** It stops helping the moment you're no longer memory-bound. On a busy server running large batches, decode has already become compute-bound through batching, and the extra verification work is pure overhead — speculation and batching are competing for the same headroom. It also fails when the acceptance rate is low, because you're paying draft cost for tokens you throw away, which happens on out-of-distribution or highly technical text. And for offline batch jobs where you only care about total compute rather than latency, it's strictly negative since it increases total FLOPs. The one-liner: speculation trades throughput for latency, so it's for interactive serving.

**22. What's self-speculative decoding?**
Use shallow layers of the target model itself as the draft. No external model needed. Trade some quality of the draft for memory simplicity. EAGLE and Medusa are related: train extra heads to predict multiple future tokens.

> **Saying it out loud.** Self-speculation removes the separate draft model. You either run a subset of the target's own layers as the draft, or you attach extra prediction heads on top of the target's hidden state — that's Medusa — or train a small feature predictor on the penultimate layer, which is EAGLE. The appeal is operational: one set of weights to deploy, no second model to keep aligned with the target after every fine-tune. And because the draft shares the target's representation, acceptance rates run higher than an independently-trained small model, which is why EAGLE-style approaches largely displaced the classic two-model setup.

**23. What's tree speculation?**
Propose a tree of candidate continuations (multiple branches from the same prefix). Verify with one target forward pass that batches all branches together. Higher effective acceptance because the target model picks the longest matching branch.

> **Saying it out loud.** Tree speculation proposes a branching set of continuations instead of one linear guess — several candidates at the first position, several after each of those, and so on. You verify the whole tree in one target pass using a carefully built attention mask so branches don't see each other, then keep the longest path that matches. The gain is that a single wrong token no longer kills the whole speculation, because a sibling branch may still match. You're spending more verification compute for a higher effective acceptance rate, which is a good trade exactly when you're memory-bound.

---

## E. FlashAttention

**24. What's FlashAttention?**
I/O-aware attention computation (Dao et al. 2022). Tile $Q, K, V$ so that blocks fit in SRAM. Use online softmax to compute partial softmax statistics block-by-block. Same FLOPs as standard attention, but avoids materializing the full $N \times N$ attention matrix in HBM. Result: 2–4x wall-clock speedup, especially at long sequence lengths.

> **Saying it out loud.** FlashAttention computes exactly the same attention, just without ever writing the big attention matrix to memory. It tiles Q, K and V into blocks small enough to live in on-chip SRAM, and uses an online softmax so it can accumulate the correct result block by block without needing the full row first. Same FLOPs, dramatically less memory traffic, and you never allocate that $N \times N$ matrix — which is also why it makes long context feasible at all. Typical wall-clock gain is 2 to 4 times, larger the longer the sequence.

**25. Why is FlashAttention faster — algorithmically or hardware-wise?**
Hardware-wise. The algorithm has the same FLOPs. The speedup comes from reducing HBM reads/writes. Modern GPUs are massively memory-bandwidth-limited for attention; reducing memory traffic is the lever.

> **Saying it out loud.** Hardware-wise, entirely. The FLOP count is unchanged — this isn't a cleverer algorithm in the complexity sense. What changes is memory traffic: standard attention writes an $N \times N$ matrix out to HBM and reads it back, twice, and at long sequence lengths that traffic dominates the runtime completely. FlashAttention keeps the intermediate in SRAM, which is roughly an order of magnitude faster. It's the canonical example of the broader lesson that on modern GPUs you optimise data movement, not arithmetic.

**26. Walk me through online softmax.**
Standard softmax: compute max, subtract, exponentiate, sum, divide. Requires two passes over data (first for max, second for normalization). Online softmax: maintain running max $m$ and running sum $s$. For each new value $x$: $m' = \max(m, x); s' = s \cdot \exp(m - m') + \exp(x - m'); m = m'; s = s'$. Single pass. Numerically equivalent. Lets attention be computed block-by-block without materializing the full matrix.

> **Saying it out loud.** Normally softmax needs two passes — once to find the maximum for numerical stability, once to exponentiate and sum. Online softmax does it in one by carrying a running maximum and a running sum, and whenever a new block reveals a larger maximum you rescale the accumulated sum by the exponential of the difference. It's the same trick as computing a running mean incrementally, and it's numerically exact, not an approximation. That correction factor is the entire reason attention can be computed block by block, and therefore the reason FlashAttention exists.

**27. What's FlashAttention-2 vs FA-1?**
FA2 (2023): better parallelization across thread blocks (parallelize over the sequence dimension and not just heads), reduced non-matmul ops. ~2x faster than FA1.

> **Saying it out loud.** FlashAttention-2 is the same algorithm with much better GPU occupancy. FA1 parallelised over batch and heads, which leaves a modern GPU underutilised when you have a long sequence but few heads; FA2 also parallelises along the sequence dimension so every streaming multiprocessor has work. It also reorders the loops to cut non-matmul operations, which matter because tensor cores are so much faster than the general-purpose units that a little bookkeeping arithmetic dominates. Net result is roughly 2x over FA1 and around 70% of theoretical peak.

**28. What's FlashAttention-3?**
FA3 (2024): uses Hopper-specific features (TMA, async warpgroup matmul, FP8 support). Reaches near-peak bf16 utilization on H100.

> **Saying it out loud.** FlashAttention-3 is FA2 rewritten around Hopper-specific hardware. It uses the tensor memory accelerator for asynchronous data movement, overlaps the softmax on the general units with matmuls on the tensor cores instead of alternating between them, and supports FP8. The point is that on H100 the earlier versions left a lot on the table because they didn't exploit asynchrony, and FA3 gets near peak bf16 throughput. It's a good illustration that these kernels are hardware-generation-specific, not portable wins.

**29. Does FlashAttention help decode?**
Less than prefill, because decode is $O(n)$ attention not $O(n^2)$. FlashDecoding and FlashDecoding++ are decode-specialized variants that bring similar I/O-aware tiling to the decode case (parallel over the KV sequence dimension).

> **Saying it out loud.** Much less, because decode isn't the shape FlashAttention was designed for. With one query token attending to a long cache, there's no big square matrix to avoid materialising — the attention is $O(n)$ and you're just reading the cache. The relevant variants are FlashDecoding and FlashDecoding++, and their trick is different: split the KV sequence into chunks processed in parallel and combine the partial softmaxes afterwards, which gives you parallelism when a single query alone wouldn't fill the GPU. Worth being precise about this, because plenty of people assume FlashAttention speeds up everything.

---

## F. Quantization

**30. What's W4A16?**
4-bit weights, 16-bit activations. Standard for LLM inference. Methods: GPTQ, AWQ. Memory savings ~4x over fp16; compute is dequantize-then-fp16 matmul, so speedup is ~2–3x not 4x.

> **Saying it out loud.** W4A16 means 4-bit weights with 16-bit activations, and it's the standard for latency-sensitive LLM serving. You quantise only the weights because that's where the bytes are, and decode is bottlenecked on reading them — the activations are tiny by comparison. GPTQ and AWQ are the two standard methods. The number to get right is that memory drops about 4x but speed only improves 2 to 3x, because you dequantise back to fp16 to do the actual matmul, so you're not using faster arithmetic, just moving fewer bytes.

**31. What's W8A8?**
8-bit weights, 8-bit activations. Compute uses INT8 tensor cores (often ~2x throughput vs fp16). Method: SmoothQuant. Quality more sensitive; activation outliers must be handled.

> **Saying it out loud.** W8A8 quantises both sides to 8 bits so you can use the INT8 tensor cores, which is roughly double the fp16 throughput. So unlike W4A16 this is a genuine compute win, not just a memory win, which makes it attractive for throughput-oriented and prefill-heavy workloads. The catch is activations: LLM activations have extreme outlier channels, and a single global scale destroys them. That's what SmoothQuant is for, and it's why W8A8 is more fragile to deploy than weight-only quantisation.

**32. What's GPTQ?**
Per-layer Hessian-based weight quantization. For each layer, quantize weights one at a time, after each step adjusting remaining weights to compensate for quantization error. Uses a small calibration set (~128 samples). Tractable for 70B+ models.

> **Saying it out loud.** GPTQ quantises one layer at a time, and within a layer one weight column at a time, and after each rounding decision it adjusts the remaining unquantised weights to compensate for the error just introduced. The compensation uses second-order information — an approximate Hessian built from a small calibration set of about 128 samples. So it's not naive rounding, it's error-corrective rounding, which is why it holds up at 4 bits where round-to-nearest doesn't. The cost is a few GPU-hours even for a 70B model, which is fine as a one-off.

**33. What's AWQ?**
Activation-aware weight quantization. Identify the small fraction of weights that have high activation magnitude (~1%). Scale those before quantization to preserve them. Faster than GPTQ to apply; comparable or better quality.

> **Saying it out loud.** AWQ's insight is that not all weights matter equally, and which ones matter is determined by the *activations* they multiply, not by the weights' own magnitude. Roughly 1% of channels see large activations, and protecting those preserves almost all the quality. So AWQ finds a per-channel scale that shifts those salient weights into a range that quantises cleanly. It needs no backprop and no Hessian, so it's much faster to apply than GPTQ, with comparable or better quality — which is why it became the common default.

**34. What's SmoothQuant?**
Migrates outlier activations to weights pre-quantization. Per-channel scaling factor $s_j$ such that $(X_j / s_j) \cdot (s_j \cdot W_j) = X_j W_j$. Activations become smaller (no outliers); weights become slightly larger. Both fit in INT8 cleanly.

> **Saying it out loud.** SmoothQuant solves the activation-outlier problem by moving the difficulty from activations to weights. Since a matmul is unchanged if you divide the activation channel by $s$ and multiply the corresponding weight column by $s$, you can pick $s$ per channel to flatten the activation outliers at the cost of slightly spikier weights. Weights are far easier to quantise than activations because they're static and well-behaved, so that's a good trade. The result is that both sides fit into INT8 cleanly, which is what makes W8A8 practical.

**35. What's FP8?**
8-bit floating point. Two formats: E4M3 (4 exp bits, 3 mantissa) for forward; E5M2 (5/2) for backward / wider range. Native on H100/Blackwell. Less calibration-sensitive than INT8 because of dynamic range. Increasingly common for both training and inference.

> **Saying it out loud.** FP8 is 8-bit floating point, and the key difference from INT8 is that it keeps an exponent, so it has dynamic range rather than a single fixed scale. That's exactly what LLM activations need, with their outlier channels, so FP8 is far less calibration-sensitive than INT8. There are two formats: E4M3 with more mantissa for forward passes, and E5M2 with more exponent range for gradients. It's native on H100 and later, which means real tensor-core speedup rather than just memory savings, and it's become the default for both training and serving.

**36. What's NF4?**
4-bit information-theoretically optimal float for normally-distributed weights. Used in QLoRA. Better dynamic-range matching than INT4 for typical weight distributions.

> **Saying it out loud.** NF4 is a 4-bit format whose quantisation levels are placed so that each one holds an equal share of the probability mass of a normal distribution, rather than being evenly spaced. That matters because neural network weights really are approximately Gaussian, so you're spending your 16 available values where the weights actually are instead of wasting them on empty tails. It came from QLoRA, where it lets you hold a 65B model on a single 48-gigabyte GPU and fine-tune adapters on top. It's a nice example of matching the number format to the data distribution.

**37. Why does INT8 quantization sometimes hurt LLM quality more than expected?**
Activation outliers. A few channels per layer have activations 100x typical. Naive per-tensor quantization with a global scale clips them, destroying information. SmoothQuant or per-channel scaling fixes it. INT4 weight quantization (no activation quantization) is less affected.

> **Saying it out loud.** Activation outliers. In large transformers a handful of channels per layer carry activations a hundred times larger than typical, and they're not noise — they're functionally important. With per-tensor INT8 quantisation, one global scale has to cover the outliers, so everything else gets crushed into a couple of quantisation levels and you lose the signal. The fixes are per-channel scaling or migrating the outliers into the weights, which is SmoothQuant. And it's worth noting weight-only INT4 sidesteps this entirely, which is why W4A16 is the safer default.

**38. Can you quantize the KV cache?**
Yes, common. INT8 KV is straightforward; INT4 KV needs group-wise quantization (e.g., per 128 tokens). Saves 2–8x KV memory → enables longer contexts and larger batches with no compute change.

> **Saying it out loud.** Yes, and it's increasingly standard because the cache often outgrows the weights at long context. INT8 or FP8 KV is nearly free in quality terms and halves the cache. Going to INT4 needs group-wise scales — quantising in groups of tokens or channels rather than one scale for everything. The detail worth knowing is that K and V want different treatment: keys have strong per-channel outlier structure while values don't, which is exactly the observation KIVI is built on. And it costs no extra compute, so it's pure capacity gain.

---

## G. Multi-GPU and parallelism

**39. What's tensor parallelism?**
Split each weight matrix across GPUs (e.g. attention $Q, K, V$ projections sharded along the head dimension). Each GPU computes part of the matmul; results all-reduced. Standard for inference of >13B models. Inter-GPU communication on every layer.

> **Saying it out loud.** Tensor parallelism splits individual weight matrices across GPUs — you shard the attention projections by head and the MLP by hidden dimension, each GPU computes a slice of the matmul, and you all-reduce to combine. The reason it's the default for inference is that it splits the memory bandwidth too, so each GPU reads less and per-token latency actually falls. The cost is an all-reduce twice per layer, which needs NVLink to be tolerable — that's why TP stays inside a node and you don't stretch it across the network.

**40. What's pipeline parallelism?**
Different GPUs hold different layers; the forward pass flows GPU-to-GPU. Communication only at layer boundaries. Bad for inference latency because of pipeline bubbles (one request can't fill the pipeline). More common in training.

> **Saying it out loud.** Pipeline parallelism puts different layers on different GPUs and passes activations along the chain. Communication is tiny — just the activations at each boundary — so it works over slow interconnects, and that's why it's popular in training. For inference latency it's bad, because a single request occupies one stage at a time and every other stage idles waiting: the bubble. You can fill it by having many requests in flight, so it helps throughput, but it never reduces per-token latency the way tensor parallelism does. In practice you use TP within a node and pipeline only across nodes when a model won't fit.

**41. What's expert parallelism?**
For Mixture-of-Experts models (Mixtral, GPT-4-class). Different experts on different GPUs; routing decides which GPU(s) handle each token. Communication pattern is all-to-all. Tricky to load-balance.

> **Saying it out loud.** Expert parallelism is for mixture-of-experts models: you place different experts on different GPUs, and each token gets routed to whichever ones it needs. The communication pattern is all-to-all — send every token to its experts, then gather the results back — which is much less friendly than an all-reduce. The hard part is load balancing, because routing is learned and data-dependent, so a popular expert becomes a straggler and everyone waits. That's why MoE training uses auxiliary balancing losses and capacity limits that drop overflow tokens.

**42. For 70B inference, what's the typical multi-GPU setup?**
Tensor parallel = 4 or 8 within a node (intra-NVLink). Data parallel across nodes. PagedAttention manages KV cache. Continuous batching for throughput.

> **Saying it out loud.** Tensor parallel 8, or 4, inside a single node so all the all-reduces ride NVLink, and then data parallel replicas across nodes for throughput. On top of that, paged KV so you don't waste memory on reservation, and continuous batching so no slot sits idle. In 2025 you'd add FP8 weights, chunked prefill, and speculative decoding. The reasoning to voice: you scale tensor parallelism only as far as the fast interconnect goes, then you replicate.

---

## H. Latency and metrics

**43. What's TTFT?**
Time To First Token: from request arrival to the first generated token streaming back. Dominated by prefill. Scales with prompt length. Critical for chat UX (target <500ms).

> **Saying it out loud.** Time to first token — how long the user stares at nothing after hitting enter. It's dominated by prefill, so it scales with prompt length, and in agentic workloads with huge system prompts it's often the whole user-visible latency problem. The target for chat is under about 500 milliseconds, and for voice more like 200. The most effective lever is usually prefix caching, because in multi-turn chat and RAG most of that prompt was already processed on a previous request.

**44. What's TPOT (or ITL)?**
Time Per Output Token: average inter-token latency during decode. Dominated by memory bandwidth and batch utilization. Target <50ms for natural reading pace.

> **Saying it out loud.** Time per output token — the gap between tokens once generation is under way, sometimes called inter-token latency. It's set by memory bandwidth and how well you're amortising the weight read across the batch. Under 50 milliseconds is roughly the threshold where text arrives faster than people read, so it feels smooth. Unlike TTFT, this is the number that improves with batching, quantisation and speculation — anything that reduces bytes read per token.

**45. How would you reduce TTFT?**
Prefill chunking, FlashAttention, smaller initial KV (sliding window), prompt caching (share prefix KV across users), faster networking for distributed prefill.

> **Saying it out loud.** TTFT is a prefill problem, so you attack prompt processing. The biggest single win is prefix caching, because in chat and RAG most of the prompt is a repeat of something you already computed. Then chunked prefill so your request isn't stuck behind someone else's enormous prompt. Then the compute-side items — FlashAttention, and for very long prompts splitting the prefill across GPUs. And the unglamorous one worth mentioning: shorten the prompt, since TTFT scales with prompt length and most system prompts are longer than they need to be.

**46. How would you reduce TPOT?**
Continuous batching (more concurrent decodes amortize weight read), KV cache compression, quantization (less weight to read), speculative decoding, MQA/GQA, tensor parallelism (split memory bandwidth across more GPUs).

> **Saying it out loud.** TPOT is a bandwidth problem, so everything you do reduces bytes moved per token or increases tokens per byte. Quantise the weights — 4-bit means a quarter of the reading. Batch more requests so one weight read serves many tokens, which is what continuous batching buys. Shrink the KV cache with GQA or MLA and KV quantisation, since the cache is read every step too. Add speculative decoding to get multiple tokens per forward pass. And tensor parallelism, because splitting the model across GPUs splits the bandwidth requirement as well.

**47. Throughput vs latency — how do you trade them?**
Bigger batch → higher throughput (more amortization of weight read), worse per-request latency (more contention). Choose batch size to maximize throughput subject to latency SLA. Workload mix matters: long-context requests need smaller batches to fit KV.

> **Saying it out loud.** They pull against each other through batch size. A bigger batch amortises the weight read across more requests, so tokens per second per GPU goes up — but each individual user now waits behind more contention, so their per-token latency goes up too. The way you actually do it is to fix a latency SLA, then find the largest batch that still meets it, and that's your operating point. The wrinkle to mention is that long-context requests eat KV memory, so the maximum feasible batch shrinks as context grows — which is why serving cost per token rises with context length even though compute per token barely moves.

---

## I. System / serving design

**48. What's prompt caching / prefix caching?**
Reuse computed KV cache for shared prompt prefixes across requests. Particularly valuable for: multi-turn chats (previous turns are shared), RAG (retrieved documents shared across users), tool-using agents (long system prompts). Implemented naturally on top of PagedAttention via block reference counting.

> **Saying it out loud.** Prefix caching means keeping the computed KV for a prompt prefix and reusing it when another request starts with the same tokens. It's enormous in practice because so much traffic shares prefixes — a fixed system prompt, tool definitions, retrieved documents, or the earlier turns of the same conversation. It falls out naturally from paged KV: blocks are content-addressed and reference-counted, so sharing is just pointing two block tables at the same block. The gotcha to name is that it only works on an exact prefix match from token zero, so changing one character at the top of your system prompt invalidates everything after it.

**49. How does an MoE model change inference?**
Each token only activates a subset of experts (typically 2 out of 8). Compute per token is reduced, but routing introduces overhead and load-balancing problems. Expert parallelism: experts spread across GPUs, all-to-all communication on routing. Memory still scales with total expert count, not active expert count. So MoE saves compute but not memory.

> **Saying it out loud.** Mixture-of-experts changes the compute-to-memory ratio, and mostly not in your favour for serving. Each token activates only a couple of experts, so FLOPs per token drop a lot — that's the selling point. But you still have to hold every expert in memory, because any token might route anywhere, so memory scales with total parameters, not active ones. And since decode is memory-bound, saving FLOPs you weren't bottlenecked on doesn't automatically help. Add all-to-all routing traffic and load imbalance, and the honest summary is: MoE saves compute, not memory, and it's a training-efficiency win more than a decode-latency win.

**50. Walk me through the cost-per-token mental model.**
$\text{cost} \approx (\text{weight-bytes} + \text{KV-bytes-for-step}) / \text{HBM-bandwidth} + \text{compute-overhead}$. For a 70B fp16 model sharded across 2× H100 (TP=2): each GPU reads ~70 GB at ~3 TB/s ≈ 23 ms per decode step → ~43 tok/s/request floor. Quantize to W4A16 (35 GB total → fits on one H100): ~12 ms → ~80 tok/s. Add speculation with 0.6 acceptance: another ~2x. This is how to reason quantitatively about latency budgets.

> **Saying it out loud.** The model is: time per token equals bytes you must read divided by memory bandwidth, plus overhead. So for 70B in fp16 across two H100s, each GPU reads about 70 gigabytes at roughly 3 terabytes a second, which is 23 milliseconds and about 43 tokens a second per request. Quantise to 4-bit weights and you're reading a quarter of that, so around 12 milliseconds and 80 tokens a second, now on a single GPU. Layer speculation on top with a decent acceptance rate and you roughly double again. The reason to memorise this chain is that it turns vague latency questions into arithmetic you can do out loud in 30 seconds.

---

## J. 2024-2025 frontier inference

**51. What's MLA (Multi-head Latent Attention)?** (DeepSeek-V2/V3)
Project K and V into a low-rank latent (dim $d_c \ll n_h \cdot d_h$); cache only the latent. At attention time, project back up. KV cache shrinks $\sim 10\times$ vs MHA, $\sim 3$–$4\times$ vs GQA-8. Subtlety: RoPE doesn't commute with the down-projection — DeepSeek splits each head into a "RoPE part" (uncompressed) and a "no-RoPE part" (compressed).

> **Saying it out loud.** Multi-head latent attention compresses what you cache. Instead of storing full keys and values per head, DeepSeek projects them down into a small shared latent vector, caches only that, and projects back up at attention time. The cache shrinks roughly 10x versus full multi-head and 3 to 4x versus GQA with 8 groups. The subtlety worth naming, because it shows you've read the paper: RoPE doesn't commute with the down-projection, so each head is split into a small RoPE part that's cached uncompressed and a larger no-RoPE part that's compressed.

**52. Why does MLA matter for serving cost?**
At long context (128K+), KV cache dominates memory. MLA shrinks KV enough that DeepSeek-V3 (671B params, MoE) is *cheaper to serve per token* than smaller GQA-only models. A practical disproof of "bigger model = always more expensive."

> **Saying it out loud.** Because at long context the KV cache, not the weights, is what limits how many users fit on a GPU. Weights are a fixed cost you amortise across the whole batch; the cache is per-user and grows with every token. So shrinking KV by 10x directly multiplies your concurrency, and concurrency is what sets cost per token. The striking consequence is that DeepSeek-V3, at 671B total parameters, can be cheaper to serve per token than much smaller models with conventional attention — which is a concrete counterexample to "bigger model always costs more.\"

**53. What's chunked prefill?**
Long prompts have two problems: bad TTFT and they block decode for other batched requests. Split prefill into chunks of ~512–2K tokens, interleave with decode steps. Standard in vLLM 0.6+, SGLang, TensorRT-LLM. Improves TTFT and decode throughput simultaneously.

> **Saying it out loud.** Chunked prefill splits a long prompt into pieces of a few hundred to a couple of thousand tokens, processed across multiple scheduler steps and interleaved with everyone else's decode. It fixes head-of-line blocking, where one 100K-token prompt stalls every other user's token stream for hundreds of milliseconds. It also lets the scheduler top up each step's token budget with prefill work, which keeps a decode-only batch from wasting compute. It's standard in vLLM, SGLang and TensorRT-LLM now, and it improves TTFT and decode throughput at the same time, which is unusual.

**54. What's disaggregated serving (DistServe / Mooncake)?**
Prefill is compute-bound, decode is memory-bound — using same GPUs for both is wasteful. Disaggregation: separate prefill workers (batched, large compute) from decode workers (smaller batches, memory locality). KV transferred between them. 2-4× higher goodput at fixed SLO. Mooncake = Moonshot AI's Kimi serving system.

> **Saying it out loud.** Disaggregation takes the prefill-versus-decode split seriously and puts them on separate machines. Prefill is compute-bound and wants big batches; decode is memory-bound and wants many concurrent sequences — running both on the same GPU means each interferes with the other's ideal configuration. So you run prefill workers, ship the resulting KV cache over the network to decode workers, and scale the two pools independently to match your actual traffic. The reported gain is 2 to 4 times the goodput at a fixed latency SLO. The cost is the KV transfer, which is why it needs fast interconnect and only pays off at scale.

**55. EAGLE vs Medusa — what's the difference?**
Both are speculative-decoding variants without a separate draft model. **Medusa**: extra "Medusa heads" predict next-2, next-3, etc. tokens from the target's last hidden state. Simple. **EAGLE**: trains a small "feature predictor" using the target's penultimate-layer features. Higher acceptance (60-80%) than vanilla draft models because it shares representation. EAGLE-2/3 push to 70-85%.

> **Saying it out loud.** Both drop the separate draft model, but they draft differently. Medusa bolts extra heads onto the target that predict the token after next, and the one after that, directly from the final hidden state — dead simple, but each head is guessing in isolation so accuracy falls off fast. EAGLE instead trains a small autoregressive predictor over the target's penultimate-layer *features*, so the draft is sequential and conditioned on rich representation, which pushes acceptance to 60-80% and 70-85% for EAGLE-2 and 3. The tradeoff is that EAGLE is more to train and integrate; Medusa you can bolt on in an afternoon.

**56. What's StreamingLLM / attention sinks?**
Empirical observation: most attention concentrates on first ~4 tokens (the "attention sink") + recent tokens. Drop the middle of the KV cache; keep early + recent. Lossy but works for very long contexts. Used when MLA + GQA still aren't enough.

> **Saying it out loud.** StreamingLLM comes from a surprising empirical finding: models dump a huge amount of attention onto the first few tokens of the sequence regardless of what those tokens are. The explanation is that softmax has to sum to one, so when a head has nothing relevant to attend to it needs somewhere to park the mass, and position zero becomes that sink. So you can throw away the middle of the KV cache and keep just the first four tokens plus a recent window, and the model stays coherent indefinitely — whereas dropping the sink tokens makes it collapse. It's lossy: anything in the discarded middle is genuinely gone, so it's for streaming, not for long-context recall.

**57. KV cache quantization — what schemes?**
KIVI: per-channel for K, per-token for V (matches their distribution shapes). KVQuant: outlier-aware. FP8 KV: easiest path on H100; ~2× cache reduction at minimal quality loss. Combined with MLA/GQA, KV memory becomes manageable even at 128K+.

> **Saying it out loud.** The main ones are KIVI, KVQuant, and plain FP8. KIVI's insight is that keys and values need different treatment — keys have persistent outlier channels so you quantise per channel, values don't so you quantise per token. KVQuant is outlier-aware in a similar spirit. And FP8 KV is the easy path on H100 because it's a native format, giving you 2x cache reduction at essentially no quality cost and no extra compute. Stack any of those with GQA or MLA and 128K context stops being a memory crisis.

**58. FP4/FP6 on Blackwell — what changes?**
NVIDIA Blackwell natively supports FP4 matmul. Inference at $\sim 4\times$ FP16 throughput. Production sweet spot: FP4 weights + FP8 activations. Aggressive: FP4 + FP4 (needs careful calibration). Format variants like NVFP4/MXFP4 use micro-block scales for better numerics. Will dominate 2025+ serving.

> **Saying it out loud.** Blackwell adds native FP4 matmul, so 4-bit stops being a memory-only trick and becomes a compute win too — roughly 4x fp16 throughput. The production sweet spot is FP4 weights with FP8 activations, since activations still have the outlier problem and want the extra dynamic range. Going FP4 on both sides is possible but needs careful calibration. The detail that matters is micro-block scaling: formats like NVFP4 and MXFP4 give each small group of values its own scale factor, which is what makes 4 bits survivable at all.

**59. Continuous-batching scheduler — what does it actually decide?**
At each step: which requests to admit (memory permitting), which to pause / preempt, which to advance. Policies: FCFS (simple), priority by length (better tail latency), memory-aware admission, preemption with KV swap to CPU. The scheduler is the load-bearing piece in production serving.

> **Saying it out loud.** Every step the scheduler decides three things: which waiting requests to admit, which running ones to advance, and which to preempt when memory runs short. Admission is memory-aware — you can only admit if you can guarantee KV blocks for the next steps, or you risk a mid-generation eviction. Preemption means either recomputing that request's prefill later or swapping its KV to host memory. Policy choice shows up directly in tail latency: pure first-come-first-served is simple but lets one enormous request wreck p99. It's the least glamorous and most load-bearing component in a serving stack.

**60. Modern latency targets?**
Chat: TTFT < 500 ms, TPOT < 50 ms. Voice: TTFT < 200 ms, TPOT < 30 ms. Achieving these on 70B+ models requires the full stack: GQA/MLA + chunked prefill + continuous batching + speculative decoding + FP8/FP4.

> **Saying it out loud.** For chat, under 500 milliseconds to first token and under 50 per token after that — that's roughly the point where it feels responsive and text outpaces reading. Voice is tighter, around 200 and 30, because conversational turn-taking breaks down past a few hundred milliseconds. Hitting those on a 70B-plus model isn't one trick, it's the whole stack: GQA or MLA to shrink the cache, chunked prefill and prefix caching for TTFT, continuous batching and quantisation and speculation for TPOT. The framing to use: TTFT is a prefill problem and TPOT is a bandwidth problem, and you attack them with different tools.

**61. What's Ring Attention?**
For sequences too long to fit on one GPU. Distribute the sequence across GPUs; pass K/V slices in a ring while each GPU computes its query's attention against rotating K/V. Used to train + serve million-token models (Gemini 1.5 Pro and later). Cost: extra inter-GPU communication. Pairs with FlashAttention.

> **Saying it out loud.** Ring attention is for sequences too long to fit on one GPU at all. You shard the sequence across devices, and then each device holds its queries fixed while K/V blocks rotate around the ring, computing partial attention against each block as it arrives and combining with the online-softmax trick. Because the communication overlaps with the computation, it's nearly free if your interconnect keeps up. That's the mechanism behind million-token context windows, and the cost is that you need a lot of GPUs and fast links, so it's an infrastructure decision, not a kernel flag.

**62. Provider-level prompt caching (Anthropic, OpenAI)?**
Beyond batch-level prefix sharing: providers cache long prompt prefixes across requests / users for hours. Pricing-tier discounts for cache hits (e.g., 90% off cached input tokens). Critical for agentic workloads with repeating system prompts + tool definitions across thousands of requests.

> **Saying it out loud.** Provider-level caching is prefix caching that persists across requests and across time — minutes to hours — rather than just within a live batch. It exists because agentic workloads send the same enormous system prompt and tool definitions thousands of times, and recomputing that is pure waste for everyone. Providers pass the saving through as a large discount on cached input tokens, often around 90%. The engineering consequence for the user is real: put your stable content first and your variable content last, because the cache is a prefix match and one changed token near the top invalidates everything after it.

**63. The 2025 inference stack in one sentence?**
GQA or MLA + chunked prefill + continuous batching with memory-aware scheduler + speculative decoding (EAGLE-style) + FP8 weights/activations (FP4 on Blackwell) + KV quantization for long context + disaggregated prefill/decode at scale.

> **Saying it out loud.** One sentence: GQA or MLA to shrink the cache, chunked prefill and a memory-aware continuous-batching scheduler to keep the GPU busy, EAGLE-style speculation for latency, FP8 or FP4 weights, KV quantisation for long context, and disaggregated prefill and decode once you're big enough for it to pay. What ties it together is that almost every item is about moving fewer bytes or moving them fewer times, because decode is bandwidth-bound. If I had to pick the two with the best return per unit of effort, it'd be prefix caching and continuous batching.

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
