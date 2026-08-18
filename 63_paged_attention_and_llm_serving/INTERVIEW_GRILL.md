# Paged Attention & LLM Serving — Interview Grill

> 50 questions on KV cache mechanics, PagedAttention, continuous batching, prefix caching, serving systems. Drill until you can answer 35+ cold.

---

## A. KV cache fundamentals

**1. Why does autoregressive decoding need a KV cache?**
Each new token attends to all previous tokens. Without caching, you re-compute K and V for the entire prefix every step → quadratic in sequence length.

> **Saying it out loud.** Without a KV cache, generating token number 1,000 means recomputing the keys and values for all 999 tokens before it — and you'd do that again at token 1,001. That makes generating a sequence quadratic in its length, for work you already did. So you keep every past key and value in GPU memory, and each new step computes exactly one new K and V and attends against the stored cache. That turns decoding from quadratic into linear, and the price is that the cache becomes your dominant memory consumer — for a 70B model at 8K context it's about 2.6 gigabytes per concurrent request.

**2. KV cache memory per token?**
$2 \cdot L \cdot H \cdot d_{\mathrm{head}} \cdot \mathrm{bytes}$. (2 for K and V; $L$ layers; $H$ = number of **KV** heads.) With GQA, $H$ is the KV-head count, not the query-head count — using query heads here overestimates the cache by the group ratio.

> **Saying it out loud.** Two, times layers, times KV heads, times head dimension, times bytes per element — and I'd say the two out loud because it's the K and the V. The one place people get this wrong is the head count: with GQA you use the number of *KV* heads, not query heads, otherwise you overestimate by the group ratio, which for Llama 70B is a factor of eight. Everything about serving economics follows from this number, because it's what decides how many requests fit on a GPU at once, and batch size is what throughput actually is.

**3. Llama 2 70B KV cache per token?**
With GQA-8 (8 KV heads, 128 head dim, 80 layers, BF16): $2 \times 8 \times 128 \times 80 \times 2 \approx 327$ KB per token. For 8K context: ~2.6 GB per request. Without GQA (full MHA, 64 heads), it would be ~2.6 MB per token — GQA-8 saves $8\times$.

> **Saying it out loud.** For Llama 2 70B with GQA-8 it's about 327 kilobytes per token: 2 for K and V, times 8 KV heads, times 128 head dimension, times 80 layers, times 2 bytes for BF16. At 8K context that's roughly 2.6 gigabytes per single request. The comparison that makes it land is what full multi-head attention would cost — 64 KV heads instead of 8, so 2.6 megabytes per token and over 20 gigabytes per request, which would mean you could serve about three concurrent users on an 80-gig card. GQA is the difference between a viable product and an unviable one.

**4. How does GQA reduce KV cache?**
Fewer KV heads (shared across query head groups). KV cache shrinks by the group factor. GQA-8 with 64 query heads → 8× smaller cache.

> **Saying it out loud.** GQA lets a group of query heads share one key-value head, so you store the cache once per group instead of once per head. In Llama 70B that's 64 query heads mapped onto 8 KV heads, so the cache is 8 times smaller — and since the attention math still gives each query head its own projection, you keep most of the expressiveness. It costs essentially nothing at training time; it's an inference decision. And the payoff isn't really 'less memory', it's more concurrent requests per GPU, which is the same thing as cheaper tokens.

**5. Why is MQA aggressive?**
One KV head shared by all queries. Smallest possible cache; some quality loss.

> **Saying it out loud.** MQA is GQA taken to the limit — one key-value head shared by every query head, so the cache shrinks by the full head count, 64x in a 64-head model. That's the smallest cache you can have without changing the algorithm. The reason nobody ships it anymore is that quality measurably drops on harder tasks: every head is now reading the same compressed view of the past, so heads can't specialise in what they attend to. GQA-8 recovers essentially all of that quality while keeping 8x of the savings, which is why the industry settled where it did.

**6. MLA reduction strategy?**
Compress KV through low-rank latent projection. Stores compressed latent + reconstructs at attention time. Even smaller than MQA.

> **Saying it out loud.** MLA compresses keys and values into a single shared low-rank latent vector, caches that instead, and projects back up at attention time. So rather than reducing the *number* of KV heads, it reduces the *dimensionality* of what you store — a different axis entirely, and it composes with the intuition rather than competing with it. DeepSeek reports roughly 10x versus multi-head attention, which beats GQA-8 while claiming better quality than MQA. The catch is complexity: it interacts badly with RoPE, so you need a separate small rotary component bolted on, and that implementation cost is real.

---

## B. The fragmentation problem

**7. What's the fragmentation problem in naive KV allocation?**
Pre-allocate max-length contiguous block per request. Most requests don't reach max length → wasted memory.

> **Saying it out loud.** The naive way to allocate a KV cache is to reserve a contiguous chunk big enough for the maximum possible output length, per request, up front. The problem is that almost no request reaches max length — someone sets max_tokens to 2,048 and the model stops after 100 — so you've reserved twenty times what you needed and it sits idle for the whole request. Multiply that by every concurrent request and you're running at a fraction of the batch size the hardware could support. The vLLM paper measured this at 60 to 80 percent of KV memory wasted in real deployments, which is the number to quote.

**8. External fragmentation?**
Free blocks scattered; can't fit a new max-length request even though total free ≥ needed.

> **Saying it out loud.** External fragmentation is when you have plenty of free memory in total but not in one piece. Requests finish at unpredictable times and leave holes of different sizes, so you might have 20 gigabytes free and still be unable to admit a request that needs a contiguous 4, because the largest hole is 3. It's exactly the malloc problem, and it's why operating systems went to paging in the first place — which is precisely the analogy PagedAttention borrows.

**9. Internal fragmentation?**
Block allocated to a request but most of it unused (request finished early or never reached max length).

> **Saying it out loud.** Internal fragmentation is waste *inside* an allocation you already made. You reserved space for 2,048 tokens, the request generated 100 and finished, and the other 1,948 slots were yours the whole time and never used. Unlike external fragmentation you can't fix this with compaction, because the memory isn't free — it's reserved by a live request that might still need it. The only real fix is to stop over-reserving, which means allocating in small increments as the sequence actually grows, and that's the core of what PagedAttention does.

**10. Why is contiguous allocation hard for LLM serving?**
Variable lengths; arbitrary completion times; memory shape demand changes per step.

> **Saying it out loud.** Because the shape of the demand changes every single step and you can't predict it. Prompts vary from ten tokens to a hundred thousand, output lengths are unknown until the model emits a stop token, requests arrive and finish at arbitrary times, and every live request grows by one token per step. Contiguous allocation needs you to know the size in advance, and you fundamentally don't. That mismatch is the whole reason the naive approach wastes most of your memory, and it's why the OS paging analogy is so apt — operating systems solved this exact problem for exactly this reason.

**11. How much memory is wasted with naive allocation?**
PagedAttention paper: 60–80% of KV memory wasted in production setups.

> **Saying it out loud.** Sixty to eighty percent, from the vLLM paper's measurements of production systems. Say the number, because it's what makes the problem sound worth solving — you're throwing away two thirds of the most expensive resource on the machine. And the consequence isn't just wasted memory, it's directly reduced batch size, which for bandwidth-bound decode is the same as directly reduced throughput. PagedAttention takes utilisation from around 20 or 30 percent up to the high nineties, which is why it translated into a claimed 2 to 4x throughput improvement.

---

## C. PagedAttention

**In plain language.** PagedAttention is an idea borrowed wholesale from operating systems. Instead of giving each request one big contiguous slab of KV cache, you chop memory into small fixed-size blocks and hand them out one at a time as a sequence grows, keeping a little per-request lookup table that says which physical block holds which stretch of tokens. That lookup table is the block table, and it plays exactly the role a page table plays in virtual memory.

**12. What's the core PagedAttention idea?**
Apply OS-style paging to KV cache. Logical KV addresses → physical KV blocks via a per-request block table.

> **Saying it out loud.** PagedAttention takes virtual memory and applies it to the KV cache. Instead of one contiguous slab per request, you carve GPU memory into small fixed-size blocks — typically 16 tokens' worth — and hand them out one at a time as a sequence grows. Each request keeps a block table mapping its logical token positions to whatever physical blocks it happens to own, so the memory a request sees is contiguous and the memory it actually occupies is scattered. That kills both kinds of fragmentation at once: no over-reservation because you allocate as you go, and no external fragmentation because every block is the same size and therefore interchangeable.

**13. What's a block?**
Fixed-size chunk of KV cache (typically 16 tokens). The unit of allocation.

> **Saying it out loud.** A block is the unit of allocation — a fixed-size chunk holding the keys and values for a fixed number of token positions, usually 16, across all layers and heads. Everything in the system is expressed in blocks: you allocate one, free one, share one, reference-count one. The reason a fixed size matters is that it makes every free block interchangeable, which is exactly what eliminates external fragmentation. And the only waste left is inside the last partially-filled block of a sequence, which is at most 15 tokens — call it under 4% instead of 60 to 80%.

**14. What's a block table?**
Per-request mapping: logical token positions → physical block indices. Like an OS page table.

> **Saying it out loud.** The block table is a per-request array that maps logical token positions to physical block indices — token 0 through 15 live in physical block 42, tokens 16 through 31 in block 7, and so on. It's exactly an OS page table, and saying that comparison out loud is the fastest way to make the whole design click for an interviewer. It's what buys you the indirection: the attention kernel walks the table to find where each stretch of KV actually lives, so blocks can be scattered, shared between requests, or freed independently without the request ever knowing.

**15. How does PagedAttention reduce fragmentation?**
Allocates per-block instead of per-request. Frees blocks back to a pool when finished. ~96% utilization vs ~30% naive.

> **Saying it out loud.** Two ways. Internal fragmentation goes away because you allocate blocks on demand as the sequence grows, instead of reserving for the worst case up front — so the only waste is the tail end of the last block, at most 15 tokens. External fragmentation goes away because all blocks are the same size, so any free block fits any request and there's no such thing as a hole that's the wrong shape. The numbers vLLM reported are the ones to have ready: from roughly 20 to 30 percent memory utilisation up to about 96 percent, which translated into 2 to 4 times the throughput because higher utilisation directly means a larger batch.

**16. Memory access pattern in PagedAttention attention kernel?**
Block-table indirection: gather KV from non-contiguous blocks. Custom kernel handles the indirection efficiently.

> **Saying it out loud.** The kernel can't just stream through contiguous memory anymore — for each request it reads the block table and gathers the KV from scattered physical blocks. That's why PagedAttention needs a custom attention kernel; you can't drop it on top of a standard implementation. The cost is one extra level of indirection per block, which sounds bad but is nearly free in practice because the access is still coalesced *within* a block, and 16 tokens is enough to amortise the lookup. That's the tradeoff to name: a small, predictable kernel overhead in exchange for roughly tripling your effective memory.

**17. Block size trade-off?**
Larger blocks: less indirection overhead. Smaller blocks: less internal fragmentation. 16 is a common sweet spot.

> **Saying it out loud.** Block size trades indirection overhead against internal fragmentation, and it's a genuine U-shape. Big blocks mean fewer block-table lookups and better memory coalescing, but more waste in the last partially-filled block and coarser sharing granularity for prefix caching. Small blocks waste almost nothing and share very precisely, but you pay more lookups and the kernel gets less efficient. Sixteen tokens is where most systems land, which is a nice number to quote because it shows you've looked at real configs rather than reasoned in the abstract.

---

## D. Prefix caching / sharing

**18. What's prefix caching?**
Multiple requests sharing the same prompt prefix can share KV blocks. Only the divergent suffix needs separate blocks.

> **Saying it out loud.** Prefix caching is what happens when you notice that two requests starting with the same 2,000-token system prompt have identical KV for those 2,000 tokens — so why compute or store it twice? With a block table, sharing is almost free: you just point both requests' tables at the same physical blocks. Only the divergent part needs private blocks. On a chat product where every request carries the same long system prompt, that's a large fraction of the prefill you simply never do, so time-to-first-token drops dramatically and memory pressure drops with it.

**19. How is sharing implemented in PagedAttention?**
Block table entries can point to shared physical blocks. Reference counting determines when a block can be freed.

> **Saying it out loud.** It falls out of the block table almost for free: two requests' tables just point at the same physical block indices, and the block pool keeps a reference count so a block is only returned when the last user releases it. That's straight out of operating systems — shared pages with refcounts. The other piece you need is copy-on-write: if one request would ever write into a shared block, it gets its own copy first, which happens at the point where two sequences diverge inside a partially-filled block. It's a nice answer because it shows the design's real virtue, which is that sharing wasn't an extra feature, it was a consequence of the indirection.

**20. When does prefix caching matter most?**
Repeated long system prompts (chat assistants, agents). Tool-use templates. Few-shot examples reused across queries.

> **Saying it out loud.** It matters most where the same long prefix shows up over and over. Chat assistants with a big system prompt, agents with long tool-definition preambles, few-shot templates where only the final example changes, and multi-turn conversations where turn N's prefix is turn N minus 1's entire history. In those workloads the shared portion can be the overwhelming majority of the tokens, so you're skipping most of the prefill — routinely a 5 to 10x cut in time-to-first-token. Where it doesn't help is a workload of unrelated one-shot prompts, and there the cache is pure overhead competing for memory you'd rather spend on batch size.

**21. What's "RadixAttention" (SGLang)?**
Generalizes prefix caching to arbitrary subsequences via a radix tree of KV blocks. Captures shared patterns beyond just prefixes.

> **Saying it out loud.** RadixAttention is SGLang's generalisation from prefixes to arbitrary shared subtrees. Instead of a flat hash of prefixes, you keep a radix tree of the KV blocks, so any request that shares *any* path through the tree reuses it — which matters a lot for agentic workloads, where one conversation branches into several tool calls that all share the same trunk. Think of it as caching a tree of conversations rather than a list of prefixes. The costs are bookkeeping complexity and an eviction policy that has to be tree-aware, since evicting an interior node invalidates everything beneath it.

**22. What's a copy-on-write KV block?**
Shared block becomes per-request-private when one request would write into it. Standard OS-style technique.

> **Saying it out loud.** Copy-on-write means a shared block stays shared until somebody needs to modify it, at which point that request gets its own private copy and the others carry on with the original. In serving, this comes up when two sequences share a prefix that ends mid-block — say they diverge at token 20 inside a 16-token block boundary — so the block holding the divergence has to be duplicated while all the earlier full blocks stay shared. It's the same technique as fork in an operating system, and the payoff is that parallel sampling or beam search over one prompt costs one copy of the prompt's KV plus a few blocks, not N copies.

---

## E. Continuous batching

**23. What's static batching?**
Form a batch; run all to completion together. Faster requests wait for slowest.

> **Saying it out loud.** Static batching means you collect a batch of requests, run them all together, and nobody leaves until everybody's done. The problem is that generation lengths vary wildly — one request finishes in 10 tokens and another runs to 2,000 — and the short one's slot sits there computing padding until the batch completes. So your effective utilisation is roughly the average length over the maximum length, which in a real workload can be under 20%. That's the failure mode: you've got a full batch on paper and mostly idle work in practice.

**24. What's continuous batching (a.k.a. inflight batching)?**
At each step, swap finished requests out and admit new ones. No request waits for others.

> **Saying it out loud.** Continuous batching makes the batch composition a per-step decision instead of a per-batch one. Every forward step, requests that hit their stop token are evicted and waiting requests are admitted into the freed slots, so the GPU is always running a full batch of live work and nobody waits on anybody else's completion. It's the single biggest throughput win in modern serving — often 2 to 3x over static batching before you've changed anything about the model. It pairs naturally with PagedAttention, because admitting a new request mid-flight requires being able to hand it memory in small increments.

**25. Why does continuous batching help throughput?**
Eliminates idle GPU time waiting for the slowest request. Higher GPU utilization.

> **Saying it out loud.** Because it removes the idle slots. With static batching, a slot occupied by a request that finished 400 steps ago is doing nothing while the batch waits for the longest generation, and decode is bandwidth-bound, so what you're wasting is the one resource that determines your throughput. Continuous batching keeps every slot filled with real work, which pushes the effective batch size up, and a larger batch amortises the weight reads across more tokens — you drag the weights out of HBM once and serve 64 tokens instead of 8. That's why the win is so large: it's fixing utilisation *and* arithmetic intensity at the same time.

**26. What does continuous batching require from the kernel?**
Variable per-step batch composition; per-request length tracking; flexible scheduling.

> **Saying it out loud.** The kernel has to tolerate a batch whose composition changes every step and whose members are all at different sequence positions. So you can't assume a single shared sequence length; you need per-request length tracking, ragged batching, and attention that handles each request against its own KV of its own size — which is exactly what the block-table indirection provides. You also need the scheduler to run at every iteration and the memory manager to allocate and free at block granularity mid-flight. That's why continuous batching and PagedAttention showed up together: neither is much use without the other.

**27. Iteration-level vs request-level scheduling?**
Iteration-level: schedule decisions every forward step. Request-level: at request boundaries. Continuous batching is iteration-level.

> **Saying it out loud.** Request-level scheduling makes decisions when a request starts or ends; iteration-level makes them at every single forward pass. Continuous batching is iteration-level by definition — that's the whole idea, that admission and eviction happen per step rather than per batch. The tradeoff is scheduling overhead: you're now running scheduler logic thousands of times a second, so it has to be cheap, and that CPU work can become the bottleneck on small models where each step is only a millisecond or two. That's a real failure mode in production — a GPU waiting on Python scheduling.

---

## F. Prefill vs decode

**28. Prefill phase — what is it?**
Process the entire input prompt in parallel; populate KV cache. Compute-bound.

> **Saying it out loud.** Prefill is where you process the entire prompt at once, running every token through the model in parallel to populate the KV cache and produce the first output token. Because you have hundreds or thousands of tokens to multiply against the weights simultaneously, it's a big dense matmul with high arithmetic intensity — so it's compute-bound, and it's what determines your time-to-first-token. The other thing to name is that prefill's attention cost is quadratic in prompt length, so a 100K-token prompt is a genuinely expensive operation that can block everything else on the GPU.

**29. Decode phase — what is it?**
Generate one token at a time. Memory-bound (each step reads entire KV cache).

> **Saying it out loud.** Decode is the one-token-at-a-time phase, and it's memory-bandwidth-bound. Every single step you read the entire weight matrix out of HBM and the entire KV cache, and you use all of that to produce one token — so at batch one the arithmetic intensity is around 1 FLOP per byte, against an H100 that needs roughly 295 to saturate its BF16 math units. That means the tensor cores are idle something like 99% of the time and you're purely waiting on memory. That single fact explains basically every inference optimisation there is: batching, quantization, GQA and speculative decoding are all ways of getting more useful work out of each byte you move.

**30. Why are they characterized differently?**
Prefill has many tokens to process in parallel → high arithmetic intensity. Decode has one token per step → arithmetic intensity tiny → bandwidth-limited.

> **Saying it out loud.** It comes down to arithmetic intensity — FLOPs performed per byte moved. Prefill has many tokens sharing the same weight read, so you get hundreds of FLOPs per byte and the GPU's math units are the limit. Decode has one token per weight read, so it's about 1 FLOP per byte at batch one, and the H100's balance point in BF16 is around 295 — you're off by more than two orders of magnitude. Consequently they respond to completely different medicine: prefill wants better kernels and more FLOPs, decode wants bigger batches, smaller weights and fewer bytes. Confusing the two is why people are surprised that a faster GPU doesn't speed up their generation.

**31. Why do servers separate prefill and decode pools?**
Different bottlenecks → different optimal hardware/configurations. Disaggregated architectures (DistServe) split them.

> **Saying it out loud.** Because they want opposite things from the hardware and they interfere with each other on shared hardware. Prefill is compute-bound and bursty; decode is bandwidth-bound and steady. If you run them together naively, one long prefill blocks every in-flight decode for hundreds of milliseconds, so your inter-token latency spikes for everyone — which is the classic head-of-line blocking failure. Disaggregating them, as DistServe does, lets you size and scale each pool separately and hit both a time-to-first-token SLO and an inter-token-latency SLO at once. The cost you pay is shipping the KV cache from the prefill machine to the decode machine over the network, which is a lot of bytes.

**32. What's chunked prefill?**
Break a long prefill into smaller chunks; interleave with decode steps. Prevents long-prefill requests from blocking decode for short ones.

> **Saying it out loud.** Chunked prefill slices a long prompt into pieces — say 512 tokens at a time — and interleaves those pieces with ongoing decode steps instead of running the whole prefill in one blocking pass. That fixes head-of-line blocking, so one user's 100K-token document doesn't freeze everyone else's token stream. There's also a subtler win: a chunk of prefill is compute-heavy and a decode step is bandwidth-heavy, so running them in the same batch uses both parts of the machine at once and raises overall utilisation. The tradeoff is that the long request's own time-to-first-token gets a bit worse — you're deliberately trading one user's latency for everyone else's.

---

## G. Speculative decoding

**33. What's speculative decoding?**
Small "draft" model generates $K$ tokens; big model verifies them in parallel; accept run-length until first rejection.

> **Saying it out loud.** Speculative decoding gets several tokens out of a single pass through the big model. A small draft model — maybe a 1B alongside a 70B — generates K candidate tokens cheaply and sequentially, then the big model runs one forward pass that scores all K positions at once, and you keep the longest prefix where the big model agrees. The reason it's not cheating is the rejection-sampling step: done properly, the output distribution is provably identical to sampling from the big model directly, so it's a pure latency win with no quality cost. Typical numbers are 60 to 80 percent acceptance and 2 to 3 times faster decoding.

**34. Why is verification fast?**
Big model processes $K$ candidates in a single batched forward pass — much cheaper than $K$ sequential decodes.

> **Saying it out loud.** Because verification is one forward pass, not K, and decode is bandwidth-bound. Scoring K candidate positions at once is the same shape as a tiny prefill — you read the weights out of HBM exactly once and do K times the arithmetic on them, and since you were only using about 1 FLOP per byte to begin with, that extra arithmetic is free. So verifying 5 tokens costs almost exactly what generating 1 token costs. That's the whole trick, and it's why speculative decoding stops helping at large batch sizes: once you're batching heavily you're already using the bandwidth well and there's no idle compute to spend.

**35. Acceptance ratio — what determines it?**
How well draft model approximates big model. Smaller draft + similar architecture often gets 60-80% acceptance.

> **Saying it out loud.** Acceptance is really asking how often the small model agrees with the big one, so it's driven by how well the draft approximates the target. Same tokenizer and same family matters a lot — a distilled or same-lineage draft model does far better than an unrelated one of the same size. Content matters too: acceptance is very high on predictable text like boilerplate code or formatting and drops on genuinely uncertain content, which is exactly where the big model earns its keep. Typical is 60 to 80 percent, and there's a sweet spot in K — go too long and you're generating draft tokens that will almost certainly be rejected, so the draft cost stops paying for itself.

**36. Self-speculative variants?**
Same model used for both draft and verify, via extra prediction heads (Medusa) or by skipping layers for the draft pass (self-speculative / LayerSkip). No need for separate draft model.

> **Saying it out loud.** Self-speculative methods drop the separate draft model and get the candidates out of the target model itself. Medusa bolts extra prediction heads onto the final hidden state, so one forward pass emits guesses for the next several positions at once. Layer-skipping approaches run a subset of layers for the draft and the full stack for the verification. The advantage is real: no second model to load, no second KV cache, no memory or ops burden — which matters a lot in production where the draft model's footprint directly reduces your batch size. The cost is that you have to train those heads, so it's not something you can bolt onto an arbitrary checkpoint for free.

**37. EAGLE?**
Eagle: draft model trained to predict from big model's hidden states. Higher acceptance than independent drafts.

> **Saying it out loud.** EAGLE's insight is that drafting from tokens throws away information — the target model's hidden states carry far more than the sampled token does. So EAGLE trains a small autoregressive head that predicts the *next hidden state* given the current one plus the token, and drafts from that. Because it's predicting in feature space rather than token space, it tracks the target model much more closely, and acceptance rates go well above what an independent draft model achieves — EAGLE-2 and 3 push the reported speedups past 3x. It's the current best-in-class approach, and the tradeoff is the usual one: you have to train it against the specific target model.

---

## H. Quantization for inference

**38. Why quantize at inference?**
Smaller weights → less memory bandwidth → faster decode. Plus more cache fits in GPU.

> **Saying it out loud.** Because decode is bandwidth-bound, quantization is close to a direct speedup rather than just a memory saving. Every token requires reading all the weights out of HBM, so halving the bytes roughly halves the time — INT8 is about 2x, INT4 about 4x on the weight-reading part. And the second-order effect is often bigger: smaller weights leave more room for KV cache, which means a larger batch, which means better arithmetic intensity. So you get the win twice, and that's the framing that scores — it's a bandwidth optimisation that happens to look like a memory optimisation.

**39. INT8 / INT4 weights?**
INT8: 2× smaller than FP16. INT4: 4× smaller. Quality loss with proper calibration is small.

> **Saying it out loud.** INT8 halves your bytes versus FP16 and is essentially free — with per-channel scales and decent calibration the quality loss is not measurable on standard evals. INT4 quarters them and is where it gets interesting: you need group-wise scales, typically one per 128 weights, and an outlier-aware method like GPTQ or AWQ, but done well the degradation is small. The important asymmetry to name is that weights quantize much more gracefully than activations, which is why the common production recipe is INT4 weights with BF16 activations rather than pushing both down.

**40. Common quantization schemes?**
GPTQ (post-hoc weight quantization minimizing reconstruction error), AWQ (activation-aware weights), SmoothQuant (migrates outlier difficulty from activations into weights via per-channel scaling).

> **Saying it out loud.** GPTQ quantizes weights layer by layer, using second-order information to decide the rounding that minimises the error in that layer's output — accurate but a slow, calibration-heavy process. AWQ starts from the observation that a small fraction of weight channels matter disproportionately because they see large activations, so it scales those channels to protect them; it's faster and often better at INT4. SmoothQuant attacks the other side entirely: activation outliers are what break INT8 activation quantization, so it migrates that difficulty into the weights with a per-channel scaling, since weights are much easier to quantize. The unifying theme in all three is outliers — that's what makes low-bit quantization hard, and each method is a different way of not letting a handful of large values set the scale for everything.

**41. KV cache quantization?**
Quantize K and V to INT8 or even INT4. Big cache savings; some quality loss.

> **Saying it out loud.** KV cache quantization stores the cached keys and values in INT8 or INT4 instead of BF16, which is 2 to 4x more concurrent requests on the same GPU — and at long context, where the cache dwarfs the weights, that's the single biggest lever you have. The practical detail is that keys and values behave differently: keys tend to have per-channel outliers, so they want per-channel scales, while values are better behaved and quantize per-token. The failure mode to name is that quality degradation grows with context length, because you're accumulating quantization error over more and more cached tokens, so 4-bit that looks fine at 4K can start to hurt at 64K.

**42. FP8 inference?**
Hopper/Blackwell GPUs natively support FP8. Faster matmul; 2× memory savings vs FP16.

> **Saying it out loud.** FP8 on Hopper and Blackwell gives you 2x memory savings against FP16 and roughly double the matmul throughput, with hardware support so there's no dequantization overhead. The reason it's more attractive than INT8 for a lot of cases is that it keeps an exponent, so it handles the outlier-heavy distributions in transformer activations far more gracefully than a fixed-point format does — you generally need much less calibration machinery. There are two variants, E4M3 with more precision for the forward pass and E5M2 with more range for gradients. It's now used for both inference and, at DeepSeek-V3 scale, pretraining.

---

## I. Production serving

**43. vLLM — what is it?**
Open-source LLM serving system. Implements PagedAttention, continuous batching, prefix caching, multiple quantization. Standard production choice.

> **Saying it out loud.** vLLM is the open-source serving system that came out of the PagedAttention paper, and it's the default choice for most people. What it gives you is PagedAttention for memory, continuous batching for utilisation, prefix caching, tensor parallelism, and support for most quantization formats, behind an OpenAI-compatible API. Its real advantage is flexibility and pace: new models and new techniques land there first. The reported headline was 2 to 4 times the throughput of prior systems like HuggingFace TGI at the time, and essentially all of that came from memory efficiency raising the achievable batch size.

**44. SGLang — what's the differentiation?**
RadixAttention for general subsequence sharing. Strong for agentic / tool-use workloads with many shared subtrees.

> **Saying it out loud.** SGLang's differentiator is RadixAttention — it keeps the KV cache in a radix tree so any shared subsequence gets reused, not just shared prefixes. That's a big deal for agent workloads, where one conversation forks into several tool calls that all share a long trunk, and for anything doing branching or parallel sampling. It also ships a structured-generation frontend for constrained decoding. The way to frame it in an interview: vLLM optimises the memory layout, SGLang additionally optimises *reuse across requests*, and which one wins depends entirely on how much your traffic actually shares.

**45. TensorRT-LLM?**
NVIDIA's optimized inference engine. Highest throughput on NVIDIA hardware; less flexible than vLLM.

> **Saying it out loud.** TensorRT-LLM is NVIDIA's own engine, and it wins on raw throughput on NVIDIA hardware because it compiles the model ahead of time into fused, hardware-specific kernels tuned for the exact GPU. The cost is exactly that ahead-of-time step: you build an engine for a specific model, precision, batch shape and GPU, so iteration is slow and flexibility is limited compared to vLLM's just-run-it approach. The honest framing is that it's the choice when you have one model, fixed and long-lived, and you're squeezing the last 20 or 30 percent out of a large fleet — and the wrong choice when you're changing models weekly.

**46. Common metrics for serving?**
TTFT (time to first token), ITL (inter-token latency), throughput (tokens/sec), goodput (effective throughput meeting SLO).

> **Saying it out loud.** Four numbers, and the important thing is that they're in tension. Time-to-first-token is how long the user waits before anything appears, which is dominated by prefill and queueing. Inter-token latency is how fast the text streams after that, which is decode. Throughput is total tokens per second across all users, which is what your cost per token is made of. And goodput is the one that actually matters commercially: throughput that meets your latency SLO, so tokens delivered too late don't count. Optimising raw throughput without goodput is the classic mistake — you crank the batch size, your numbers look great, and every individual user has a miserable experience.

**47. What's a goodput-vs-throughput trade-off?**
Higher batch size: better throughput, worse per-request latency. Pick based on SLO.

> **Saying it out loud.** The tradeoff is that bigger batches make the machine more efficient and each individual user slower. Since decode is bandwidth-bound, a bigger batch amortises the same weight read across more tokens, so throughput climbs almost linearly for a while — but every request now waits behind more work per step, so inter-token latency rises. If you tune for pure throughput you'll pick a batch size where a lot of requests miss the SLO, and those tokens are worthless even though your dashboard counts them. So the right objective is goodput: push the batch size up only until the marginal request starts failing your latency target.

**48. Why do tail latencies (p99) matter?**
User experience: most users want predictable response times. Long tails fail SLO even at good average.

> **Saying it out loud.** Because users experience the tail, not the mean. A system with a great average and a bad p99 feels unreliable — one request in a hundred hanging for ten seconds is what people remember and what breaks anything downstream with a timeout. And in LLM serving the tail is usually structural, not random: it's a long prefill blocking the decode queue, a request preempted when memory ran short, or a cache miss on a prefix everyone else hit. That's why chunked prefill and admission control exist — they're tail-latency fixes, and they usually cost you a little average throughput to buy predictability.

**49. What's "request preemption"?**
Pause a request mid-generation to admit a higher-priority one. Trade-off: better priority handling, more bookkeeping.

> **Saying it out loud.** Preemption is pausing a running request to free its resources for another one — either evicting its KV blocks and recomputing them later, or swapping them out to CPU memory. You need it because you admit requests optimistically, and if all of them grow their KV cache at once you can run out of GPU memory mid-generation with nowhere to put the next token. The alternatives are worse: refuse new work entirely, or crash. The tradeoff is that recompute-on-resume wastes prefill compute and swapping burns PCIe bandwidth, so preemption is a safety valve you want to fire rarely — if your metrics show frequent preemption, your admission control is too aggressive.

**50. Disaggregated serving?**
Separate prefill machines from decode machines. Each optimized for its workload. Lets you scale them independently.

> **Saying it out loud.** Disaggregated serving puts prefill and decode on separate machines because they're fundamentally different workloads — prefill is compute-bound and bursty, decode is bandwidth-bound and steady. Splitting them means one user's giant prompt can't block everyone else's token stream, and you can scale and configure each pool independently: more FLOPs and maybe higher tensor parallelism for prefill, more memory bandwidth and capacity for decode. DistServe and Mooncake are the reference systems. What you pay is the KV transfer between them — gigabytes per request over the network for long prompts — so it only pays off at scale and with a fast interconnect, and below that threshold chunked prefill on shared hardware is the simpler answer.

---

## Quick fire

**51.** *KV cache reason?* Avoid recomputing K, V for prefix.
**52.** *PagedAttention block size typical?* 16.
**53.** *Continuous batching admits new requests?* Every step.
**54.** *Prefill bottleneck?* Compute.
**55.** *Decode bottleneck?* Memory bandwidth.
**56.** *Speculative decoding acceptance typical?* 60-80%.
**57.** *vLLM main innovation?* PagedAttention + continuous batching.
**58.** *KV per token Llama 70B?* ~0.32 MB (≈327 KB) BF16 with GQA-8; ~2.6 MB if it were full MHA.
**59.** *GQA-8 cache savings?* 8×.
**60.** *RadixAttention generalization?* Arbitrary shared subtrees, not just prefix.

> **Saying it out loud (quick-fire bank).** These should come out in under five seconds each, no preamble. KV cache exists so you don't recompute the prefix. Block size is 16. Continuous batching admits new requests every step. Prefill is compute-bound, decode is memory-bandwidth-bound. Speculative acceptance is 60 to 80 percent. vLLM's innovation is PagedAttention plus continuous batching. Llama 70B's KV cache is about 327 kilobytes per token with GQA-8 — and be careful here, that's the GQA number; full multi-head attention would be eight times more, about 2.6 megabytes. RadixAttention generalises prefix sharing to arbitrary subtrees. Drill these until they're reflexes, because the fluency on the small facts is what buys you room to think on the hard questions.

---

## Self-grading

If you can't answer 1-15, you don't know KV cache basics. If you can't answer 16-35, you'll struggle on PagedAttention / serving system questions. If you can't answer 36-50, infra interview questions on LLM serving will go past you.

Aim for 40+/60 cold.
