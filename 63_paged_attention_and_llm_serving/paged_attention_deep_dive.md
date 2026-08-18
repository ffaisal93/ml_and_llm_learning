# Paged Attention Deep Dive

This note is meant to explain paged attention and modern LLM serving in descriptive terms rather than only as a short interview summary.

## 1. Why LLM Serving Feels Different From Training

Training and serving are often discussed together because both use the same model weights.

But they are bottlenecked by different things.

During training:
- you run forward and backward passes
- you process many tokens in parallel
- you pay heavy compute and activation-memory costs

During autoregressive serving:
- generation is sequential
- each new token depends on the previous context
- you repeatedly read historical state
- latency matters directly to the user

That changes the optimization problem.

Training often asks:

"How do I maximize throughput for huge batches while staying numerically stable?"

Serving often asks:

"How do I keep per-request latency acceptable while fitting many active requests into limited fast memory?"

That is why a technique that helps training may not help serving, and vice versa.

> **Saying it out loud.** Training and serving run the same weights and hit completely different walls. Training processes thousands of tokens at once, so it's a big dense matmul and you're compute-bound and fighting numerical stability. Serving generates one token at a time, so every step you drag the entire model out of memory to produce a single token — at batch one that's about 1 FLOP per byte moved, on a chip that wants around 295 to keep its math units busy. So the machine is idle over 99% of the time, waiting on memory. That's why a technique that helps training can be useless for serving: training optimisations save FLOPs, and serving optimisations save bytes.

## 2. Why KV Cache Exists in the First Place

Without KV caching, every decode step would recompute the full prefix through the transformer stack.

If the prompt is 1000 tokens long and you want 100 output tokens, the model would keep reprocessing almost the same prefix over and over.

That is wasteful because the old keys and values for previous tokens do not change.

KV caching stores them so that future decode steps only need to compute:
- the new query
- the new key
- the new value

and then attend over cached history.

This dramatically cuts repeated computation.

So KV cache is the first major serving optimization.

But it creates a new problem:

you now have a growing memory object attached to every active request.

> **Saying it out loud.** The KV cache exists because the keys and values of past tokens never change. If your prompt is 1,000 tokens and you want 100 more, without a cache you'd push nearly the same 1,000-token prefix through all 80 layers a hundred times over — enormous repeated work for an identical answer. So you store them once, and each decode step only computes the new query, key and value and attends over the stored history. That takes generation from quadratic to linear. But it hands you a new problem, which is the rest of this document: every active request now owns a big, growing block of GPU memory, and managing those is the hard part of serving.

## 3. Why KV Cache Becomes the Serving Bottleneck

**In plain language.** The phrase "memory-bound" has a precise meaning worth unpacking. For any operation you can ask how many arithmetic operations it does per byte it reads from memory — that ratio is called arithmetic intensity. Every GPU has a break-even point where its math throughput and its memory bandwidth are balanced; on an H100 in BF16 that's roughly 989 teraflops divided by 3.35 terabytes per second, or about 295 FLOPs per byte. Below that ratio you're waiting on memory no matter how fast the chip's math units are, and single-token decoding is far, far below it.

The cache grows with:
- sequence length
- number of layers
- number of KV heads
- head dimension
- batch size

So even when each decode step is only one token at a time, the model may need to read a large amount of historical KV state to produce that token.

That means serving can become limited by:
- VRAM capacity
- memory bandwidth
- cache layout efficiency

This is why a model can be computationally feasible and still be expensive to serve.

> **Saying it out loud.** The cache grows with sequence length, layers, KV heads, head dimension and batch size all multiplied together, which is why it gets big so fast. And the killer isn't just capacity, it's bandwidth: to generate one token you have to read the *entire* cache for that request, every layer, every step. So a model can be perfectly cheap in FLOPs and still be expensive to serve, because you're paying in bytes moved rather than math done. Concretely, Llama 70B with GQA-8 is about 327 kilobytes per token, so 2.6 gigabytes for one 8K-context request — and that's the number that caps how many users fit on a GPU, which is the number that sets your cost per token.

## 4. Why Naive KV Allocation Wastes So Much Memory

Imagine a server that gives each request one large contiguous KV buffer sized for a pessimistic maximum length.

That sounds simple, but it wastes memory in several ways.

### Over-Reservation

Many requests finish early or never use the full reserved space.

So the server holds memory for tokens that never arrive.

### External Fragmentation

When requests finish, they leave holes in memory.

You may have enough total free memory, but not enough in a large contiguous region for the next request.

### Variable-Length Workloads

Real serving traffic is messy:
- some requests are very short
- some continue for a long time
- some share prefixes
- some branch into multiple generations

A rigid contiguous allocator handles this badly.

This is why the memory problem is not just "KV cache is large."

It is also "KV cache is dynamic and irregular."

> **Saying it out loud.** The naive allocator reserves a contiguous buffer sized for the worst case, per request, and it wastes memory in two distinct ways. Over-reservation is internal waste — someone sets max_tokens to 2,048, the model stops at 100, and the other 1,948 slots sat reserved and idle the whole time. External fragmentation is the other half: requests finish at unpredictable times and leave holes, so you can have 20 gigabytes free and still be unable to admit a request needing a contiguous 4. The vLLM paper measured the combined waste at 60 to 80 percent of KV memory in real deployments. And the point that makes it click is that the problem isn't that the cache is large, it's that it's *dynamic and irregular* — you don't know any request's final size when you have to allocate for it.

## 5. Why Paging Is the Right Analogy

Operating systems faced a similar problem long ago.

Programs wanted the illusion of large contiguous memory, but physical memory was limited and fragmented.

The solution was virtual memory:
- split logical memory into fixed-size pages
- map them onto physical page frames
- keep a table that translates logical to physical locations

Paged KV caching borrows exactly this idea.

Instead of giving a request one giant contiguous KV region, the serving engine gives it:
- a sequence of fixed-size KV blocks
- a block table that maps logical order to physical blocks

This preserves the logical sequence while relaxing the physical layout.

That is the key idea.

> **Saying it out loud.** The paging analogy is exact, and saying it out loud is the fastest way to make the design obvious. Operating systems had the identical problem in the seventies: programs wanted the illusion of one big contiguous memory and physical RAM was limited and full of holes. The fix was virtual memory — chop it into fixed-size pages, scatter them wherever they fit, and keep a table translating logical addresses to physical ones. PagedAttention does exactly that for KV cache: fixed-size blocks, scattered physically, with a per-request block table preserving the logical order. Once every block is the same size, external fragmentation is impossible, because any free block fits any request.

## 6. What a Block Table Is Doing

**In plain language.** A block table is the small bookkeeping array that makes all of this work. Each request has one, and it just lists, in order, which physical chunks of GPU memory hold its keys and values: my first 16 tokens are in block 42, my next 16 are in block 7, and so on. Nothing about the model changes — the table is only a lookup that lets the attention kernel find scattered pieces and walk them in the right order.

A block table is conceptually simple.

For a given request, it answers:

"If I need the KV data for logical block 0, 1, 2, ... where do those blocks actually live in memory?"

This means the request can grow incrementally:
- first one block
- then another
- then another

without requiring one giant contiguous region.

That is what makes memory reuse practical.

> **Saying it out loud.** The block table is just a per-request list saying where each chunk of my history physically lives — logical block 0 is in physical block 42, logical block 1 is in physical block 7. It's an OS page table with a different name. And the reason it's the crux of the whole design is that it makes growth incremental: the request asks for one more block when it needs one, gets whatever block is free, and never needs anything contiguous. That's what makes reuse practical, and it's also what makes sharing practical, because two requests' tables can simply point at the same physical block.

## 7. What Paging Fixes and What It Does Not

Paging mainly fixes:
- allocator waste
- external fragmentation
- rigid growth behavior

It does not magically make KV cache small.

The underlying historical information is still there.

So a strong answer should say:

"Paged attention improves memory efficiency and serving utilization, but it does not eliminate the fundamental cost of carrying long context."

That distinction matters.

> **Saying it out loud.** Paging fixes allocator waste, external fragmentation and rigid growth — it does not make the KV cache smaller. That distinction is the thing to say, because it's what separates people who understand the technique from people who've heard of it. All the history is still there; you're just storing it efficiently instead of wastefully. So PagedAttention takes you from about 30 percent memory utilisation to the high nineties, which is a 2 to 4x throughput win, and then you're back to the fundamental cost of carrying long context. If you want the cache to actually be *smaller*, that's a different set of tools: GQA, MLA, KV quantization, or eviction.

## 8. Internal Fragmentation Still Exists

Paging does not remove all waste.

If a request uses 3001 tokens and the block size is 128, the last block will not be full.

So there is still some unused capacity in the tail block.

But this waste is bounded.

Instead of wasting an arbitrarily large contiguous reserve, you waste at most roughly one block per request.

That is a much better trade.

This also creates the block-size trade-off:
- larger blocks mean fewer lookups and simpler traversal
- smaller blocks mean tighter packing and less tail waste

That is a real systems trade-off, not a theoretical curiosity.

> **Saying it out loud.** Internal fragmentation doesn't disappear, it just gets bounded — and the bound is what matters. If a request uses 3,001 tokens and blocks hold 128, the last block is nearly empty, so you've wasted up to 127 tokens' worth. But compare that to reserving 2,000 tokens of headroom you'll never use: you've gone from arbitrarily large waste to at most one block per request, which is a few percent instead of two-thirds. That's what creates the block-size tradeoff: bigger blocks mean fewer table lookups and better memory coalescing, smaller blocks mean tighter packing and finer-grained sharing. Sixteen tokens is where vLLM landed, and it's a real systems tradeoff, not a theoretical one.

## 9. Why Paged Attention Is Not a New Attention Formula

This is a very common misunderstanding.

Paged attention does not invent a new probabilistic attention rule.

The model is still attending over the same logical sequence.

What changes is how the kernel gathers the keys and values:
- they may be physically scattered
- the kernel uses the block table to traverse them in logical order

So the math remains equivalent to standard attention over the same history.

The implementation and memory layout are what change.

> **Saying it out loud.** This is the most common misunderstanding about PagedAttention, so it's worth being emphatic: it is not a new attention formula. The probabilities are identical, the math is identical, the output is bit-comparable to standard attention over the same history — nothing is approximated. What changes is purely how the kernel *fetches* the keys and values: instead of streaming a contiguous buffer, it walks the block table and gathers from scattered physical blocks in logical order. It's the same relationship FlashAttention has to attention — a memory-layout and data-movement optimisation wearing an algorithm's name.

## 10. Prefix Sharing Is a Huge Practical Win

Many requests share the same prefix:
- the same system prompt
- the same conversation stem
- the same beam-search history

If each request stored that prefix separately, memory use would explode.

Prefix sharing allows multiple requests to point to the same KV blocks for the common prefix.

That is why block metadata often includes reference counts or similar ownership tracking.

This is especially powerful in:
- beam search
- branching agent trajectories
- repeated enterprise prompts

> **Saying it out loud.** Prefix sharing is where paging pays off a second time. In real traffic, huge numbers of requests begin identically — the same system prompt, the same conversation stem, the same few-shot block — and if each stored its own copy of that KV you'd be spending most of your memory on duplicates. With a block table, sharing is nearly free: point both requests at the same physical blocks and keep a reference count so the block is only released when the last user is done. The workloads where it's transformative are beam search, branching agent trajectories, and enterprise prompts with a long fixed preamble — there you can skip most of the prefill entirely, which is often a 5 to 10x cut in time-to-first-token.

## 11. Why Copy-on-Write Matters

Shared blocks are safe only while requests are reading the same prefix.

The moment one request diverges and needs new continuation state, it must stop writing into the shared block.

So the engine allocates fresh blocks for the diverging continuation.

That is copy-on-write at block granularity.

The reason this matters in interviews is that it shows you understand how sharing and mutation coexist safely.

> **Saying it out loud.** Copy-on-write is what makes sharing safe once requests start to diverge. Two sequences can read the same prefix blocks happily, but the moment one of them needs to write its own continuation into a block that someone else is using, it takes a private copy first and writes there. In practice this only bites at the block where the divergence happens — everything before it stays shared. It's exactly what fork does in an operating system, and the reason it's worth mentioning is that it shows you understand how sharing and mutation coexist: without copy-on-write, one request's next token would silently corrupt another request's history.

## 12. Why Continuous Batching Changes Throughput

Static batching is easy to reason about but wasteful for variable-length decoding.

If one request in a batch finishes early, static batching may leave device capacity underused until the whole batch retires.

Continuous batching fixes this by:
- retiring completed requests immediately
- admitting new requests when capacity opens
- maintaining a rolling active set

That usually improves utilization and throughput.

But it also makes the scheduler more important.

This is why a good answer does not say only "continuous batching is better."

It says:

"Continuous batching improves utilization in heterogeneous workloads, but it introduces scheduling complexity and may not be optimal when ultra-low latency per request is the main objective."

> **Saying it out loud.** Continuous batching means the batch is re-formed at every forward step: finished requests are retired immediately and waiting ones are admitted into the freed slots, so the GPU always has a full set of live work. Static batching leaves those slots idle until the slowest request in the batch completes, which in a workload with 10-token and 2,000-token generations is most of the batch most of the time — so this is typically a 2 to 3x throughput win on its own. The honest caveat is what you're trading: the scheduler now runs thousands of times a second and becomes a real component, and if what you care about is minimum latency for one request rather than throughput across many, a big rolling batch actively hurts you.

## 13. Why GQA and MQA Matter So Much for Serving

The number of KV heads directly affects KV-cache size.

That means an architecture choice made during model design has a first-order impact on inference cost.

This is one of the best examples of a bridge between model architecture and systems behavior.

If you want to sound strong in interview loops, explicitly connect them:

"GQA helps because it reduces KV-cache footprint and memory bandwidth compared with MHA, while usually preserving more quality than full MQA."

That sentence touches model design, serving, and trade-off reasoning at once.

> **Saying it out loud.** This is the cleanest example there is of an architecture decision that's really an infrastructure decision. The number of KV heads multiplies straight through the cache-size formula, so going from 64 KV heads to 8 makes the cache 8 times smaller, which makes your batch 8 times bigger, which since decode is bandwidth-bound makes your throughput roughly 8 times better. That's a serving-cost decision made months earlier by whoever chose the architecture. MQA takes it further with a single shared KV head but measurably loses quality on hard tasks, so GQA-8 is the settled compromise — and connecting model design, cache footprint, bandwidth and serving cost in one breath is exactly the answer interviewers are listening for.

## 14. Why These Tricks Do Not Automatically Transfer to Other Workloads

Paged KV caching is powerful because LLM serving has three specific properties:
- long-lived per-request memory
- variable-length dynamic allocation
- a memory-bound decode loop

If another workload is mostly compute-bound or has static tensor shapes, paging-like indirection may add overhead without enough payoff.

This is an important nuance because mature engineers know not to generalize every optimization beyond its natural workload.

> **Saying it out loud.** It's worth being explicit about why paging works *here*, because the honest version of the answer includes when it wouldn't. LLM serving has three properties that make it a fit: per-request memory that lives a long time, allocation that's dynamic and variable-length, and a decode loop that's memory-bound so utilisation is everything. Take any of those away and the calculus changes — a compute-bound workload with static tensor shapes gets the indirection overhead of a block table and none of the payoff, because it was never wasting memory in the first place. Knowing the boundary of an optimisation is a stronger signal than knowing the optimisation.

## 15. Where Speculative Decoding Fits In

Speculative decoding solves a different problem.

It tries to reduce latency by letting a cheaper draft model propose multiple tokens and having a stronger model verify them.

Paged KV caching instead tries to improve memory efficiency and scheduling flexibility.

These methods are complementary, not substitutes.

So a strong answer can say:

"Speculative decoding attacks the number of expensive verification steps, while paged KV management attacks the memory and allocator bottlenecks of maintaining many active decode states."

> **Saying it out loud.** Speculative decoding and paged KV management attack different bottlenecks, so they compose rather than compete. Speculative decoding reduces the *number* of expensive forward passes: a draft model proposes several tokens, the big model verifies them all in one pass, and you accept the longest agreeing prefix — typically 2 to 3x with 60 to 80 percent acceptance. PagedAttention reduces the *memory waste* per active request, which raises how many requests you can run at once. One is a latency fix, the other a capacity fix. The interaction worth naming is that they compete for memory — a draft model and its KV cache eat space that would otherwise be batch — and that speculative decoding's benefit shrinks as the batch grows, because a big batch is already using the bandwidth well.

## 16. How to Answer "Why Is vLLM Faster?"

A strong answer is not:

"Because it uses PagedAttention."

A stronger answer is:

"Because serving performance depends heavily on KV-cache efficiency and scheduling. vLLM improves both. It manages KV memory in blocks instead of large contiguous buffers, which reduces fragmentation and waste, and it pairs that with dynamic scheduling techniques like continuous batching. That lets the server fit more useful active work into the same GPU memory budget and maintain higher utilization."

That is the kind of answer that sounds complete.

> **Saying it out loud.** The weak answer to why vLLM is faster is one word, PagedAttention. The strong answer is that serving throughput is set by how many requests you can keep active at once, and that's set by two things: how efficiently you use KV memory, and how well you schedule. vLLM improves both. Block-based allocation takes memory utilisation from around 30 percent to the high nineties by killing over-reservation and fragmentation, and continuous batching keeps the active set full so no slot idles behind a finished request. More active requests means more tokens per weight read, which for a bandwidth-bound decode loop is the whole ballgame — that's where the reported 2 to 4x comes from. Notice the model didn't get faster; the memory manager did.

## 17. Questions You Should Be Able to Answer Smoothly

Practice these in full descriptive sentences:

- Why is serving often memory-bound while training is often compute-heavy?

> **Saying it out loud.** Because they have completely different arithmetic intensity. Training pushes thousands of tokens through the weights at once, so each byte you read from memory feeds hundreds of FLOPs and the math units are the limit. Serving generates one token per pass, so you read all the weights and the entire KV cache to produce a single token — around 1 FLOP per byte at batch one, when an H100 in BF16 needs roughly 295 to break even. That's a two-order-of-magnitude gap, and it means the GPU is mostly sitting there waiting on HBM. Which is why serving optimisations are all about moving fewer bytes — quantization, GQA, bigger batches — rather than doing less math.

- Why does plain KV caching solve one problem but create another?

> **Saying it out loud.** It solves recomputation and creates a memory-management problem. Caching past keys and values takes decoding from quadratic to linear, which is essential — nobody serves without it. But now every active request owns a large object that grows by one token every step, whose final size is unknowable, and which has to live in the most expensive memory on the machine for the request's whole lifetime. So you've traded a compute problem for an allocator problem, and at 2.6 gigabytes per request for a 70B model at 8K context, that allocator problem is what caps your batch size — which for bandwidth-bound decode is what caps your throughput.

- Why is fragmentation such a serious issue for serving engines?

> **Saying it out loud.** Because fragmentation directly costs you batch size, and batch size is throughput. Decode is memory-bandwidth-bound, so the only way to get more tokens per second out of a GPU is to serve more requests per weight read — and how many requests fit is determined entirely by KV memory. If 60 to 80 percent of that memory is being wasted on over-reservation and holes, as the vLLM paper measured, then you're running at a third of the batch you could be. So fragmentation isn't a tidiness issue, it's a straight multiplier on your cost per token, and fixing it took utilisation to about 96 percent and throughput up 2 to 4x without touching the model.

- Why does a block-table design help with variable-length requests?

> **Saying it out loud.** Because a block table lets a request's memory grow one small piece at a time instead of being committed up front. You never know how long a generation will be until it stops, so any scheme that needs the size in advance forces you to reserve for the worst case and waste the difference. With a table of fixed-size blocks, the request takes one more block when it needs one, gets whichever block is free, and never needs anything contiguous — so over-reservation goes away and external fragmentation is impossible, since every block is interchangeable. The only waste left is the tail of the last block, at most 15 tokens, which is a bounded few percent instead of an unbounded majority.

- Why does prefix sharing matter for real-world workloads?

> **Saying it out loud.** Because real traffic is full of duplicates. Chat requests carry the same system prompt every time, multi-turn conversations repeat the entire history each turn, agents share a long tool-definition preamble across every call, and beam search shares a stem across all its branches. If every request stores its own copy, memory explodes on redundant data; with block tables, they just point at the same physical blocks with a reference count. The payoff is on both axes: memory you didn't spend and prefill you didn't compute, which is routinely a 5 to 10x reduction in time-to-first-token on chat workloads. It's dead weight only when your traffic genuinely shares nothing.

- Why can a method improve throughput without necessarily improving per-request latency?

> **Saying it out loud.** Because throughput is measured across all users and latency is measured for one, and batching trades directly between them. Increase the batch and each weight read serves more tokens, so system throughput climbs — but every individual request now waits behind more work per step, so its inter-token latency gets worse. Continuous batching is the same story: it maximises utilisation across a heterogeneous workload but adds scheduling and can leave a single request no faster than before. That's why the metric that actually matters commercially is goodput — throughput that meets your latency SLO — because tokens delivered after the user gave up don't count for anything.
