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

## 3. Why KV Cache Becomes the Serving Bottleneck

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

## 6. What a Block Table Is Doing

A block table is conceptually simple.

For a given request, it answers:

"If I need the KV data for logical block 0, 1, 2, ... where do those blocks actually live in memory?"

This means the request can grow incrementally:
- first one block
- then another
- then another

without requiring one giant contiguous region.

That is what makes memory reuse practical.

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

## 9. Why Paged Attention Is Not a New Attention Formula

This is a very common misunderstanding.

Paged attention does not invent a new probabilistic attention rule.

The model is still attending over the same logical sequence.

What changes is how the kernel gathers the keys and values:
- they may be physically scattered
- the kernel uses the block table to traverse them in logical order

So the math remains equivalent to standard attention over the same history.

The implementation and memory layout are what change.

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

## 11. Why Copy-on-Write Matters

Shared blocks are safe only while requests are reading the same prefix.

The moment one request diverges and needs new continuation state, it must stop writing into the shared block.

So the engine allocates fresh blocks for the diverging continuation.

That is copy-on-write at block granularity.

The reason this matters in interviews is that it shows you understand how sharing and mutation coexist safely.

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

## 13. Why GQA and MQA Matter So Much for Serving

The number of KV heads directly affects KV-cache size.

That means an architecture choice made during model design has a first-order impact on inference cost.

This is one of the best examples of a bridge between model architecture and systems behavior.

If you want to sound strong in interview loops, explicitly connect them:

"GQA helps because it reduces KV-cache footprint and memory bandwidth compared with MHA, while usually preserving more quality than full MQA."

That sentence touches model design, serving, and trade-off reasoning at once.

## 14. Why These Tricks Do Not Automatically Transfer to Other Workloads

Paged KV caching is powerful because LLM serving has three specific properties:
- long-lived per-request memory
- variable-length dynamic allocation
- a memory-bound decode loop

If another workload is mostly compute-bound or has static tensor shapes, paging-like indirection may add overhead without enough payoff.

This is an important nuance because mature engineers know not to generalize every optimization beyond its natural workload.

## 15. Where Speculative Decoding Fits In

Speculative decoding solves a different problem.

It tries to reduce latency by letting a cheaper draft model propose multiple tokens and having a stronger model verify them.

Paged KV caching instead tries to improve memory efficiency and scheduling flexibility.

These methods are complementary, not substitutes.

So a strong answer can say:

"Speculative decoding attacks the number of expensive verification steps, while paged KV management attacks the memory and allocator bottlenecks of maintaining many active decode states."

## 16. How to Answer "Why Is vLLM Faster?"

A strong answer is not:

"Because it uses PagedAttention."

A stronger answer is:

"Because serving performance depends heavily on KV-cache efficiency and scheduling. vLLM improves both. It manages KV memory in blocks instead of large contiguous buffers, which reduces fragmentation and waste, and it pairs that with dynamic scheduling techniques like continuous batching. That lets the server fit more useful active work into the same GPU memory budget and maintain higher utilization."

That is the kind of answer that sounds complete.

## 17. Questions You Should Be Able to Answer Smoothly

Practice these in full descriptive sentences:

- Why is serving often memory-bound while training is often compute-heavy?
- Why does plain KV caching solve one problem but create another?
- Why is fragmentation such a serious issue for serving engines?
- Why does a block-table design help with variable-length requests?
- Why does prefix sharing matter for real-world workloads?
- Why can a method improve throughput without necessarily improving per-request latency?
