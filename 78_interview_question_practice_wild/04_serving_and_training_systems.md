# Serving and training systems

**Foundation explainer:** Core material · **Priority:** Very important

Very important for MLE and LLM roles: explain the resource bottleneck and the resulting design.

[Chapter guide](README.md) · [First 100](FIRST_100.md) · [Next 100](NEXT_100.md)

<a id="f031"></a>

## F031. Why do prefill and decode have different bottlenecks?

**Start here.** Prefill processes many prompt tokens together; decode repeatedly produces the next token for each active request.

**Work it through.**

1. Prefill uses larger matrix operations and can often exploit substantial parallel compute.
2. Decode at small batch sizes frequently rereads large weights for relatively little work and also reads a growing KV cache.
3. Measure arithmetic intensity, batch size, sequence length and hardware. The bottleneck can change; “prefill compute-bound, decode bandwidth-bound” is a useful tendency, not a universal law.

**Small example.** One request with a long prompt can have high time-to-first-token, while a large active batch can increase inter-token latency.

**Common trap.** Reporting only tokens per second while hiding slow first responses or poor tail latency.

**Follow-up.** Which metrics distinguish the phases? Time-to-first-token, inter-token latency, prompt throughput and generation throughput.

---

<a id="f032"></a>

## F032. Implement the scheduling core of a dynamic batcher.

**Start here.** A batcher trades a small queueing delay for more efficient model calls.

**Work it through.**

1. Keep a FIFO queue of request id, payload and arrival time. Assign ids monotonically.
2. Dispatch when size reaches b_max or the oldest request reaches its inclusive deadline. Remove up to b_max items and call the model once.
3. Map outputs back by request id. Preserve a backlog for subsequent dispatches; define model-failure handling separately. The coding lab implements the deterministic core.

**Small example.** With b_max=3 and five overdue requests, two dispatches should call the model on batches of sizes 3 and 2.

**Common trap.** Calling the model once per item, returning outputs to the wrong clients, or expiring all overdue requests as one-item batches.

**Follow-up.** Is t_max an unconditional latency guarantee? No; overloaded queues can wait longer than their dispatch deadlines.

---

<a id="f033"></a>

## F033. Explain continuous batching and paged KV storage.

**Start here.** Continuous batching changes the active set between generation steps; paged storage makes growing caches easier to allocate.

**Work it through.**

1. Admit new requests as capacity becomes available and remove finished requests instead of waiting for a fixed batch to complete.
2. Store each request's KV sequence in fixed-size blocks referenced by a logical block table, allowing physical blocks to be non-contiguous.
3. Track live-token memory, fragmentation, eviction and scheduler fairness. Paged allocation changes storage management; it does not change what attention means.

**Small example.** If one request finishes after 5 tokens and another after 100, replacing the finished slot can improve utilization over waiting for both.

**Common trap.** Claiming block allocation eliminates all fragmentation or makes the logical cache constant-size.

**Follow-up.** How would you handle a long prefill that stalls decode? Consider chunked prefill and explicit latency-aware scheduling.

---

<a id="f034"></a>

## F034. Estimate an LLM serving capacity budget.

**Start here.** Capacity is constrained by memory, computation, memory traffic and arrival patterns, so one parameter count is insufficient.

**Work it through.**

1. Estimate weight bytes, KV bytes for the live-token budget, temporary buffers and runtime overhead.
2. Benchmark prefill and decode at representative lengths and concurrency; measure throughput and tail latency together.
3. Set admission limits below the measured saturation point and handle overload with bounded queues, rejection or a stated fallback.

**Small example.** Ten requests at 4000 cached tokens consume the same idealized KV tensor space as twenty at 2000, assuming identical models and no block rounding.

**Common trap.** Sizing only for average context length or filling nominal GPU memory until allocations fail.

**Follow-up.** What happens if traffic shifts toward longer outputs? Cache residency and decode work rise even at unchanged request arrival rate.

---

<a id="f035"></a>

## F035. Compare data, tensor and pipeline parallel training.

**Start here.** They partition different parts of the training work and create different communication patterns.

**Work it through.**

1. Data parallelism replicates the model across workers processing different batches, then synchronizes gradients.
2. Tensor parallelism splits individual matrix operations and usually requires frequent high-bandwidth communication within layers.
3. Pipeline parallelism assigns groups of layers to stages and streams microbatches through them; balance stages and account for pipeline bubbles. Combine approaches only after estimating memory and network costs.

**Small example.** Two data-parallel workers with local batch 8 process a global batch of 16 if one synchronized update uses both batches without extra accumulation.

**Common trap.** Assuming all parallelism reduces per-worker model memory, or ignoring communication when predicting speedup.

**Follow-up.** Where does gradient accumulation fit? It increases effective batch per update without adding devices, at extra sequential compute time.

---

<a id="f036"></a>

## F036. Derive the backward communication of a row-sharded linear layer.

**Start here.** The gradient communication follows which forward tensors were replicated or assembled, not the name of a collective alone.

**Work it through.**

1. Fix a convention: X:(N,d_in), W:(d_in,d_out), Y=XW. If output columns of W are split, each worker produces a slice of Y.
2. For replicated X, each worker computes its local dW=Xᵀ*dY_slice and partial dX=dY_slice*W_sliceᵀ.
3. Sum partial dX across workers. If forward concatenated slices with all-gather, backward distributes the corresponding gradient slices with the correct sum semantics for the surrounding graph.

**Small example.** For W=[W0 W1], dX=dY0*W0ᵀ+dY1*W1ᵀ. Neither worker's contribution alone is the complete input gradient.

**Common trap.** Using “row-sharded” without stating weight orientation; different libraries use the term differently.

**Follow-up.** When is reduce-scatter useful? When the summed input gradient should also remain sharded for the next operation.

---

<a id="f037"></a>

## F037. Apply the roofline model to a matrix multiplication.

**Start here.** The roofline compares available compute throughput with the rate at which needed data can reach the processor.

**Work it through.**

1. For A:(M,K) and B:(K,N), estimate about 2MKN floating-point operations.
2. Estimate bytes moved under an explicit reuse assumption, then compute arithmetic intensity=FLOPs/bytes.
3. The idealized throughput ceiling is min(peak_FLOPs,bandwidth*intensity). Use profiling to include tiling, launch overhead and cache behavior.

**Small example.** If bandwidth is 1 TB/s and intensity is 10 FLOPs/byte, the bandwidth ceiling is 10 TFLOPs/s, even if peak compute is 100 TFLOPs/s.

**Common trap.** Counting every scalar reference as an uncached memory transfer, or treating a theoretical bound as achieved performance.

**Follow-up.** Why can batching help decode? It reuses weight reads across more tokens, increasing arithmetic intensity.

---

<a id="f038"></a>

## F038. Choose a quantization approach and validate its cost.

**Start here.** Quantization stores or computes numbers with fewer bits, trading representational precision for memory and possibly speed.

**Work it through.**

1. Specify which tensors are quantized: weights, activations or KV cache, and at what granularity and scale representation.
2. Estimate actual bytes including scales and metadata. Benchmark kernels on the target hardware; smaller storage does not automatically mean faster execution.
3. Evaluate representative tasks, long contexts and sensitive slices against a baseline. Keep a rollback path if quality regresses.

**Small example.** A nominal 4-bit weight tensor uses one quarter the raw weight bytes of a 16-bit tensor, before scales, packing and unquantized tensors.

**Common trap.** Claiming a fourfold end-to-end memory or latency improvement from raw weight bits alone.

**Follow-up.** Why do outliers matter? A few large values can consume a shared quantization range and reduce resolution for common values.

---

<a id="f039"></a>

## F039. Design a fair GPU job scheduler.

**Start here.** The scheduler must satisfy resource constraints while making progress and controlling queueing, rather than merely picking the shortest job.

**Work it through.**

1. Represent GPU count, memory, topology, priority, estimated duration and checkpoint support in each job request.
2. Use admission checks, quotas and a queue policy; reserve whole compatible allocations for distributed jobs to avoid partial-allocation deadlocks.
3. Support leases, heartbeats, retries and checkpoint-aware preemption. Monitor utilization, wait percentiles and starvation across tenants.

**Small example.** An eight-GPU job requiring one fast interconnect group cannot be satisfied by eight isolated single-GPU slots with inadequate network bandwidth.

**Common trap.** Optimizing utilization by indefinitely postponing large or low-priority jobs.

**Follow-up.** How do you prevent starvation? Aging priorities, guaranteed shares or reserved capacity, with explicit tradeoffs.

---

<a id="f040"></a>

## F040. Bound latency in a batch service under load.

**Start here.** A dispatch deadline is a scheduling rule; queue stability is a separate requirement.

**Work it through.**

1. If a worker handles at most b requests per model call taking s seconds, its optimistic sustained capacity is b/s requests per second.
2. A bound like wait<=t_max+s needs additional conditions: a ready batch can be served after at most one current model call, with no earlier queued backlog or competing delay.
3. Under sustained arrivals above service capacity, backlog grows; bound the queue and apply admission control or add capacity. Below average capacity, bursts can still violate a per-request bound.

**Small example.** With b=8 and s=0.1, the optimistic capacity is 80 requests/s. Arrivals at 100/s add about 20 queued requests/s if nothing is rejected.

**Common trap.** Claiming arrival rate below b/s alone guarantees every request's deadline.

**Follow-up.** How do you connect clients without a busy loop? Futures or condition variables, a dispatcher timer and request-id-based completion.

---
