# Topic 63: Paged Attention and LLM Serving Internals

> 🔥 **For interviews, read these first:**
> - **`paged_attention_deep_dive.md`** — descriptive deep dive: serving vs training bottlenecks, KV-cache fragmentation, paging analogy, block tables, prefix sharing, continuous batching, prefill/decode disaggregation.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions covering KV-cache math (GQA/MQA/MLA savings), PagedAttention internals, continuous batching, prefix caching / RadixAttention, speculative decoding, INT8/INT4/FP8 quantization, vLLM/SGLang/TensorRT-LLM differentiation.

## What You'll Learn

This topic goes deeper on the serving internals that matter in strong LLM systems interviews.

You will learn:
- why serving is often memory-bound rather than compute-bound
- how KV-cache growth drives inference cost
- why naive KV allocation causes waste and fragmentation
- how paged attention organizes KV memory into blocks
- how block tables, prefix sharing, and copy-on-write work
- why continuous batching changes serving behavior
- how speculative decoding and paged KV caching fit together

## Why This Matters

Interviewers increasingly ask systems questions like:
- "Why is LLM serving expensive?"
- "What is PagedAttention actually fixing?"
- "Why is KV cache such a big bottleneck?"
- "How does vLLM get better throughput?"

Good answers require more than saying "use KV cache" or "batch requests."

They require you to explain the memory model.

## Core Intuition

During autoregressive generation, the model keeps producing new keys and values for every token at every layer.

Those keys and values must stay available because future tokens attend to them.

So the serving problem becomes:

"How do we keep large growing per-request memory around without wasting too much space or stalling the GPU?"

Naive serving often reserves memory as if each request will use one large contiguous buffer.

That causes two problems:
- over-reservation: you hold space a request may never use
- fragmentation: free space exists, but not in the right contiguous shape

Paged attention solves this by making KV cache look more like virtual memory:
- break KV storage into fixed-size blocks
- let each request own a logical sequence of blocks
- translate logical positions to physical blocks with a block table

The attention math stays the same.

What changes is how memory is laid out and addressed.

## Technical Details Interviewers Often Want

### Why KV Cache Dominates Serving

At decode time, compute per new token may be moderate, but memory traffic is large because the model has to read all prior keys and values for the current sequence.

KV-cache cost grows with:
- number of layers
- number of KV heads
- head dimension
- sequence length
- batch size

That is why inference questions often become memory-bandwidth questions.

### Fragmentation in Naive Allocation

Suppose each request gets a large contiguous cache reservation for its worst-case length.

Then:
- short requests waste space
- completed requests leave holes
- new requests may not fit even when total free memory looks large

This is classic fragmentation.

Paged KV caching reduces that waste because requests can grow block by block instead of reserving one giant contiguous region.

### Block Tables

Each request keeps a mapping from logical token positions to physical KV blocks.

That means:
- the sequence is logically continuous
- the underlying memory can be physically non-contiguous

The key interview point is:

"PagedAttention changes the storage layout, not the semantics of attention."

### Prefix Sharing and Copy-on-Write

If multiple requests share the same prefix, they can often share the same KV blocks for that prefix.

That is powerful for:
- branching generations
- beam search
- repeated system prompts

Copy-on-write means the shared blocks stay shared until one branch needs to modify or extend in a way that requires new ownership.

### Continuous Batching

Traditional static batching waits for a batch to finish before admitting new work.

Continuous batching admits and retires requests dynamically.

That improves utilization because:
- finished sequences stop consuming decode slots
- shorter requests do not block the entire batch
- the server keeps the GPU busy with a changing set of active requests

The trade-off is that scheduling logic becomes more complex.

### How This Relates to Speculative Decoding

Speculative decoding attacks latency by proposing multiple future tokens cheaply and verifying them.

Paged KV caching attacks memory waste and scheduling inefficiency.

They solve different bottlenecks and can complement each other.

## Common Failure Modes

### 1. Explaining KV Cache but Not Memory Layout

That is enough for an entry-level answer, but stronger systems interviews usually want the fragmentation and allocator story too.

### 2. Treating Paging as a Change to the Attention Equation

Paged attention does not change the mathematical objective of attention.

It changes how KV memory is stored and gathered efficiently.

### 3. Ignoring Memory Bandwidth

Candidates sometimes talk only about FLOPs.

Serving often bottlenecks on moving KV data around, not just on raw arithmetic.

### 4. Claiming Throughput Gains Without Mentioning Scheduling

High throughput in a real server depends on:
- batching policy
- request length distribution
- memory allocator behavior
- prefix reuse

### 5. Forgetting GQA and MQA Effects

Serving cost is strongly affected by the number of KV heads.

That is why MQA and GQA matter not only for architecture, but also for serving efficiency.

## Edge Cases and Follow-Up Questions

### What if paged allocation reduces fragmentation but block lookup adds overhead?

Then the real question is whether the memory-efficiency gain outweighs the lookup and gather overhead on the target hardware and workload.

### What if prefixes are rarely shared?

Then prefix sharing gives little benefit, but paged KV management can still help by reducing fragmentation and improving allocator flexibility.

### What if request lengths are very uniform?

Then the scheduling and fragmentation benefit of paging may be smaller than in highly variable workloads.

### What if latency matters more than throughput?

Then aggressive batching may not be the right answer, even if it improves device utilization.

### What if GQA reduces KV memory but hurts quality?

Then you must discuss the trade-off honestly: better serving efficiency may come at some representational cost.

## Boilerplate Code

See:
- [paged_attention.py](/Users/faisal/Projects/ml_and_llm_learning/63_paged_attention_and_llm_serving/paged_attention.py)
- [serving_notes.md](/Users/faisal/Projects/ml_and_llm_learning/63_paged_attention_and_llm_serving/serving_notes.md)

The Python file contains small interview-friendly helpers for:
- KV-cache memory estimation
- naive contiguous allocation waste estimates
- paged block usage
- a toy continuous-batching scheduler

The goal is not to reimplement vLLM.

The goal is to make the core ideas mechanically clear.

For a longer descriptive explanation of the serving logic, allocator trade-offs, and why paging helps, read [paged_attention_deep_dive.md](/Users/faisal/Projects/ml_and_llm_learning/63_paged_attention_and_llm_serving/paged_attention_deep_dive.md).

## What to Practice Saying Out Loud

1. Why is LLM serving often memory-bound even when training is compute-heavy?
2. What problem does paged attention solve that plain KV caching does not?
3. Why does fragmentation matter in a serving engine?
4. How does continuous batching improve throughput?
5. Why do GQA and MQA matter for KV-cache cost?
6. What does prefix sharing buy you, and when might it not help much?

## Suggested Use

Use this topic after:
- [06_llm_inference](/Users/faisal/Projects/ml_and_llm_learning/06_llm_inference/README.md)
- [29_system_design_ml](/Users/faisal/Projects/ml_and_llm_learning/29_system_design_ml/README.md)
- [61_large_scale_llm_systems](/Users/faisal/Projects/ml_and_llm_learning/61_large_scale_llm_systems/README.md)
