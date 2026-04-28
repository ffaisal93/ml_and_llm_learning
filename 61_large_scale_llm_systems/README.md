# Topic 61: Large-Scale LLM Systems

> 🔥 **For interviews, read these first:**
> - **`LARGE_SCALE_LLM_DEEP_DIVE.md`** — frontier-lab deep dive: training memory math ($16P$ rule), activation checkpointing, BF16/FP8, ZeRO-1/2/3 / FSDP, Megatron tensor parallelism (column/row), pipeline parallelism + bubble formula, 3D parallelism layout, expert parallelism for MoE, sequence/context parallelism, MFU, training failure modes (loss spikes, NaNs, hangs, stragglers).
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions.

## What You'll Learn

This topic covers the systems fluency that often appears in research-scientist interviews when they shift from model ideas to scale constraints.

You will learn:
- memory breakdown in training
- why optimizer states are expensive
- gradient accumulation
- mixed precision
- activation checkpointing
- FSDP and ZeRO intuition
- data, tensor, and pipeline parallelism intuition
- throughput vs latency
- serving trade-offs
- failure modes at scale

## Why This Matters

Even research interviews often push on scale:
- "Why did training OOM?"
- "How would you fit a longer context?"
- "Why is serving so expensive?"
- "What do you shard?"

The repo already covers some single-GPU and inference basics. This topic makes the multi-GPU and large-scale reasoning explicit.

## Core Intuition

Large-scale LLM systems questions are usually bottleneck questions.

The interviewer is often asking some version of:

"What resource is running out first, and what lever would you pull?"

That resource might be:
- GPU memory
- training throughput
- inference latency
- communication bandwidth
- engineering reliability

The best answers break the system into components, identify the dominant bottleneck, and then explain the trade-off of each mitigation.

## Files in This Topic

- [large_scale_systems.md](/Users/faisal/Projects/ml_and_llm_learning/61_large_scale_llm_systems/large_scale_systems.md): detailed interview explanations
- [systems_tradeoffs.py](/Users/faisal/Projects/ml_and_llm_learning/61_large_scale_llm_systems/systems_tradeoffs.py): small memory and throughput calculators

## Technical Details Interviewers Often Want

### Memory Breakdown

Training memory is not just parameters.

A useful decomposition is:
- model parameters
- gradients
- optimizer states
- activations

For Adam-style optimizers, optimizer states can be a major memory cost because each parameter may need multiple additional tensors.

This is why changing the optimizer or sharding optimizer states can have a large effect.

### Why Long Context Is Expensive

Longer context affects both memory and compute.

In attention-based models, longer sequences increase the size of attention score matrices and the amount of activation storage.

That means context-length questions are rarely just "more tokens is better."

They are trade-off questions involving:
- memory
- latency
- throughput
- model quality

### Mixed Precision and Checkpointing

Mixed precision reduces memory and can improve throughput, but it also increases sensitivity to numerical instability.

Activation checkpointing reduces activation memory by recomputing parts of the forward pass during backpropagation.

That means it trades memory for extra compute.

### FSDP and ZeRO Intuition

You do not need every implementation detail in an interview, but you should know the high-level purpose:
- shard parameters, gradients, and or optimizer states across devices
- reduce per-device memory footprint
- accept communication overhead as a trade-off

### Throughput vs Latency

This distinction is frequently misunderstood.

Throughput asks how much total work the system can do over time.

Latency asks how long one request takes.

Batching often helps throughput but can hurt latency.

That is a classic interview trade-off.

## Common Failure Modes

### 1. Treating OOM as a Single Problem

Out-of-memory errors can come from different sources:
- activations too large
- optimizer states too large
- sequence length too large
- microbatch too large

The fix depends on which component dominates.

### 2. Naming Parallelism Without Explaining the Trade-Off

Saying "use FSDP" or "use tensor parallelism" is incomplete unless you also say what cost you are paying, usually communication or implementation complexity.

### 3. Ignoring Inference Constraints

A model can look good in training discussion and still be impractical at serving time because of KV-cache growth, latency, or hardware cost.

### 4. Confusing Throughput Improvement with Better User Experience

Higher throughput does not automatically mean lower latency for an individual request.

## Edge Cases and Follow-Up Questions

### What if training fits but inference is still too expensive?

Then the bottleneck has shifted.

You may need to discuss:
- KV-cache memory
- batching policy
- quantization
- speculative decoding
- serving architecture

### What if gradient checkpointing slows training too much?

Then it may be the wrong lever if compute, not memory, is already the limiting resource.

### What if communication dominates after sharding?

Then additional sharding may no longer help.

This is why scaling techniques must be evaluated in the context of interconnect speed and cluster topology.

### What if the user asks for lower latency and higher throughput at the same time?

You should explain that those goals can conflict and that the right solution depends on workload shape and batching strategy.

## Core Interview Pattern

When asked a scale question:

1. State the bottleneck.
2. Break memory or latency into components.
3. Name the levers.
4. Explain the trade-off of each lever.
5. Pick the best first action for the stated constraint.

That structure makes your answer sound practical instead of vague.

## What to Practice Saying Out Loud

1. What component is dominating memory here: parameters, activations, gradients, or optimizer state?
2. Why does longer context increase both quality potential and systems cost?
3. What does checkpointing save, and what does it cost?
4. Why can sharding solve memory while worsening communication overhead?
5. What is the difference between improving throughput and improving latency?

## Suggested Use

For deeper follow-up on frontier methodology and serving-engine internals, continue to:
- [62_frontier_training_playbook](/Users/faisal/Projects/ml_and_llm_learning/62_frontier_training_playbook/README.md)
- [63_paged_attention_and_llm_serving](/Users/faisal/Projects/ml_and_llm_learning/63_paged_attention_and_llm_serving/README.md)
