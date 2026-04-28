# Large-Scale LLM Systems

## 1. Training Memory Breakdown

A useful way to explain GPU memory is:

- model weights
- gradients
- optimizer states
- activations
- temporary buffers / fragmentation

For Adam-style training, optimizer states are expensive because you usually keep:
- parameters
- gradients
- first moment
- second moment

That is why memory can be much larger than just parameter count.

## 2. What to Do When You OOM

Common levers:

1. reduce batch size
2. use gradient accumulation
3. use mixed precision
4. use activation checkpointing
5. shorten sequence length
6. shard optimizer states and parameters

## 3. Gradient Accumulation

Useful explanation:

"Gradient accumulation lets me simulate a larger effective batch size without storing all microbatches in memory at once. I still pay more wall-clock time, but I reduce instantaneous memory pressure."

## 4. Activation Checkpointing

Useful explanation:

"Checkpointing saves memory by not storing every intermediate activation. Instead, some activations are recomputed during backward pass. So I trade extra compute for lower memory."

## 5. Mixed Precision

Useful explanation:

"Mixed precision reduces memory and often improves throughput, but it can introduce numerical instability if scaling and sensitive operations are not handled carefully."

## 6. FSDP / ZeRO Intuition

### FSDP

Shard model parameters, gradients, and optimizer state across devices so no single GPU holds the full copy all the time.

### ZeRO

Partition optimizer state, gradients, and sometimes parameters across ranks to reduce redundant memory replication.

Easy interview phrasing:

"The main idea is to avoid every GPU holding a full copy of everything."

## 7. Parallelism Types

### Data Parallelism

Each GPU gets different data, same model.

Good when:
- model fits on each device
- you want higher throughput

### Tensor Parallelism

Split tensors or layers across devices.

Good when:
- a single layer is too large for one device

### Pipeline Parallelism

Split layers into stages across devices.

Good when:
- model depth is large
- you can tolerate pipeline scheduling complexity

## 8. Long Context Costs

Longer context usually increases:
- activation memory
- attention memory
- latency

Useful answer:

"If context length doubles, attention cost often grows more than linearly, and in vanilla full attention it grows quadratically with sequence length."

## 9. Serving Trade-Offs

At serving time, common trade-offs are:

- latency vs throughput
- batch size vs tail latency
- model size vs cost
- quantization vs accuracy
- cache size vs memory

## 10. Failure Modes at Scale

Things that often break:

- OOM from activations
- NCCL or communication bottlenecks
- rank desynchronization
- mixed precision instability
- checkpoint corruption
- data pipeline starvation

## 11. What Interviewers Often Want

Usually they do not need deep framework-specific commands.

They want to hear:
- that you know the bottleneck
- that you know the available levers
- that you understand the trade-off of each lever
