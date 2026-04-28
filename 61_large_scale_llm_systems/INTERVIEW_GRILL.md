# Large-Scale LLM Systems — Interview Grill

> 50 questions on memory math, sharding, parallelism, communication, training failures. Drill until you can answer 35+ cold.

---

## A. Memory math

**1. For a model with $P$ params trained with Adam in mixed precision, total state memory?**
$\sim 16P$ bytes: 2 (weights BF16) + 2 (grads BF16) + 4 (momentum FP32) + 4 (variance FP32) + 4 (master FP32).

**2. 70B params — total state memory?**
$16 \cdot 70 \times 10^9 \approx 1.1$ TB.

**3. H100 memory?**
80 GB (HBM3); H200/B200 increase this further.

**4. How many H100s minimum for 70B Adam state?**
$\sim 14$ for state alone (1.1 TB / 80 GB). In practice need many more for activations + comm overhead.

**5. Activation memory scaling?**
$O(\mathrm{batch} \times \mathrm{seq} \times d_{\mathrm{model}} \times L)$ for residual stream, plus larger inside attention/FFN. Long context blows this up.

**6. Activation checkpointing — what does it do?**
Recompute activations during backward instead of storing. ~70% activation memory savings at ~33% extra compute.

**7. BF16 vs FP16?**
BF16 has FP32's exponent range → no overflow → no loss scaling needed. FP16 has more precision but limited range. BF16 default for modern training.

**8. FP8 training — what's new?**
Hopper/Blackwell native. Even less memory + faster matmuls. Requires per-tensor scale management.

---

## B. Data parallelism

**9. DDP — what gets replicated?**
Full weights, grads, optimizer state per rank. Limit: model must fit on single GPU.

**10. ZeRO-1?**
Partition optimizer state across DP ranks. ~4× memory savings.

**11. ZeRO-2?**
Partition optimizer state + gradients. ~8× savings.

**12. ZeRO-3?**
Partition optimizer state + grads + weights. $\sim N$× savings.

**13. FSDP = ?**
PyTorch's implementation of ZeRO-3. Fully Sharded Data Parallel.

**14. ZeRO-3 forward pass?**
All-gather weights for current layer, compute, free. Repeat per layer.

**15. ZeRO-3 backward pass?**
All-gather weights, compute backward, reduce-scatter gradients to owners.

**16. ZeRO-3 vs DDP communication?**
ZeRO-3 has ~1.5× the communication volume of DDP per step (Rajbhandari 2020): all-gather + reduce-scatter is bandwidth-equivalent to one all-reduce, plus an extra all-gather in forward. The trade-off buys $N\times$ memory savings, enabling models that wouldn't otherwise fit.

**17. all-reduce ring algorithm cost?**
$2(N-1)/N \cdot P$ bytes per rank — bandwidth-optimal.

**18. all-gather vs reduce-scatter?**
all-gather: each rank ends up with everyone's data. reduce-scatter: each rank ends up with one shard of the sum.

---

## C. Tensor parallelism

**19. Megatron column parallelism for $Y = XW$?**
Split $W$ along columns: $W = [W_1, W_2]$. Each GPU computes $X W_i$. Concatenate or all-gather outputs.

**20. Row parallelism?**
Split $W$ along rows: $W = [W_1; W_2]$. Each GPU computes $X_i W_i$. All-reduce to sum partial outputs.

**21. Megatron transformer FFN parallelism?**
First linear column-parallel, second linear row-parallel. One all-reduce per FFN block.

**22. Megatron attention parallelism?**
Q, K, V projections column-parallel (split by heads). Output projection row-parallel. One all-reduce per attention block.

**23. Why TP only intra-node?**
All-reduce per layer is latency-bound. Needs NVLink-class bandwidth (~600 GB/s). Cross-node Infiniband (~50 GB/s) too slow.

**24. Typical TP degree?**
8 (matches 8 GPUs per node).

**25. Sequence parallelism — what does it parallelize?**
LayerNorm, dropout, residual ops along sequence dim. Reduces activation memory at cost of extra all-gather/scatter.

---

## D. Pipeline parallelism

**26. PP basic idea?**
Split model by layer: GPU 0 has layers 0-7, GPU 1 has 8-15, etc. Pass activations forward, gradients backward.

**27. Pipeline bubble?**
Idle GPU time during warmup/cooldown of pipeline. Bubble fraction $\approx (P-1)/(P-1+m)$ where $P$ = stages, $m$ = micro-batches.

**28. GPipe — what does microbatching do?**
Splits batch into $m$ micro-batches, pipelines them. Reduces bubble. More $m$ → less bubble but more activation memory.

**29. 1F1B (PipeDream / Megatron)?**
Interleaved forward and backward of different micro-batches. Steady-state every GPU busy. Lower memory than GPipe.

**30. PP communication cost?**
Send activations forward, gradients backward — small per step, infrequent.

**31. PP layer balancing — why important?**
All stages must take similar compute, else slowest stage bottlenecks. Embedding/output layers often need special handling.

---

## E. 3D parallelism

**32. TP × PP × DP for 175B on 1024 GPUs — typical config?**
TP = 8 (intra-node), PP = 16 (cross-node), DP = 8 (replicate). Each rank holds $175\mathrm{B}/(8 \cdot 16) = 1.4\mathrm{B}$ params.

**33. Why this layout?**
TP wants high bandwidth (intra-node). PP wants moderate (small cross-node sends). DP wants infrequent (across many nodes).

**34. What does context parallelism do?**
Splits attention computation across sequence positions. Critical for million-token context.

**35. Ring attention — basic mechanism?**
Each GPU holds a slice of the sequence. K/V are passed in a ring; each rank computes its query against all K/V over time.

---

## F. Expert parallelism (MoE)

**36. EP — what is it?**
Different experts on different GPUs. Tokens routed via all-to-all to their assigned expert.

**37. MoE communication pattern per layer?**
Two all-to-alls: dispatch tokens to experts, gather outputs back.

**38. Why is all-to-all the bottleneck?**
Most expensive collective; scales poorly across nodes; no overlap with compute.

**39. Capacity factor — purpose?**
Caps tokens per expert per batch. Drops overflow. Prevents one expert from being overloaded.

**40. Auxiliary loss in MoE?**
Penalty added to encourage uniform routing across experts. DeepSeek-V3 uses auxiliary-loss-free balancing instead.

---

## G. Compute / communication overlap

**41. DDP overlap?**
Gradient all-reduce of layer $\ell$ overlaps with backward of layer $\ell - 1$.

**42. FSDP overlap?**
All-gather of weights for layer $\ell + 1$ overlaps with compute of layer $\ell$.

**43. MFU — what is it?**
Model FLOPs Utilization. Achieved FLOPs / theoretical peak. Frontier targets > 50%.

**44. HFU vs MFU?**
HFU (Hardware FLOPs Utilization) counts all FLOPs done. MFU counts only useful (non-recomputed) FLOPs. MFU < HFU when activation checkpointing is on.

---

## H. Failures at scale

**45. Loss spike — common cause + fix?**
Numerical instability (BF16 limits, optimizer state). Fix: gradient clipping (1.0), warmup, lower LR, BF16 over FP16, restart from checkpoint.

**46. NaN in attention — common cause?**
FP16 overflow in softmax for large logits. Fix: BF16 (FP32 exponent range), or compute attention in higher precision.

**47. NCCL hang — what's happening?**
One rank stuck (e.g., bad GPU, ECC error) while others wait on collective. Fix: timeouts, watchdog, health checks before training.

**48. Straggler — what's the impact?**
One slow GPU bottlenecks the whole job (synchronous SGD). Fix: pre-flight checks; redundant replicas; periodic re-detection.

**49. Checkpoint write times?**
Naive synchronous checkpointing of 100 GB is slow. Fix: async checkpointing, local node checkpoint + later upload, sharded format.

**50. Restart on different topology?**
Need re-sharded checkpoints. Frameworks like DeepSpeed and TorchTitan support this.

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

---

## Self-grading

If you can't answer 1-15, you can't talk about training memory. If you can't answer 16-35, you'll fail any infra question on a frontier-lab interview. If you can't answer 36-50, large-scale LLM systems interviews will go past you.

Aim for 40+/60 cold.
