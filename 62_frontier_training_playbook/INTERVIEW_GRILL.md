# Frontier Training Playbook — Interview Grill

> 50 questions on training methodology, scaling laws, data curation, stability, ablations. Drill until you can answer 35+ cold.

---

## A. Scaling laws

**1. State Kaplan's scaling law form.**
$L(N, D) = E + A/N^\alpha + B/D^\beta$. Loss as power-law in params and data.

**2. State the Chinchilla compute-optimal allocation.**
$N \propto C^{0.5}$, $D \propto C^{0.5}$. Roughly 20 tokens per parameter at training optimum.

**3. Compute approximation $C \approx ?$**
$6 N D$ FLOPs for transformer training (forward + backward).

**4. Why is GPT-3 (175B, 300B tokens) considered Chinchilla-suboptimal?**
Severely under-trained. Optimal would be ~3.5T tokens for that param count.

**5. Why do modern Llama models train past Chinchilla?**
Inference cost dominates lifetime cost. Smaller over-trained models are cheaper to serve.

**6. Llama 3 8B trained on 15T tokens — tokens per param?**
~1875. Far past Chinchilla.

**7. Compute-optimal Chinchilla-style allocation for $C = 10^{24}$ FLOPs?**
Use $C = 6 N D$ with $D \approx 20 N$ (the Chinchilla ratio): $C = 6 \cdot N \cdot 20 N = 120 N^2 \Rightarrow N^2 = 10^{24}/120 \approx 8.3 \times 10^{21} \Rightarrow N \approx 91$B, $D \approx 1.8$T. (Note the ratio is $D = 20N$, not $20D = N$.)

**8. What does scaling law imply about predictability?**
Loss is predictable from compute. Lets you forecast capability before training.

---

## B. Architecture choices that ship

**9. What attention variant does Llama 2/3 use?**
Grouped-Query Attention (GQA).

**10. Why GQA over MHA?**
Smaller KV cache → faster inference, lower memory. Almost no quality loss at GQA-8.

**11. Why is RMSNorm preferred over LayerNorm?**
Drops mean centering — slightly faster, equally good empirically.

**12. Pre-LN vs Post-LN at scale?**
Pre-LN: stable, default for modern LLMs. Post-LN: hard to train deep.

**13. Default activation in modern FFN?**
SwiGLU. The gating doubles matmul count (vs vanilla 2-matmul FFN, SwiGLU uses 3); modern recipes (Llama, Mistral) compensate by scaling FFN hidden dim down by $\frac{2}{3}$ (so $\frac{8}{3}d$ instead of $4d$), keeping parameter count roughly constant. Consistently better in evaluation.

**14. Default positional encoding?**
RoPE. Allows context extension via NTK / YaRN.

**15. MLA — what's the innovation?**
Compresses KV via low-rank latent projection. Strongest KV cache reduction. DeepSeek-V2/V3.

**16. Dense vs MoE — main trade-off?**
Dense: simpler, smaller total params for same compute. MoE: bigger total capacity, lower active compute, but routing/load-balancing/communication complexity.

**17. When is MoE worth the complexity?**
When total capacity matters more than wall-clock simplicity. Frontier flagship models often MoE.

---

## C. Data

**18. Why is dedup the most reliable data improvement?**
Prevents memorization of duplicates; reduces effective epochs on common substrings; consistent quality boost.

**19. What's MinHash used for in data curation?**
Near-duplicate detection at scale. Approximates Jaccard similarity efficiently.

**20. What's a quality classifier?**
Binary classifier: "Wikipedia-like vs random web." Filter web data above a threshold.

**21. Why blend new data with general during mid-training?**
Prevents catastrophic forgetting. Typical 90% general / 10% new.

**22. Test-set contamination — why does it matter?**
Public benchmark answers leak into training data over time. Inflates reported numbers without real progress. Always check.

**23. Why filter on perplexity from a small reference LM?**
Drops gibberish + low-quality samples that the reference LM finds unlikely.

**24. Synthetic data — when is it useful?**
Instruction tuning, math reasoning chains, code completion. Risk: hallucinated facts amplify.

---

## D. Hyperparameters and recipes

**25. Standard $\beta_2$ for LLM AdamW?**
0.95 (lower than the default 0.999). More responsive variance estimation.

**26. Standard peak LR for billion-scale LLM?**
$\sim 3 \times 10^{-4}$.

**27. Standard warmup duration?**
Few thousand steps (low single digit thousands).

**28. Cosine decay schedule — to what fraction of peak?**
~10% of peak typically.

**29. Effective batch size in tokens for flagship?**
Millions to tens of millions of tokens (via gradient accumulation across DP).

**30. What's muP and why is it used at frontier labs?**
Maximal Update Parameterization: optimal LR is invariant to model width. Tune small, deploy big without re-sweeping LR.

---

## E. Training stability

**31. What causes loss spikes?**
Bad batches (OOD), numerical instability, optimizer state mismatch.

**32. Standard fix for loss spikes?**
Gradient clipping (1.0), BF16 over FP16, restart with bad batches skipped.

**33. NaN in attention — common cause?**
FP16 overflow in softmax. Fix: BF16, or compute attention in higher precision.

**34. Why is BF16 preferred over FP16 at scale?**
FP32-equivalent exponent range. No need for loss scaling. More stable.

**35. What's z-loss?**
Adds penalty on $\log Z$ to discourage unbounded logit magnitudes. Loss-level intervention.

**36. What's logit softcapping?**
Forward-pass smooth bound: $c \tanh(\mathrm{logits}/c)$. Bounds magnitudes without clipping.

**37. QK normalization?**
Normalize Q and K before attention dot product. Prevents extreme attention scores → softmax instability.

**38. What's catastrophic forgetting in mid-training?**
New focused data overwrites general knowledge from pre-training.

**39. Mitigation for catastrophic forgetting?**
Blend new + general data; replay buffer; weight regularization (e.g., EWC).

**40. Why do hardware failures matter at training scale?**
Single bad GPU can hang or corrupt training. Per-step health checks; auto-checkpoint and resume.

---

## F. Mid-training and post-training

**41. What's mid-training?**
Curated quality boost or domain emphasis after general pre-training. Examples: math/code injection, long-context extension, recency.

**42. Why long-context extension as a separate stage?**
Pre-training at long context is expensive. Train at 8K, extend to 128K with curated long-context data + RoPE rescaling.

**43. NTK / YaRN — what do they do?**
Scale RoPE frequencies to extend usable context length without re-pre-training.

**44. SFT vs preference optimization — what each adds?**
SFT: instruction format, basic capability. RLHF/DPO/GRPO: alignment, helpfulness, refusal calibration, fine-grained capability.

**45. Why does post-training matter so much?**
Determines instruction following, tool use, refusal, reasoning format, preference behavior. Many "model capabilities" are really post-training capabilities.

**46. Reward hacking — what is it?**
Model finds easy way to maximize reward signal that doesn't correspond to good behavior. E.g., longer responses correlate with higher reward → model just makes everything longer.

**47. How do you detect reward hacking?**
Monitor: average response length growing? Sycophantic patterns? Performance on held-out tasks not in reward training? Drift from SFT distribution?

---

## G. Evaluation and ablation

**48. Two-tier ablation strategy?**
Small-scale (1B, 30B tokens) for fast iteration; mid-scale (10B, 200B tokens) for validation; flagship only for proven winners.

**49. Hold compute constant — why?**
Bigger models cost more. To attribute gains correctly, match flops between conditions.

**50. Why multiple seeds for ablations?**
Variance is real at this scale. Single-seed gains often within noise.

**51. Public benchmarks — risks?**
Contamination, prompt sensitivity, cherry-picking, statistical noise. Use private/held-out as ground truth.

**52. Held-out validation perplexity — what to track?**
Per-domain (web, code, math, books). Should monotonically decrease. Spikes = data quality issues.

**53. Chatbot Arena / ELO ratings?**
Crowd-sourced head-to-head comparison. Less prone to gaming than single-model benchmarks.

**54. What's the "depth-vs-width transfer" issue in ablations?**
Architecture changes that help at small scale may hurt at large scale (or vice versa). Common confounder.

---

## H. Operational

**55. Common operational training failures?**
Dataloader stalls, storage bottlenecks, throughput drift, checkpoint corruption, seed inconsistency across TP, silently bad shards.

**56. Throughput drift — what's it usually?**
Network congestion, memory fragmentation, GPU thermal throttling, slow node.

**57. Why are dataloader bugs especially insidious?**
Loss curve looks normal but model trains on wrong/duplicated data. Hard to detect without monitoring data uniqueness.

**58. Frequency of checkpointing?**
Every few thousand steps typically. Async/local for speed; periodic sync to network FS.

**59. Restart on different topology — what's needed?**
Re-sharded checkpoints. DeepSpeed, TorchTitan, Megatron support this.

**60. What does a senior answer to "how would you train Llama-class?" sound like?**
Goal first → constraints → conservative baseline → biggest trade-offs → ablation order → stability/infra risks → strongest claim you'd make.

---

## Self-grading

If you can't answer 1-15, you don't know scaling laws / architecture choices. If you can't answer 16-35, you'll struggle on data / hyperparameter / stability questions. If you can't answer 36-55, frontier-lab methodology interviews will go past you.

Aim for 40+/60 cold.
