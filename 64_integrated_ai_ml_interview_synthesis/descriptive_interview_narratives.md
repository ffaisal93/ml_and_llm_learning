# Descriptive Interview Narratives

This file is for a specific skill:

turning knowledge into spoken, descriptive, interviewer-friendly answers.

Many candidates know the facts but still sound weak because their answers feel like disconnected notes. These examples are written in the style you should aim to speak.

## 1. "Why Does GQA Help?"

Weak answer:

"GQA shares keys and values across groups, so it is more efficient."

Stronger answer:

"Grouped-query attention is mainly an efficiency-quality compromise. Full multi-head attention keeps separate key and value heads for every query head, which makes the KV cache large and expensive to move during serving. Multi-query attention goes to the other extreme and shares one KV set across all heads, which is cheap but can lose quality because every head reads from the same compressed memory. GQA sits in the middle: it reduces KV-cache size materially while preserving more specialization than MQA. So the reason it helps is not just architectural elegance. It changes serving cost and memory bandwidth in a favorable way while usually keeping much of the quality of full MHA."

Why this answer is stronger:
- it explains the trade-off
- it connects architecture to serving
- it shows why the middle-ground design exists

> **Saying it out loud.** GQA is there to shrink the KV cache, and the KV cache is what actually fills your GPU at serving time. Full multi-head attention stores a separate key and value for every single head, so the cache is enormous and you're re-reading all of it from memory on every decoded token. Multi-query goes to one shared set — much smaller, but quality drops and it's finicky to train. GQA splits the difference by letting a group of query heads share one KV head. Llama-2 70B runs 64 query heads over 8 KV heads, so an 8x smaller cache with essentially no quality loss — and the FLOPs barely change, because you're still running every query head. The win is bytes moved, not math done.

## 2. "How Would You Train a Frontier-Style Model?"

Weak answer:

"I would use a transformer with RoPE, GQA, good data, and then do SFT and RL."

Stronger answer:

"I would start by locking the product goal and evaluation protocol early, because otherwise architecture decisions become ungrounded. Then I would choose a conservative baseline recipe with known failure modes, usually something dense with GQA and a standard optimizer unless there is a strong reason to absorb the extra complexity of MoE or a more exotic optimizer. From there, I would de-risk the program by running small reliable ablations on only a few variables at a time: attention setup, positional encoding strategy, optimizer choice, and the learning-rate schedule. In parallel, I would make the data pipeline a first-class part of the project, with deduplication, contamination checks, and a planned multi-stage mixture rather than one static soup of data. Finally, I would treat post-training as part of the training story instead of an afterthought, because instruction following, reasoning style, and tool use are often determined there."

Why this answer is stronger:
- it sounds like a plan, not a list
- it treats evals and data as central
- it shows methodology and scientific discipline

> **Saying it out loud.** I'd lock the evaluation before I touch the architecture, because otherwise every later decision is unfalsifiable — you can't tell whether a change helped if the yardstick keeps moving. Then a deliberately boring baseline: dense, GQA, a standard optimizer, nothing exotic unless there's a specific reason to absorb the risk. Ablate a few variables at a time on small runs, and treat the data pipeline as a first-class project with deduplication and contamination checks, not as a preprocessing step somebody does in a week. And I'd plan post-training from the start, because instruction-following and reasoning style are mostly decided there. The named failure is the confounded ablation — change four things at once and you've spent the compute and learned nothing causal.

## 3. "Why Can a Better Metric Still Be Misleading?"

Weak answer:

"Maybe the benchmark is bad."

Stronger answer:

"A metric can improve for the wrong reason. For example, a model can get better perplexity because it predicts local token patterns more accurately, but that does not guarantee better factuality, instruction following, or user preference. Likewise, a retrieval system can improve recall@k while the final answer quality stays flat if the generator still ignores the evidence. So when I see a gain, I first ask what behavior the metric really tracks, then what behaviors it ignores, and then whether the reported gain is robust across slices and seeds. The issue is usually not that the metric is useless. It is that the metric is narrower than the claim being made."

Why this answer is stronger:
- it distinguishes metric from claim
- it uses concrete examples
- it sounds like research judgment

> **Saying it out loud.** Because the metric is almost always narrower than the claim someone is making with it. Perplexity can improve because the model got better at predicting boilerplate, and that tells you nothing about whether it's more factual or follows instructions better. Retrieval recall can jump while answer quality sits flat, because the generator was already ignoring the evidence it had. So when I see a gain I ask three things: what behavior does this metric actually track, what does it ignore, and does the gain survive across slices and random seeds. The number that usually settles it is the seed variance — if the improvement is smaller than the spread across seeds, there's no result there yet.

## 4. "How Would You Debug a Model That Is Not Learning?"

Weak answer:

"I would lower the learning rate and inspect gradients."

Stronger answer:

"I would debug it in layers instead of guessing. First I would verify the data and labels, because if the target or split logic is wrong, optimizer tuning is irrelevant. Then I would check tensor shapes and the loss definition, especially whether the model is outputting logits or probabilities and whether the loss expects one or the other. After that I would inspect gradient flow: are the gradients zero, NaN, or just never applied because the parameters are frozen or missing from the optimizer? Only after those checks would I start tuning learning rate or clipping, because hyperparameters are often blamed for problems that are actually caused by semantics or numerics."

Why this answer is stronger:
- it is procedural
- it narrows the search space
- it shows debugging maturity

> **Saying it out loud.** I'd go in layers rather than start guessing at hyperparameters, because learning rate gets blamed for a lot of bugs that aren't about learning rate. Data and labels first — if the target column or the split logic is wrong, nothing downstream matters. Then shapes and the loss definition, especially whether the model is emitting logits or probabilities and whether the loss function expects the other one, which is a silent and extremely common bug. Then gradient flow: are gradients zero, are they NaN, or are the parameters simply not in the optimizer because somebody froze them. Only then do I touch the learning rate. And the fastest single test is overfitting one batch — if the model can't drive loss to zero on ten examples, it's a bug, not a tuning problem.

## 5. "Why Is PagedAttention Useful?"

Weak answer:

"It reduces KV-cache memory."

Stronger answer:

"Plain KV caching avoids recomputing past keys and values, but it also creates a large growing memory object for every active request. If a serving engine allocates that memory naively as large contiguous buffers, you get waste and fragmentation, especially when requests have different lengths and some terminate early. Paged attention fixes the memory-management side of the problem by storing KV cache in fixed-size blocks and using a block table to map logical sequence positions to physical memory blocks. The actual attention semantics stay the same, but memory becomes much easier to reuse and schedule. That is why it improves serving efficiency: not because the model changed, but because the allocator and scheduler can use GPU memory much more effectively."

Why this answer is stronger:
- it explains the before and after
- it shows the problem plain KV cache leaves unsolved
- it distinguishes semantics from memory layout

> **Saying it out loud.** It's an allocator fix, not a model change — that's the sentence to lead with. Ordinary KV caching hands each request one big contiguous buffer sized for the longest output it might produce, and since almost no request hits that maximum, most of the memory sits reserved and unused. Paged attention borrows the operating system's trick: store the cache in fixed 16-token blocks with a table mapping logical positions to physical blocks, so you allocate as you go and different requests can share blocks for a common prefix. The attention math is completely unchanged. The result is roughly a two-to-four-times throughput improvement, and the failure it removes is fragmentation-driven preemption, where the scheduler evicts a live request and has to redo its prefill.

## 6. "Why Might MoE Fail Even If It Looks Better on Paper?"

Weak answer:

"Because routing is hard."

Stronger answer:

"Mixture-of-experts can look very attractive because total parameter count becomes much larger without activating all parameters on every token. But that headline hides several failure modes. The router has to send tokens in a useful way, experts need to stay sufficiently balanced so some do not collapse while others overload, and the distributed systems stack has to tolerate the communication and dispatch pattern. So MoE can absolutely be the right answer, but it is not a free parameter-efficiency upgrade. It is a different optimization and systems problem. If the infra is immature or the timeline is tight, a dense model may still be the better decision even if MoE looks superior in theory."

Why this answer is stronger:
- it translates theory into operations
- it shows why dense baselines still matter
- it avoids hype language

> **Saying it out loud.** Because MoE trades a modeling problem for a systems problem, and the systems problem is the one that kills projects. On paper you get a huge total parameter count while only activating a fraction per token, so the loss curve looks great for the FLOPs. In practice the router has to learn a sensible assignment, experts have to stay balanced or a few collapse while others saturate, and your distributed stack now has an all-to-all communication pattern it may not handle well. Memory scales with total parameters even though compute scales with active ones, so you don't get to serve it on the hardware the FLOP count suggests. If the infrastructure is immature or the deadline is real, a dense model is frequently the better decision even when MoE wins the paper comparison.

## 7. "What Makes an Ablation Believable?"

Weak answer:

"Change one thing at a time."

Stronger answer:

"Changing one thing at a time is the starting principle, but a believable ablation also requires that the surrounding setup stays comparable in the ways that matter. If one model gets a different optimizer, longer training, better data curation, and a different decoding setup, then isolating the architecture claim becomes almost impossible. I also want low-noise evaluation, stable rankings over time, and metrics that match the intended use case. So the real goal of an ablation is not to produce another table. It is to support a causal statement about what actually drove the improvement."

Why this answer is stronger:
- it turns a slogan into a standard
- it includes evaluation noise and comparability
- it sounds like someone who has seen confounded experiments

> **Saying it out loud.** An ablation is believable when the only thing that differs is the thing you're claiming caused the difference — and that's much harder than "change one variable," because the confounders hide in the surroundings. If the new model also got a longer schedule, a cleaner data mix, or different decoding settings, the architecture claim is unsupported no matter how good the table looks. I also want the evaluation itself to be low-noise and the rankings to be stable across seeds, because otherwise you're reporting variance. The standard I'd state out loud: an ablation exists to support a causal statement about what drove the gain, not to fill a table — and if the effect is inside the seed-to-seed spread, there isn't one yet.

## 8. How to Use These Narratives

Do not memorize these word for word.

Use them to learn the shape of a strong answer:
- start with the real problem
- explain the mechanism
- connect to trade-offs
- mention the important limitation

If you can do that consistently, your answers will sound much more complete even when the interviewer keeps changing topic.

> **Saying it out loud.** The reason to practice these aloud rather than read them is that the gap between knowing something and saying it well is enormous, and it only closes with reps. The shape to internalize is always the same: lead with the real problem, explain the mechanism in one or two plain sentences, then land on a tradeoff, a failure mode, or a number. Avoid hype words — "powerful," "state of the art" — because they signal that you're describing rather than reasoning. And time yourself: most of these should take sixty to ninety seconds spoken, and running long is itself a scored failure, because an interviewer who has to interrupt you learns you can't judge what matters.
