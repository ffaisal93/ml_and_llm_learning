# Frontier Training Playbook — Interview Grill

> 50 questions on training methodology, scaling laws, data curation, stability, ablations. Drill until you can answer 35+ cold.

---

## A. Scaling laws

**1. State Kaplan's scaling law form.**
$L(N, D) = E + A/N^\alpha + B/D^\beta$. Loss as power-law in params and data.

> **Saying it out loud.** Kaplan's result is that loss falls off as a power law in both parameters and data, plus an irreducible floor — so loss equals E plus A over N to the alpha plus B over D to the beta. The E term is the part you can never train away, the entropy of language itself. What makes it useful isn't the exact exponents, it's that a power law is straight on a log-log plot: fit two small runs and you can extrapolate the loss of a run a hundred times bigger. That predictability is why labs are willing to commit tens of millions of dollars to a single training run before seeing a single result.

**2. State the Chinchilla compute-optimal allocation.**
$N \propto C^{0.5}$, $D \propto C^{0.5}$. Roughly 20 tokens per parameter at training optimum.

> **Saying it out loud.** Chinchilla says that for a fixed compute budget you should grow parameters and data at the same rate — both scale as roughly the square root of compute, which works out to about 20 tokens per parameter. The reason this landed so hard is that everyone had been doing it wrong: the field was building bigger and bigger models on roughly fixed data. The famous demonstration was that a 70B model trained on 1.4 trillion tokens beat a 280B model trained on 300 billion, using the same compute. So the lesson is that under-training a big model is a worse mistake than fully training a smaller one.

**3. Compute approximation $C \approx ?$**
$6 N D$ FLOPs for transformer training (forward + backward).

> **Saying it out loud.** Six times parameters times tokens, and the six is worth being able to derive: roughly two FLOPs per parameter per token for the forward pass — one multiply, one add — and about twice that again for the backward, since you compute gradients with respect to both the inputs and the weights. Two plus four is six. It ignores attention's quadratic term, which is fine until context gets very long. It's the single most useful back-of-envelope in the field: Llama 3 70B on 15 trillion tokens is 6 times 7e10 times 1.5e13, about 6 times 10 to the 24 FLOPs.

**4. Why is GPT-3 (175B, 300B tokens) considered Chinchilla-suboptimal?**
Severely under-trained. Optimal would be ~3.5T tokens for that param count.

> **Saying it out loud.** GPT-3 is the canonical under-trained model — 175 billion parameters on only 300 billion tokens, which is under 2 tokens per parameter when the compute-optimal ratio is about 20. To be Chinchilla-optimal at that size it needed something like 3.5 trillion tokens. The fair thing to add is that Chinchilla came out after GPT-3, so this is hindsight, not incompetence. And the practical consequence is that GPT-3 is expensive to serve for the quality it delivers, which is exactly the mistake the industry then over-corrected on.

**5. Why do modern Llama models train past Chinchilla?**
Inference cost dominates lifetime cost. Smaller over-trained models are cheaper to serve.

> **Saying it out loud.** Because Chinchilla optimises the wrong thing for a product. It minimises loss for a fixed *training* budget, but the money you actually spend over a model's life is inference, and inference cost is set by parameter count, not by how many tokens you trained on. So if you're going to serve billions of requests, it's rational to take a smaller model and train it far past the optimum — Llama 3 8B at 15 trillion tokens is roughly 1,875 tokens per parameter, almost a hundred times Chinchilla. You're paying more in training to be permanently cheaper in serving, and the returns do diminish, they just don't stop.

**6. Llama 3 8B trained on 15T tokens — tokens per param?**
~1875. Far past Chinchilla.

> **Saying it out loud.** About 1,875 tokens per parameter — 15 trillion over 8 billion — against a Chinchilla optimum of 20. That's roughly 90 times past compute-optimal, and it's deliberate. The model is far more expensive to train than it needed to be for its loss, and far cheaper to serve than a compute-optimal model of equivalent quality. The gains were still real at 15T tokens, which is the empirical point: the power law flattens but doesn't stop, so the stopping point is an economics decision, not a science one.

**7. Compute-optimal Chinchilla-style allocation for $C = 10^{24}$ FLOPs?**
Use $C = 6 N D$ with $D \approx 20 N$ (the Chinchilla ratio): $C = 6 \cdot N \cdot 20 N = 120 N^2 \Rightarrow N^2 = 10^{24}/120 \approx 8.3 \times 10^{21} \Rightarrow N \approx 91$B, $D \approx 1.8$T. (Note the ratio is $D = 20N$, not $20D = N$.)

> **Saying it out loud.** I'd do this out loud in three steps. Compute is 6ND and Chinchilla says D is about 20N, so C equals 120 N squared. That gives N squared equals 10 to the 24 over 120, about 8.3 times 10 to the 21, so N is around 91 billion parameters. Then D is 20N, so about 1.8 trillion tokens. The trap in this question is inverting the ratio — it's 20 tokens per parameter, so D is much bigger than N, and if you find yourself with more parameters than tokens you've flipped it.

**8. What does scaling law imply about predictability?**
Loss is predictable from compute. Lets you forecast capability before training.

> **Saying it out loud.** The deep implication is that you can know roughly how good a model will be before you build it. You run a ladder of small models, fit the power law, and extrapolate — and it holds across several orders of magnitude, which is remarkable. That's what makes a hundred-million-dollar training run a business decision rather than a gamble. The caveat to name is that loss extrapolates well but *capabilities* don't necessarily: smooth loss curves can hide capabilities that appear abruptly, so you can predict the perplexity and still be surprised by what the model can do.

---

## B. Architecture choices that ship

**9. What attention variant does Llama 2/3 use?**
Grouped-Query Attention (GQA).

> **Saying it out loud.** Grouped-Query Attention, from Llama 2 70B onwards and across all of Llama 3. The 70B uses 8 KV heads against 64 query heads, so eight query heads share each key-value pair. It's purely an inference decision — the training cost barely changes — and it cuts the KV cache by 8x, which is what makes long-context serving affordable. It's now effectively the industry default; Qwen, Mistral and most others ship it too.

**10. Why GQA over MHA?**
Smaller KV cache → faster inference, lower memory. Almost no quality loss at GQA-8.

> **Saying it out loud.** Because the KV cache is what actually fills your GPU at serving time, and most of it is redundant. Full multi-head attention gives every query head its own key and value head; GQA lets a group of query heads share one, so with 8 KV heads in a 64-head model you cut cache memory by 8x. The quality cost at GQA-8 is close to unmeasurable, which is the whole point — MQA, the extreme version with a single shared KV head, does measurably degrade. So GQA-8 is the sweet spot on a curve between cache size and head diversity, and the payoff shows up as a bigger max batch size, which is the thing that actually determines serving throughput.

**11. Why is RMSNorm preferred over LayerNorm?**
Drops mean centering — slightly faster, equally good empirically.

> **Saying it out loud.** RMSNorm just drops the mean-centering step — it divides by the root-mean-square of the activations and skips subtracting the mean and the bias term. It turns out that centering was doing almost nothing for transformers, so you get the same quality for slightly fewer operations and one less reduction over the feature dimension. That last part matters more than it sounds at scale: a reduction is a synchronisation point in the kernel, so removing one is a real few-percent throughput win across every layer.

**12. Pre-LN vs Post-LN at scale?**
Pre-LN: stable, default for modern LLMs. Post-LN: hard to train deep.

> **Saying it out loud.** Pre-LN puts the normalisation inside the residual branch, so there's a clean identity path from the input all the way to the output with nothing scaling it. That means gradients reach the early layers intact and you can train very deep models without an elaborate warmup. Post-LN normalises after the residual add, which puts a LayerNorm on the main path and makes gradient magnitudes depend on depth — it can reach slightly better final loss, but it's notoriously hard to train past a few dozen layers. Every modern LLM is Pre-LN, often with an extra normalisation before the output head.

**13. Default activation in modern FFN?**
SwiGLU. The gating doubles matmul count (vs vanilla 2-matmul FFN, SwiGLU uses 3); modern recipes (Llama, Mistral) compensate by scaling FFN hidden dim down by $\frac{2}{3}$ (so $\frac{8}{3}d$ instead of $4d$), keeping parameter count roughly constant. Consistently better in evaluation.

> **Saying it out loud.** SwiGLU — a gated variant where one linear projection produces the values and a second, passed through a Swish, produces a gate that multiplies them. That's three matrices instead of the usual two, so to keep parameter count constant, recipes shrink the hidden dimension from 4d to 8/3 d. There's no clean theory for why it's better, it just consistently is by a small margin across scales — Llama, Mistral, PaLM all use it. Say the two-thirds detail out loud, because it's the part that shows you've actually read a config file rather than a paper abstract.

**14. Default positional encoding?**
RoPE. Allows context extension via NTK / YaRN.

> **Saying it out loud.** RoPE — rotary position embeddings — which rotate the query and key vectors by an angle proportional to position, so the attention dot product ends up depending only on the *relative* distance between tokens. That's structurally nicer than adding a position vector, and it's why RoPE extrapolates and interpolates so well. The practical payoff is context extension: because position enters as a set of frequencies, you can rescale those frequencies with NTK-aware scaling or YaRN and stretch an 8K model to 128K with a short fine-tune instead of retraining from scratch.

**15. MLA — what's the innovation?**
Compresses KV via low-rank latent projection. Strongest KV cache reduction. DeepSeek-V2/V3.

> **Saying it out loud.** MLA compresses the keys and values into a shared low-rank latent vector, caches that, and projects back up when it's needed — so instead of caching per-head K and V you cache one much smaller thing. It's the most aggressive KV reduction that's been made to work at frontier scale, around 10x versus multi-head attention, versus roughly 8x for GQA but with a different quality profile. DeepSeek-V2 introduced it and V3 shipped it. The tradeoff is complexity: it interacts awkwardly with RoPE, so DeepSeek splits off a small separate rotary component, which is a real implementation cost that GQA doesn't have.

**16. Dense vs MoE — main trade-off?**
Dense: simpler, smaller total params for same compute. MoE: bigger total capacity, lower active compute, but routing/load-balancing/communication complexity.

> **Saying it out loud.** Dense is simple and predictable: every parameter runs on every token, so your compute and your capacity are the same number. MoE decouples them — you can hold ten times the parameters and only activate a small fraction per token, so you buy capacity at roughly constant FLOPs. What you pay for it is systems complexity: routing decisions made at runtime, load imbalance that silently drops tokens when the capacity factor is hit, and two all-to-all collectives per layer, which is the worst-scaling collective across nodes. So the trade is memory and network complexity for parameter capacity per FLOP.

**17. When is MoE worth the complexity?**
When total capacity matters more than wall-clock simplicity. Frontier flagship models often MoE.

> **Saying it out loud.** MoE is worth it when you're capacity-limited rather than FLOP-limited, and when you have the infrastructure maturity to handle it. If you're serving at huge scale and total knowledge matters more than simplicity, MoE wins — DeepSeek-V3 gets 671B parameters of capacity for about 37B of active compute per token. But if your cluster's interconnect is mediocre, the all-to-all will eat the gain, and if your team hasn't done it before, load-balancing bugs will cost you months. The honest framing is that MoE is a better model per FLOP and a worse model per engineer-hour.

---

## C. Data

**18. Why is dedup the most reliable data improvement?**
Prevents memorization of duplicates; reduces effective epochs on common substrings; consistent quality boost.

> **Saying it out loud.** Because duplicates quietly break the assumption that one epoch is one epoch. A passage that appears a thousand times in your corpus gets trained on a thousand times, so the model memorises it verbatim instead of generalising, and your effective data diversity is much lower than your token count suggests. Deduplication is the rare intervention that helps consistently across every scale and every benchmark, with no tuning and no downside — most data interventions are finicky, this one just works. It also directly reduces verbatim regurgitation, which is a real legal and safety concern, not just a quality one.

**19. What's MinHash used for in data curation?**
Near-duplicate detection at scale. Approximates Jaccard similarity efficiently.

> **Saying it out loud.** MinHash is how you find near-duplicates without comparing every document to every other one, which at trillions of tokens is impossible. The trick is that if you hash a document's shingles many times and keep the minimum of each hash, the probability that two documents agree on a given minimum equals their Jaccard similarity — so a short signature estimates overlap. Then you bucket documents by bands of that signature with locality-sensitive hashing, and only compare within buckets. It turns a quadratic problem into a near-linear one, which is the only reason web-scale dedup is feasible at all.

**20. What's a quality classifier?**
Binary classifier: "Wikipedia-like vs random web." Filter web data above a threshold.

> **Saying it out loud.** A quality classifier is a cheap binary model trained to tell 'good' text from 'random web text' — the classic setup uses Wikipedia, books and reference pages as positives and a random web crawl sample as negatives, then scores everything and keeps what's above a threshold. It's crude but it works remarkably well, and it's what turned Common Crawl from unusable into the backbone of modern pretraining. The failure mode to name is that you're encoding a specific notion of quality: filter too hard toward Wikipedia-like prose and you strip out code, dialogue, and non-English text, and you'll see it later as a weirdly narrow model.

**21. Why blend new data with general during mid-training?**
Prevents catastrophic forgetting. Typical 90% general / 10% new.

> **Saying it out loud.** Because a model trained only on the new focused data forgets what it already knew — that's catastrophic forgetting, and it's fast and severe. If you push pure math data through a general model, math goes up and everything else quietly degrades. The standard fix is a blend, something like 90% general to 10% new, so the general distribution keeps anchoring the weights while the new data shifts them. The number to remember is that the new-data fraction is usually surprisingly small; the temptation is to crank it up, and that's exactly what breaks the model.

**22. Test-set contamination — why does it matter?**
Public benchmark answers leak into training data over time. Inflates reported numbers without real progress. Always check.

> **Saying it out loud.** Because benchmark questions and their answers leak onto the public web, get crawled, and end up in your training data — at which point your reported score measures memorisation, not capability. It gets worse over time: the older and more famous a benchmark is, the more contaminated it is, so the numbers everyone quotes are the least trustworthy ones. You defend against it by n-gram or MinHash matching your training corpus against every eval set and decontaminating, and by holding out private evals the internet has never seen. The signal that something's wrong is a model that scores brilliantly on MMLU and disappoints in actual use.

**23. Why filter on perplexity from a small reference LM?**
Drops gibberish + low-quality samples that the reference LM finds unlikely.

> **Saying it out loud.** You take a small, cheap language model trained on known-good text and score every candidate document by how surprising it finds it. Very high perplexity usually means gibberish — OCR noise, machine translation garbage, keyword spam — and dropping it is a reliable win. The subtlety is that you generally filter from both ends: extremely *low* perplexity is also suspicious, because that's boilerplate, templated pages and repeated text. So it's a band, not a threshold, and the risk of tuning it too tightly is that you filter out genuinely unusual but valuable text, like technical writing or poetry.

**24. Synthetic data — when is it useful?**
Instruction tuning, math reasoning chains, code completion. Risk: hallucinated facts amplify.

> **Saying it out loud.** Synthetic data is most useful where the thing you want is rare on the web but easy to verify — math with checkable answers, code you can run against tests, instruction-response pairs in a format nobody writes naturally. That's why post-training leans on it so heavily. The danger is that when you can't verify, you're distilling one model's errors into the next one: hallucinated facts get amplified rather than filtered, and diversity collapses because the generator has its own stylistic tics. So the rule of thumb is that synthetic data works in proportion to how cheaply you can check it.

---

## D. Hyperparameters and recipes

**25. Standard $\beta_2$ for LLM AdamW?**
0.95 (lower than the default 0.999). More responsive variance estimation.

> **Saying it out loud.** 0.95, notably lower than PyTorch's 0.999 default. Beta-2 controls how long a memory the variance estimate keeps, so 0.999 averages over roughly the last thousand steps — and at LLM scale, with gradient statistics that shift as training progresses, that's too sluggish. Dropping to 0.95 means about a 20-step window, which reacts faster to changing gradient scale and, importantly, recovers faster after a loss spike. The connection worth naming is that a stale variance estimate is one of the classic reasons a spike turns into a divergence instead of a bump.

**26. Standard peak LR for billion-scale LLM?**
$\sim 3 \times 10^{-4}$.

> **Saying it out loud.** Around 3 times 10 to the minus 4 for a billion-scale model, and the key point is that it scales down as models get wider — something like 1.5e-4 for a 70B. That inverse relationship with width is exactly what muP formalises. It's always paired with warmup and a cosine decay, and the failure mode of getting it wrong is asymmetric: too low just wastes compute, too high gives you loss spikes and possibly a run that never recovers. So at frontier scale people tune conservatively low, because a divergent run costs far more than a slightly suboptimal one.

**27. Standard warmup duration?**
Few thousand steps (low single digit thousands).

> **Saying it out loud.** A few thousand steps — usually low single-digit thousands, or a percent or two of total training. The reason you need it at all is Adam's variance estimate: at step one it's based on essentially no data, so the update sizes are unreliable and a full learning rate can blow the model apart before it's learned anything. Warmup also matters more the bigger your batch, because large-batch training pushes toward larger learning rates that need more easing in. Skip warmup at scale and the classic symptom is a loss spike in the first few hundred steps that never fully recovers.

**28. Cosine decay schedule — to what fraction of peak?**
~10% of peak typically.

> **Saying it out loud.** Down to about 10% of peak, and the decay shape matters more than people expect. The intuition is a coarse-to-fine search: early on you want big steps to find the right basin, late on you want small ones to settle into it. Empirically cosine consistently beats a constant rate or a linear decay, and stopping the decay at 10% rather than zero leaves the model in a state you can keep training from. The gotcha is that a cosine schedule bakes in your total step count — if you decide to train longer partway through, the schedule is wrong and you've lost some of the benefit.

**29. Effective batch size in tokens for flagship?**
Millions to tens of millions of tokens (via gradient accumulation across DP).

> **Saying it out loud.** Millions to tens of millions of tokens per step, which is far beyond what fits anywhere, so it's built out of data parallelism plus gradient accumulation. The reason to go so big is gradient noise: at these model sizes a small batch gives an estimate too noisy to make stable progress, and large batches also just parallelise better across thousands of GPUs. The tradeoff is diminishing returns past a critical batch size — beyond it you're spending twice the compute per step for much less than twice the progress. And large batches need a proportionally larger learning rate and longer warmup, which is where the instability comes from.

**30. What's muP and why is it used at frontier labs?**
Maximal Update Parameterization: optimal LR is invariant to model width. Tune small, deploy big without re-sweeping LR.

> **Saying it out loud.** muP re-parameterises the initialisation and per-layer learning rates so that the optimal learning rate stops depending on model width. That's a big deal economically: normally you'd have to sweep hyperparameters at the scale you're training at, which for a flagship run is unaffordable, so people guess. With muP you sweep on a small proxy model, transfer the winning learning rate directly to the big one, and it's still optimal. The catch is that it's a genuine change to your parameterisation — initialisation scales, attention scaling, output multipliers all shift — so retrofitting it onto an existing codebase is more invasive than it sounds.

---

## E. Training stability

**31. What causes loss spikes?**
Bad batches (OOD), numerical instability, optimizer state mismatch.

> **Saying it out loud.** Three things, usually. A bad batch — something out of distribution, a corrupted shard, an unusually long or repetitive document — produces a huge gradient. Numerical instability, where attention logits grow until low-precision arithmetic misbehaves. Or optimizer state that's out of sync with the weights, which is especially common right after a restart from checkpoint. The pattern to name is that spikes get more frequent as models get bigger and deeper, and the difference between an annoyance and a dead run is whether the loss comes back down within a few hundred steps.

**32. Standard fix for loss spikes?**
Gradient clipping (1.0), BF16 over FP16, restart with bad batches skipped.

> **Saying it out loud.** Gradient clipping at global norm 1.0 is the first line of defence and it's essentially free — it bounds the damage any single bad batch can do. BF16 over FP16 removes a whole class of overflow problems, since BF16 has FP32's exponent range. And then the operational fix: when a spike doesn't self-recover, you roll back to the last good checkpoint and skip forward past the offending data. That last one is the answer that signals you've actually run a big job — automated spike detection with rollback and data skipping is standard practice, not a hack.

**33. NaN in attention — common cause?**
FP16 overflow in softmax. Fix: BF16, or compute attention in higher precision.

> **Saying it out loud.** Almost always FP16 overflow in the softmax. Attention logits are query-dot-key scaled, and as training progresses those can grow large; once a logit exceeds about 65,000 in FP16 you get infinity, the exponential gives you infinity over infinity, and NaN propagates through the residual stream into the weights on the very next step. The fixes are BF16, which simply doesn't overflow there, or accumulating the softmax in FP32 even when everything else is half precision — which FlashAttention does anyway. If it's not the softmax, check the LayerNorm denominator for a near-zero variance.

**34. Why is BF16 preferred over FP16 at scale?**
FP32-equivalent exponent range. No need for loss scaling. More stable.

> **Saying it out loud.** Because at scale you care about range far more than precision. BF16 keeps all eight of FP32's exponent bits and sacrifices mantissa, so it spans the same dynamic range — no gradient underflow, no softmax overflow, and no loss scaling machinery to maintain. FP16 has better precision in its narrow band, but that band is the problem: you need loss scaling, and dynamic loss scalers themselves fail in interesting ways, skipping steps or oscillating. Fewer mantissa bits turns out not to matter because SGD is a noisy process anyway. That's the tradeoff in one line: BF16 trades precision you don't need for range you do.

**35. What's z-loss?**
Adds penalty on $\log Z$ to discourage unbounded logit magnitudes. Loss-level intervention.

> **Saying it out loud.** Z-loss adds a small penalty on the square of the log-partition-function — the log of the softmax denominator — which pushes the model to keep its logits from drifting to large magnitudes. The reason you want that is that big logits are the precursor to numerical trouble: they make the softmax saturate and they push you toward overflow in low precision. It's cheap, the coefficient is tiny, typically 1e-4, and it's basically free stability insurance. PaLM popularised it and it's standard in large runs now.

**36. What's logit softcapping?**
Forward-pass smooth bound: $c \tanh(\mathrm{logits}/c)$. Bounds magnitudes without clipping.

> **Saying it out loud.** Softcapping bounds the logits smoothly in the forward pass: you take c times tanh of logits over c, so anything small passes through nearly unchanged and anything large asymptotes to c. Compare that with hard clipping, which has zero gradient beyond the threshold and kills learning for exactly the tokens that most need it. It's a forward-pass intervention, whereas z-loss is a loss-pass one, so they attack the same problem from different directions. Gemma 2 uses it on both attention logits and the final logits; the cost is a slightly awkward interaction with FlashAttention kernels, which have to support it explicitly.

**37. QK normalization?**
Normalize Q and K before attention dot product. Prevents extreme attention scores → softmax instability.

> **Saying it out loud.** QK-norm applies an RMSNorm to the queries and keys before the attention dot product, which bounds their magnitudes and therefore bounds the attention scores. That directly kills the failure mode where a couple of attention logits run away, the softmax saturates into a one-hot, and gradients through attention vanish. It's become common in large runs — ViT-22B popularised it and several recent LLMs use it — because it removes attention as a source of instability almost entirely. Cost is negligible: two extra normalisations per attention block.

**38. What's catastrophic forgetting in mid-training?**
New focused data overwrites general knowledge from pre-training.

> **Saying it out loud.** Catastrophic forgetting is what happens when you continue training on a narrow distribution and the weights drift to fit it, overwriting general capability. It's fast — you can see general benchmarks degrade within a few billion tokens of pure domain data — and it's often invisible if you're only watching the metric you're trying to improve. That's the real trap in mid-training: math goes up, you declare success, and you don't notice that the model's coding and multilingual ability quietly dropped. The lesson is to always evaluate the things you're *not* trying to improve.

**39. Mitigation for catastrophic forgetting?**
Blend new + general data; replay buffer; weight regularization (e.g., EWC).

> **Saying it out loud.** The simplest and most effective fix is blending: keep 80 or 90 percent of the general distribution in the mix so the weights stay anchored while the new data nudges them. Replay is the same idea done explicitly, holding a buffer of pre-training data and interleaving it. Beyond that, you can constrain how far the weights can move — a KL penalty against the base model, or something like EWC that penalises movement in the parameters the old task cared about, though those are more common in research than in production LLM recipes. And the mundane lever that matters most is learning rate: mid-training at a much lower rate than pre-training does most of the work.

**40. Why do hardware failures matter at training scale?**
Single bad GPU can hang or corrupt training. Per-step health checks; auto-checkpoint and resume.

> **Saying it out loud.** Because at a thousand GPUs, hardware failure is routine, not exceptional, and the synchronous nature of training means one bad card affects everyone. If a GPU throws an ECC error mid-collective, the job doesn't crash — it hangs, because NCCL collectives are barriers, so every other rank waits until the timeout fires while you burn the full cluster's cost. A thermally throttled card that's 10% slow taxes the whole job 10%, because everyone waits at the gradient sync. So you run health checks before starting, keep a watchdog reporting each rank's last collective, checkpoint often enough that losing an hour is survivable, and automate replacement — the goodput of a big run is mostly an operations achievement.

---

## F. Mid-training and post-training

**41. What's mid-training?**
Curated quality boost or domain emphasis after general pre-training. Examples: math/code injection, long-context extension, recency.

> **Saying it out loud.** Mid-training is the stage between general pre-training and post-training: you keep doing next-token prediction, but on a deliberately curated mix rather than the raw web distribution. Typical uses are injecting math and code, extending context from 8K to 128K, refreshing recency, or boosting an under-represented language. You do it at the end because it's expensive to have that high-quality data in the whole run, and because the model learns fastest from it once it already has general competence. The thing to watch is forgetting, which is why the new data is always blended with general data rather than used alone.

**42. Why long-context extension as a separate stage?**
Pre-training at long context is expensive. Train at 8K, extend to 128K with curated long-context data + RoPE rescaling.

> **Saying it out loud.** Because attention cost grows quadratically with sequence length, so pre-training everything at 128K would be wildly wasteful when the vast majority of your documents are short. The efficient recipe is to do the bulk of training at 4K or 8K, then spend a small fraction of tokens — often under 1% — on a long-context extension stage with curated genuinely-long documents and rescaled RoPE frequencies. It works because the model has already learned language; it only needs to learn to use the longer positions. The failure mode is passing the needle-in-a-haystack test while still degrading on tasks that need real reasoning across the whole window, so evaluate on both.

**43. NTK / YaRN — what do they do?**
Scale RoPE frequencies to extend usable context length without re-pre-training.

> **Saying it out loud.** Both are ways of rescaling RoPE's rotation frequencies so positions the model never saw during training still land in a range it understands. Naive position interpolation just squashes all frequencies uniformly, which works but blurs the high-frequency components that encode fine local ordering. NTK-aware scaling fixes that by scaling frequencies unevenly — leave the high-frequency dimensions mostly alone, stretch the low-frequency ones that carry long-range information. YaRN refines this further with a per-dimension ramp and a temperature adjustment on attention, and gets to 128K with a very short fine-tune. The headline is that you extend context by changing how position is encoded, not by retraining.

**44. SFT vs preference optimization — what each adds?**
SFT: instruction format, basic capability. RLHF/DPO/GRPO: alignment, helpfulness, refusal calibration, fine-grained capability.

> **Saying it out loud.** SFT teaches the format and the basic shape of being an assistant — take an instruction, produce a helpful response, use this tool syntax, stop here. It's straightforward supervised learning on demonstrations, and it's remarkably sample-efficient because the capability is already latent in the base model; you're eliciting it, not creating it. Preference optimisation — RLHF, DPO, GRPO — does the part demonstrations can't: it teaches relative judgement, which of two acceptable answers is better, when to refuse, how confident to sound. The framing that scores is that SFT sets the distribution and preference optimisation sharpens within it, and you need both because writing a demonstration of 'slightly less sycophantic' is basically impossible.

**45. Why does post-training matter so much?**
Determines instruction following, tool use, refusal, reasoning format, preference behavior. Many "model capabilities" are really post-training capabilities.

> **Saying it out loud.** Because a huge fraction of what users perceive as model capability is actually post-training. The base model has the knowledge and the reasoning, but it's a text completer — it won't follow instructions, won't stop, won't refuse, won't use tools in your format. Post-training is what turns that into something usable, and the difference between two products built on similar base models is mostly here. The concrete version of the claim: reasoning models like o1 and R1 are largely a post-training result, the same pretrained substrate taught to spend tokens thinking, which is why capability jumps can now come without a new pretraining run.

**46. Reward hacking — what is it?**
Model finds easy way to maximize reward signal that doesn't correspond to good behavior. E.g., longer responses correlate with higher reward → model just makes everything longer.

> **Saying it out loud.** Reward hacking is when the model maximises your proxy instead of your intent — it finds a shortcut in the reward model that pays out without being genuinely better. The textbook example is length: human raters mildly prefer longer answers, so the reward model encodes that, and the policy discovers it can raise its score by padding everything. Sycophancy is the same phenomenon: agreeing with the user is rewarded, so the model stops disagreeing even when the user is wrong. The underlying reason is Goodhart's law — the reward model is trained on a finite sample, so it's only accurate near that distribution, and RL is very good at finding the places where it isn't.

**47. How do you detect reward hacking?**
Monitor: average response length growing? Sycophantic patterns? Performance on held-out tasks not in reward training? Drift from SFT distribution?

> **Saying it out loud.** Mostly by watching for the symptoms rather than trying to define hacking directly. Average response length creeping up is the single most reliable canary. Then agreement rate with the user, especially when the user is wrong — a planted-false-premise eval catches sycophancy well. Then held-out tasks that weren't in the reward model's training distribution, because a hacked policy improves on the reward and stalls or regresses on genuine capability. And the structural safeguard is the KL penalty against the SFT policy: if the KL is climbing while your held-out win rate is flat, you're buying reward with drift, and that's the moment to stop.

---

## G. Evaluation and ablation

**48. Two-tier ablation strategy?**
Small-scale (1B, 30B tokens) for fast iteration; mid-scale (10B, 200B tokens) for validation; flagship only for proven winners.

> **Saying it out loud.** You run a ladder, because ablations at flagship scale are unaffordable and ablations at tiny scale don't transfer. So: 1B models on 30 billion tokens for fast, cheap iteration where you can try dozens of ideas; 10B on 200 billion tokens to check that the survivors still win at a scale where the dynamics are more realistic; and only then does anything reach the flagship. The value of the ladder isn't just filtering — it's the *trend*: an intervention whose advantage shrinks as you go up the ladder is a warning, even if it's still winning. That's the depth-versus-width transfer problem, and the ladder is how you catch it before spending eight figures.

**49. Hold compute constant — why?**
Bigger models cost more. To attribute gains correctly, match flops between conditions.

> **Saying it out loud.** Because otherwise you can't tell whether a change helped or whether you just spent more compute. If your new architecture has more parameters or trains longer, of course it does better — that's not an ablation, that's a bigger model. So you fix the FLOP budget across conditions and let each one allocate it however it wants. The subtlety worth mentioning is what 'constant' means: matched training FLOPs is standard, but for anything you plan to serve you should also look at matched *inference* cost, because an intervention that wins on training FLOPs and loses on serving latency isn't a win for a product.

**50. Why multiple seeds for ablations?**
Variance is real at this scale. Single-seed gains often within noise.

> **Saying it out loud.** Because at these scales the run-to-run variance from seed alone is often as big as the effect you're measuring. Data order, initialisation and any nondeterminism in the kernels all move the final loss, and a 0.5% benchmark gain from one seed is very often noise. So you run three or more seeds per condition and report the spread, not just the mean. The uncomfortable implication is that most single-seed ablation results in the wild are unreliable, and the discipline of insisting on seeds is exactly what stops a lab from shipping a change that does nothing.

**51. Public benchmarks — risks?**
Contamination, prompt sensitivity, cherry-picking, statistical noise. Use private/held-out as ground truth.

> **Saying it out loud.** Four risks, and I'd name them in order. Contamination — the answers are on the web and probably in your training data. Prompt sensitivity — the same model can swing several points on MMLU depending on formatting and few-shot choice, so cross-lab comparisons are shaky. Cherry-picking — everyone reports the subset where they win. And plain statistical noise, since many benchmarks have only a few hundred items per subject, so the confidence interval is wider than the gaps being reported. The remedy is private held-out evals the internet has never seen, plus human preference comparison, and treating public numbers as a floor sanity check rather than ground truth.

**52. Held-out validation perplexity — what to track?**
Per-domain (web, code, math, books). Should monotonically decrease. Spikes = data quality issues.

> **Saying it out loud.** Track it per domain, not as one aggregate — web, code, math, books, and any language you care about, each on its own held-out shard. The reason is that a single number hides exactly the problem you're looking for: an aggregate that's decreasing can conceal one domain quietly degrading. The curves should be smooth and monotone, so any kink, spike or plateau in one domain is a data problem — a bad shard, a broken tokenizer path, a mid-training blend that's too aggressive — and it's usually the earliest warning you get, far earlier than downstream benchmarks show anything.

**53. Chatbot Arena / ELO ratings?**
Crowd-sourced head-to-head comparison. Less prone to gaming than single-model benchmarks.

> **Saying it out loud.** Arena is head-to-head: users send a prompt, get two anonymous responses, pick a winner, and the ratings get fit with an Elo-style model. Its advantage is that it's much harder to game — you can't train on the test set, because the test set is whatever users type tomorrow. Its weaknesses are equally real: it measures what anonymous users prefer, which rewards formatting, length and confident tone, and it under-weights expert domains where the average voter can't judge correctness. So it's the best available signal for general helpfulness and a poor one for capability in a specialised area.

**54. What's the "depth-vs-width transfer" issue in ablations?**
Architecture changes that help at small scale may hurt at large scale (or vice versa). Common confounder.

> **Saying it out loud.** It's the observation that an architectural change can win at 1B and lose at 70B, because different aspects of the architecture become binding at different scales. Anything that changes how the model scales with depth or width — normalisation placement, initialisation, attention scaling, learning-rate parameterisation — is a prime suspect, since small models are often width-limited and big ones depth-limited. It's the main reason ablation ladders exist and the main reason you look at the trend across scales rather than the winner at any one. And it's the strongest practical argument for muP, which is designed precisely so that hyperparameters transfer across width.

---

## H. Operational

**55. Common operational training failures?**
Dataloader stalls, storage bottlenecks, throughput drift, checkpoint corruption, seed inconsistency across TP, silently bad shards.

> **Saying it out loud.** The operational failures are the ones nobody puts in papers and everybody hits. Dataloader stalls and storage bottlenecks, where the GPUs are fine and starving. Throughput drift, where step time creeps up over hours from fragmentation or a throttling node. Checkpoint corruption, which you only discover when you need it. Seed inconsistency across tensor-parallel ranks, which silently breaks dropout. And silently bad shards — a chunk of the corpus that's empty, duplicated or mis-tokenised. The common thread is that none of these crash the job; they degrade it, which is why monitoring goodput and per-domain validation loss matters more than watching the training loss.

**56. Throughput drift — what's it usually?**
Network congestion, memory fragmentation, GPU thermal throttling, slow node.

> **Saying it out loud.** Usually something environmental rather than something in your code, because your code didn't change. The suspects in order: network congestion from another job on the shared fabric, memory fragmentation in the allocator making every allocation slower, a GPU thermally throttling, or one node that's degraded and dragging the collective. The way to find it is per-rank step-time telemetry, because an aggregate average hides a single slow rank completely — and with synchronous training, one rank 10% slow means the whole job is 10% slow. That's the diagnostic instinct to demonstrate: look at the distribution, not the mean.

**57. Why are dataloader bugs especially insidious?**
Loss curve looks normal but model trains on wrong/duplicated data. Hard to detect without monitoring data uniqueness.

> **Saying it out loud.** Because they don't look like bugs. The loss curve stays smooth and plausible while the model trains on duplicated data, skips a shard entirely, or sees the same order every epoch — there's no crash, no NaN, no obvious signal. You can burn a week of cluster time before anyone notices. The defences are direct instrumentation: hash a sample of tokens and check uniqueness over time, log the per-domain mixture ratio actually delivered against what you configured, and verify that a restart resumes at the right offset instead of quietly starting the epoch over. That restart-offset bug is the classic one — training silently re-runs data it's already seen.

**58. Frequency of checkpointing?**
Every few thousand steps typically. Async/local for speed; periodic sync to network FS.

> **Saying it out loud.** Every few thousand steps, tuned so that the compute you'd lose to a failure is bounded to something like half an hour. The implementation matters as much as the frequency: write asynchronously by snapshotting to host memory and flushing in the background, write to local NVMe first and upload to network storage after, and use a sharded format so every rank writes its own piece in parallel rather than funnelling through rank zero. The tension is checkpoint overhead against recovery cost, and the mistake people make is optimising the wrong side — a naive synchronous 100-gigabyte checkpoint stalls the entire cluster and can end up costing more than the failures it protects against.

**59. Restart on different topology — what's needed?**
Re-sharded checkpoints. DeepSpeed, TorchTitan, Megatron support this.

> **Saying it out loud.** You need checkpoints that aren't tied to the parallelism layout they were written with. If you saved with TP 8 and PP 16 and you're restarting on a smaller cluster with different degrees, the tensors are physically cut differently and something has to stitch and re-slice them. The clean answer is a topology-agnostic distributed checkpoint format that stores logical tensors with their sharding metadata, which DeepSpeed, TorchTitan and Megatron all support. It matters more than it sounds: hardware failures routinely force you onto a different node count, and without resharding you're stuck waiting for the exact cluster you started with.

**60. What does a senior answer to "how would you train Llama-class?" sound like?**
Goal first → constraints → conservative baseline → biggest trade-offs → ablation order → stability/infra risks → strongest claim you'd make.

> **Saying it out loud.** The shape of a senior answer is a decision process, not a parts list. Start with the goal and the constraint — what is this model for, what's the compute budget, what's the serving budget — because those decide everything downstream. Then give a conservative, boring baseline that you know works: dense transformer, GQA, RMSNorm, Pre-LN, SwiGLU, RoPE, BF16, AdamW with beta-2 at 0.95, cosine schedule. Then name the two or three real tradeoffs and how you'd resolve them with an ablation ladder — 1B, then 10B, then flagship — and say what would make you abandon each. Then the risks: loss spikes, dataloader bugs, hardware failure, contamination. And finish with the claim you'd actually stand behind, which should be narrower than you'd like. Junior answers list techniques; senior answers say what they'd do first and what would change their mind.

---

## Self-grading

If you can't answer 1-15, you don't know scaling laws / architecture choices. If you can't answer 16-35, you'll struggle on data / hyperparameter / stability questions. If you can't answer 36-55, frontier-lab methodology interviews will go past you.

Aim for 40+/60 cold.
