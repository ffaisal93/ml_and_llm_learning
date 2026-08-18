# LoRA & PEFT — Interview Grill

> 35 questions on parameter-efficient fine-tuning. Drill until you can answer 25+ cold.

---

## A. Foundations

**1. What is PEFT?**
Parameter-Efficient Fine-Tuning. Train a small fraction of parameters while keeping the base model frozen. Saves memory (no optimizer state for frozen weights), enables fine-tuning huge models on modest hardware.

> **Saying it out loud.** PEFT means you freeze the giant pretrained model and train a tiny number of new parameters instead. The big win isn't really the parameter count, it's the memory: frozen weights need no gradients and no optimizer state, and optimizer state with Adam is twice the size of the weights themselves. So you go from needing four copies of the model in memory to needing roughly one. That's the difference between renting a cluster and using one GPU.

**2. Why not just full fine-tuning?**
Memory. A 70B model needs ~140GB just for weights at fp16; full fine-tuning needs ~3–4x that for gradients + optimizer state. PEFT fits in much less memory.

> **Saying it out loud.** It's a memory problem before it's anything else. A seventy-billion-parameter model is about a hundred forty gigabytes just to hold in half precision, and full fine-tuning needs gradients plus Adam's two moment estimates on top, so you're looking at four times that or more. That's a multi-node job. PEFT keeps the base frozen so none of that extra state exists, and the same fine-tune fits on a single card.

**3. What are the major PEFT families?**
LoRA (low-rank weight updates), adapter modules (bottleneck MLPs in each layer), prefix/prompt tuning (trainable virtual tokens), IA³ (multiplicative scaling). LoRA dominates.

> **Saying it out loud.** Four families worth naming. LoRA adds a low-rank update to existing weight matrices. Adapters insert small bottleneck MLPs between layers. Prefix and prompt tuning prepend trainable vectors and change nothing about the weights. And IA-three just learns multiplicative scaling factors on activations. LoRA won because it's the only one that merges back into the base weights, so inference costs exactly nothing extra.

---

## B. LoRA

**4. What is LoRA?**
Hu et al. 2021. For each weight $W_0$, add a low-rank update $\Delta W = B A$ where $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}, r \ll \min(d, k)$. New forward pass: $y = W_0 x + B A x$. Train only $A, B$; freeze $W_0$.

> **Saying it out loud.** LoRA says the change you want to make to a weight matrix is low rank, so instead of learning the whole $d$-by-$k$ update, learn two skinny matrices whose product has rank $r$. The forward pass becomes the original weight times $x$ plus $B$ times $A$ times $x$, and you only train $A$ and $B$ while the base stays frozen. Because the update is just added on, you can fold it back into the original matrix when you're done. That mergeability is the whole reason LoRA beat adapters.

**5. How many parameters does LoRA use?**
$r \cdot (d + k)$ per matrix, vs $d \cdot k$ for full FT. For $d = k = 4096, r = 16$: 130K vs 16M (~125x reduction).

> **Saying it out loud.** Instead of $d$ times $k$ parameters you train $r$ times $(d + k)$. At a four-thousand-ninety-six square matrix with rank sixteen, that's about a hundred thirty thousand parameters versus sixteen million, roughly a hundred twenty-five fold reduction. Across a whole model you typically end up training well under one percent of the parameters. The number that actually sells it: a LoRA checkpoint is a few tens of megabytes, so you can store thousands of task-specific adapters for the cost of one base model.

**6. How is LoRA initialized?**
$A$: random small values (Kaiming or Gaussian). $B$: zero. So $\Delta W = B A = 0$ at init. The model behaves like the base model at start of training; LoRA effect grows during training.

> **Saying it out loud.** $A$ gets small random values and $B$ gets initialized to exactly zero, so their product is zero and the model at step zero is bit-for-bit the base model. Training then grows the update away from zero. If you flipped it and zeroed $A$ instead, it would also start at zero but the gradient to $B$ would be zero too, so nothing would move. So the asymmetry matters: one side random, one side zero, in that specific arrangement.

**7. Why initialize $B$ to zero specifically?**
If both $A$ and $B$ were random, the initial $\Delta W$ would be a random perturbation of $W_0$ — destroying pretrained capabilities at step 0. Zero-init $B$ keeps the base model intact at start.

> **Saying it out loud.** Because a random $\Delta W$ at step zero would be pure noise added to carefully pretrained weights, and you'd blow up the model's capabilities before the first gradient step. Zero-initializing $B$ means you start exactly at the pretrained model and improve from there, so fine-tuning is a strict refinement rather than a recovery from self-inflicted damage. The concrete symptom of getting this wrong is a big loss spike at the start of training that may never fully recover.

**8. What's the α scaling factor?**
$y = W_0 x + (\alpha/r) \cdot B A x$. The $\alpha/r$ scaling decouples LR sensitivity from rank choice. Default in HuggingFace PEFT: $\alpha = 2r$.

> **Saying it out loud.** Alpha is a fixed scaling constant: the update gets multiplied by alpha over $r$ before being added. It's not learned, it's a knob that decouples how big the update is from what rank you chose. The default in the HuggingFace PEFT library is alpha equal to twice the rank. Practically, people tune the ratio rather than the raw numbers, and alpha over $r$ equal to two is a solid starting point.

**9. Why α/r scaling specifically?**
With it, the "magnitude" of the LoRA update is approximately constant in $r$. You can change $r$ without re-tuning the learning rate.

> **Saying it out loud.** Without the scaling, raising $r$ would automatically make the update bigger, since you're summing more rank-one terms, so every rank change would force a learning-rate retune. Dividing by $r$ keeps the effective magnitude roughly constant, so you can sweep rank independently of learning rate. That's the entire justification: it's there to make your hyperparameters separable. And it's why people quote the ratio alpha over $r$ rather than alpha alone.

**10. Where do you apply LoRA in a transformer?**
Most common: attention $Q$ and $V$ projections. More aggressive: $K, O$, and FFN. Empirically more matrices = better quality but more parameters. LLaMA-style: $Q, V$ (about 1% extra parameters).

> **Saying it out loud.** The minimal setup is LoRA on the query and value projections in attention, which is what the original paper used and costs about one percent extra parameters. Being more aggressive, you add the key and output projections and then the feedforward matrices. The empirical finding is that covering more matrices helps, and if you're going to spend a fixed parameter budget, spreading it thinly across all matrices beats concentrating it at a high rank on just $Q$ and $V$. The feedforward layers hold most of a transformer's parameters, so ignoring them leaves a lot on the table.

**11. Typical $r$ value?**
16–32 for most tasks. Smaller (4–8) for simple tasks or low memory budget. Larger (64–128) for complex domain shifts.

> **Saying it out loud.** Sixteen to thirty-two covers most tasks. Four to eight is fine for simple style or format adaptation, and sixty-four to a hundred twenty-eight is for serious domain shifts like a new technical field. The useful thing to say is that quality saturates fast: past rank sixty-four you're usually spending memory for nothing. And rank interacts with data volume, so a big rank on a thousand examples just overfits faster.

**12. Why does low-rank work?**
Aghajanyan et al. 2020 showed empirically that fine-tuning trajectories lie on a low-dimensional manifold. LoRA imposes this structure explicitly.

> **Saying it out loud.** The empirical finding is that fine-tuning doesn't need many degrees of freedom. Aghajanyan and colleagues showed you can constrain updates to a random low-dimensional subspace and still reach full fine-tuning performance, meaning the intrinsic dimension of the task is small. The interpretation is that pretraining already learned the features and fine-tuning is mostly reweighting them, not learning new ones. LoRA just imposes that structure explicitly instead of discovering it. And the honest caveat: it's an empirical observation, not a theorem, which is why it breaks down on genuinely new capabilities like a new language.

---

## C. QLoRA

**13. What is QLoRA?**
Dettmers et al. 2023. Quantize the base model to 4-bit (NF4); train LoRA in fp16 on top. Forward pass: dequantize on-the-fly for matmul. Backward: gradients flow only through LoRA. Massive memory savings.

> **Saying it out loud.** QLoRA quantizes the frozen base model down to four bits and trains fp16 LoRA adapters on top of it. During the forward pass, weights get dequantized block by block as they're needed for each matmul, so the memory savings are real rather than bookkeeping. Gradients only flow through the adapters, since the base isn't being trained anyway. That combination is what put seventy-billion-parameter fine-tuning on a single GPU.

**14. Three innovations of QLoRA?**
(a) NF4 quantization — info-theoretically optimal 4-bit for Gaussian weights. (b) Double quantization — quantize the quantization constants. (c) Paged optimizer — store optimizer state on CPU, page to GPU as needed.

> **Saying it out loud.** Three pieces. NF4, a four-bit format whose buckets are placed to be information-theoretically optimal for normally distributed weights. Double quantization, which quantizes the per-block quantization constants themselves, saving roughly another half bit per parameter. And paged optimizers, which push optimizer state to CPU memory and page it back on demand so a gradient spike doesn't out-of-memory your run. The middle one sounds like a rounding detail but at seventy billion parameters half a bit each is gigabytes.

**15. What's NF4?**
NormalFloat 4-bit. Quantization buckets chosen to be info-theoretically optimal for normally-distributed weights (which neural network weights approximately are). Better than uniform INT4 because it allocates more buckets to common values.

> **Saying it out loud.** NF4 is NormalFloat, a four-bit data type whose sixteen levels are placed at the quantiles of a normal distribution rather than spread uniformly. That matters because neural network weights really are roughly Gaussian, so uniform INT4 wastes most of its levels on the tails where almost no weights live. NF4 puts the resolution where the mass is. It's information-theoretically optimal under the Gaussian assumption, and empirically it beats INT4 at the same bit width.

**16. Why doesn't QLoRA hurt quality much?**
The base is frozen — quantization noise is fixed. LoRA fine-tunes "on top of" the quantized base, learning to compensate for quantization noise while learning the new task. Net quality close to fp16 LoRA.

> **Saying it out loud.** Because the base is frozen, the quantization error is a fixed, deterministic distortion rather than noise that accumulates. And the LoRA adapters are trained on top of the already-quantized model, so during training they learn to compensate for that distortion at the same time as learning the task. Effectively the quantization error becomes part of the thing you're fitting around. The result lands within about a point of fp16 LoRA on standard benchmarks, which is a remarkable deal for a four-times memory saving.

**17. Memory savings of QLoRA?**
A 70B model: 140GB at fp16. QLoRA: ~35GB (4-bit weights) + small overhead for LoRA adapters + activations. Fits on a single 80GB A100.

> **Saying it out loud.** A seventy-billion model is roughly a hundred forty gigabytes in fp16 and about thirty-five in four bits, and the adapters plus activations add only a few more. That gets you onto a single eighty-gigabyte A100, which is the headline result of the paper. The thing to add is that memory drops fourfold but speed doesn't improve, since you're paying dequantization overhead on every matmul, so QLoRA is typically slower per step than plain LoRA. You're buying feasibility, not throughput.

---

## D. Other PEFT methods

**18. What are adapter modules?**
Houlsby et al. 2019. Insert small bottleneck MLPs in each transformer block: $\text{down-project} \to \text{activation} \to \text{up-project} + \text{residual}$. ~0.5–3% of total parameters. Replaced by LoRA in production.

> **Saying it out loud.** Adapters are small bottleneck MLPs inserted inside each transformer block: project down to a small dimension, apply a nonlinearity, project back up, and add to the residual stream. They were the original PEFT method, at roughly half a percent to three percent of parameters, and they work well. The reason LoRA replaced them is structural: an adapter is a genuinely new sublayer that has to run at inference, so it adds latency permanently and it's sequential, meaning it can't be parallelized away.

**19. LoRA vs adapter — why is LoRA mergeable?**
LoRA's update $B A$ can be added to $W_0$ to form a new dense weight matrix. No extra inference computation. Adapters add a sublayer with its own matmul; mandatory inference latency overhead.

> **Saying it out loud.** LoRA's update is a plain additive change to an existing weight matrix, so you can compute $W_0$ plus $BA$ once and ship a normal dense model with literally zero inference overhead. An adapter is a new sublayer with its own matmul and nonlinearity, so it can never be folded in and you pay for it on every forward pass, typically a ten to thirty percent latency hit. In a serving context that difference decides the argument. The tradeoff is that once merged you lose the ability to swap adapters per request.

**20. What's prefix tuning?**
Li & Liang 2021. Prepend trainable "virtual tokens" (vectors) to each layer's $K, V$ cache. Model attends to them like real tokens. Trainable: per-layer prefix matrices.

> **Saying it out loud.** Prefix tuning prepends trainable vectors to the key and value caches at every layer, so the model attends to them as if they were tokens that were really there, except they're free parameters rather than embeddings of actual words. The weights never change at all. It's an elegant idea and it works, but it eats part of your context window and the optimization is notoriously finicky, often needing a reparameterization through an MLP to train stably. That instability is a big part of why LoRA won.

**21. What's prompt tuning?**
Lester et al. 2021. Simpler than prefix tuning: prepend trainable embeddings only at the input layer. Very few parameters. Works well at large model scales.

> **Saying it out loud.** Prompt tuning is the stripped-down version: trainable vectors only at the input embedding layer, nothing per-layer. That's an extremely small number of parameters, sometimes a few thousand. The striking result from the paper is that it's weak on small models but catches up with full fine-tuning as the model gets past ten billion parameters or so. So its viability is a function of scale, which is a nice thing to be able to say.

**22. What's IA³?**
Liu et al. 2022. Infused Adapter by Inhibiting and Amplifying inner activations. Multiplicative scaling on $K, V$, FFN intermediate. Tiny parameter count. Sometimes competitive with LoRA.

> **Saying it out loud.** IA-three learns a single multiplicative scaling vector for the keys, the values, and the feedforward intermediate activations. That's it, just element-wise rescaling, so it's even smaller than LoRA by an order of magnitude. It was competitive with LoRA on the few-shot benchmarks in the T-Few paper. Being multiplicative, it can be merged into the surrounding weights too, so it shares LoRA's zero-overhead inference property.

**23. What's DoRA?**
Liu et al. 2024. Decompose $W = m \cdot (V / \|V\|)$ where $V = W_0 + B A$. Train magnitude $m$ and direction separately. Beats LoRA at low ranks.

> **Saying it out loud.** DoRA splits the weight into a magnitude and a direction, keeps a learnable scalar magnitude per column, and applies the LoRA update to the direction only. The motivation came from an analysis showing full fine-tuning and LoRA change magnitude and direction in noticeably different patterns, so LoRA was constrained in a way that didn't match what fine-tuning actually does. Separating them closes much of that gap, and the gain is biggest at low ranks, where DoRA at rank four can match LoRA at rank eight. The cost is a bit more compute per step and slightly messier merging.

**24. What's GaLore?**
Zhao et al. 2024. Project the gradient into a low-rank space during optimization. Same memory savings as LoRA, but tracks the same trajectory as full FT. Reportedly closer to full FT quality than LoRA.

> **Saying it out loud.** GaLore keeps the low-rank idea but moves it from the weights to the gradients: you train all the parameters, but you project each gradient into a low-rank subspace before the optimizer sees it, so the optimizer state is what shrinks. That's a real difference, because LoRA restricts where the model can go while GaLore only compresses how you store the trip. It gets you comparable memory savings and can be used for pretraining, which LoRA fundamentally cannot. The catch is that you have to periodically recompute the projection subspace, which costs an SVD.

---

## E. Engineering

**25. How do you serve multiple LoRAs efficiently?**
Three approaches: (a) Merge each LoRA into separate dense models — zero overhead but storage cost per task. (b) Multi-LoRA inference (S-LoRA, Punica) — share base, batch LoRA computations. (c) Hot-swapping — load/unload adapters per request.

> **Saying it out loud.** Three options depending on how many adapters you have. Merge each one into its own dense model, which gives zero inference overhead but a full model's worth of storage per task. Multi-LoRA serving, which S-LoRA and Punica do, keeping one shared base in memory and batching requests with different adapters using custom kernels. Or hot-swapping adapters per request, which is simple but adds load latency. Once you're past a handful of tasks, multi-LoRA is the answer, because it's the only one where memory doesn't scale with the number of tasks.

**26. What's LoRA merging?**
Compute $W_{\text{new}} = W_0 + B A$ once and serve as a regular dense model. No inference overhead. Cost: separate merged model per task.

> **Saying it out loud.** Merging just means computing $W_0$ plus the scaled $BA$ once, offline, and saving the result as an ordinary dense model. Nothing at inference time knows LoRA was ever involved, so there's zero latency cost and no special serving code. The tradeoff is that you now store and load a complete model per task instead of a few tens of megabytes, and you've lost the ability to serve several tasks off one base. Merge when you have one task and lots of traffic; don't when you have many tasks.

**27. Multi-LoRA challenges?**
Memory (many adapters add up), batched throughput (specialized kernels needed), routing (which LoRA per request). S-LoRA / Punica provide production-ready solutions.

> **Saying it out loud.** Three challenges. Memory, since adapters are small individually but a thousand of them isn't. Throughput, because a batch containing requests for different adapters means different weight matrices per row, which naive kernels handle terribly, so you need the grouped GEMMs that Punica and S-LoRA implement. And routing, deciding which adapter a given request needs and keeping the hot ones resident. The core insight that makes it work is that the base model's compute is shared across the whole batch and only the tiny LoRA part differs.

**28. Can you compose multiple LoRAs?**
Yes — sum their $\Delta W$ contributions: $W = W_0 + \Delta W_1 + \Delta W_2$. Sometimes useful for multi-task. Quality varies; doesn't always combine cleanly because the LoRAs were trained for different tasks.

> **Saying it out loud.** You can just add the deltas together, since they're all additive updates to the same base. It works reasonably when the tasks are unrelated, like a style adapter plus a domain adapter, and badly when they compete, because nothing constrained them to be orthogonal, so their subspaces can overlap and interfere. Weighted merging with tuned coefficients helps, and there are methods like TIES and DARE that resolve sign conflicts before merging. The honest answer is that it sometimes works and you have to evaluate it, not assume it.

**29. What about LoRA dropout?**
Apply dropout on the $B A$ output. Standard regularization in HuggingFace PEFT (`lora_dropout=0.1` typical).

> **Saying it out loud.** LoRA dropout applies dropout to the output of the $BA$ branch, so the adapter's contribution gets randomly zeroed during training while the base path stays intact. It's ordinary regularization, aimed at the overfitting you get when fine-tuning on a small dataset. Point one is the usual default in HuggingFace PEFT. Bump it toward point two or three if you're training on a few thousand examples and seeing validation loss turn upward early.

---

## F. When and where

**30. When does LoRA underperform full fine-tuning?**
When the task requires substantial weight updates not captured by low-rank structure. Heavy domain shift, very specialized tasks. Empirically: LoRA matches or comes within 1–2 points of full FT on most tasks.

> **Saying it out loud.** LoRA falls behind when the task genuinely requires changes that aren't low rank, which means big domain shifts, new languages, or new capabilities rather than new styles. The intuition is that LoRA is good at re-weighting what the model already knows and bad at teaching it something it doesn't. On typical instruction-tuning and task adaptation it lands within a point or two of full fine-tuning. Adding rank helps a bit but doesn't fully close the gap, which suggests the constraint is structural, not just capacity.

**31. When is full fine-tuning still preferred?**
When you can afford the compute and need maximum quality. When LoRA's quality gap matters for the application. When you want to deploy without LoRA-merging infrastructure.

> **Saying it out loud.** Full fine-tuning still wins when you can afford it and the last point or two of quality matters, when the domain shift is large enough that low-rank updates can't express it, or when you'd rather not build LoRA serving infrastructure at all. It's also the right call if you're going to serve exactly one specialized model at very high volume, since you were going to merge anyway. The counterargument to keep in mind is that full fine-tuning is far more prone to catastrophic forgetting, so you may pay for that quality in lost general capability.

**32. When is QLoRA the right choice?**
Default for fine-tuning models > ~13B on consumer GPUs. Default for 70B+ on a single A100/H100. For most production fine-tuning, QLoRA is the workhorse.

> **Saying it out loud.** QLoRA is the default whenever the model doesn't comfortably fit in your GPUs at fp16. Practically that means anything past about thirteen billion parameters on consumer hardware, or seventy billion and up on a single A100 or H100. For most production fine-tuning today it's the workhorse. The cost you accept is slower steps from on-the-fly dequantization, so if the model fits without quantizing, use plain LoRA and get your throughput back.

**33. When is LoRA wrong?**
Pretraining (no base to LoRA-ify). Tasks requiring deep weight surgery (e.g., teaching new languages from scratch). Cases where you need very fast iteration on small differences (just train fully).

> **Saying it out loud.** LoRA is wrong for pretraining, because there's no pretrained base to add a low-rank correction to and you'd just be training a rank-constrained model badly. It's wrong when you need deep changes, like teaching a model a language it barely saw. And it's overkill for a small model that fits comfortably in memory, where full fine-tuning is simpler and better. The general rule: LoRA adapts, it doesn't teach from scratch.

---

## G. Subtleties

**34. Why doesn't $r = 1$ always work?**
A rank-1 matrix has very limited expressive capacity. While intrinsic dimension is low, it's typically larger than 1. $r = 4$–$8$ is usually the practical floor.

> **Saying it out loud.** Because rank one gives you a single outer product, one direction of change per weight matrix, which is far too little to express the update a real task needs. Intrinsic dimension is low, but low means dozens, not one. In practice four to eight is the floor where quality stops falling off a cliff. It's also a nice illustration that low rank is a soft prior about structure, not a claim that one direction suffices.

**35. What's the relationship between LoRA and matrix factorization?**
LoRA's $\Delta W = B A$ is a rank-$r$ factorization. Mathematically: SVD truncation at $r$ would give the optimal rank-$r$ approximation, but LoRA learns the factorization end-to-end during training rather than computing SVD.

> **Saying it out loud.** LoRA's $\Delta W = BA$ is exactly a rank-$r$ factorization, so it's the same object matrix factorization studies. The difference is how you find it: SVD would give you the provably optimal rank-$r$ approximation of a known matrix, but you don't know the ideal $\Delta W$ ahead of time, so there's nothing to decompose. LoRA learns the two factors by gradient descent against the task loss instead. So Eckart-Young tells you low-rank approximation can be very good; it doesn't tell you gradient descent will find the good one.

---

## Quick fire

**36.** *LoRA paper?* Hu et al. 2021.
**37.** *QLoRA paper?* Dettmers et al. 2023.
**38.** *DoRA paper?* Liu et al. 2024.
**39.** *Default $\alpha$?* $2r$.
**40.** *NF4 stands for?* NormalFloat 4-bit.

---

## Self-grading

If you can't answer 1-15, you don't know LoRA. If you can't answer 16-30, you'll struggle on PEFT-focused interviews. If you can't answer 31-40, frontier-lab interviews on efficient fine-tuning will go past you.

Aim for 25+/40 cold.
