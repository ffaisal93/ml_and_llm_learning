# LoRA & Parameter-Efficient Fine-Tuning: A Frontier-Lab Interview Deep Dive

> **Why this exists.** LoRA is the dominant fine-tuning method for LLMs in production. Interviewers probe: why does low-rank work, the role of $r$ and $\alpha$, QLoRA's trick, how to serve multi-LoRA, and recent variants (DoRA, VeRA). This document covers the math and the engineering.

---

## 1. The big picture

Full fine-tuning of a 70B model requires updating 70B parameters and storing optimizer state for all of them — many GB of GPU memory beyond just the weights. **PEFT (Parameter-Efficient Fine-Tuning)** trains a small fraction of parameters while keeping the base model frozen.

LoRA (Low-Rank Adaptation, Hu et al. 2021) is the dominant PEFT method. The other major families:

- **Adapter modules** (Houlsby et al. 2019): bottleneck MLPs inserted in each layer.
- **Prefix / prompt tuning**: trainable virtual tokens prepended to inputs.
- **IA³**: rescale activations multiplicatively.
- **DoRA, VeRA, GaLore**: recent LoRA variants.

LoRA dominates because it's simple, effective, and **mergeable** at inference time (no extra latency).

> **Saying it out loud.** The whole field exists because of memory arithmetic. Full fine-tuning a seventy-billion-parameter model means holding the weights, the gradients, and two Adam moment buffers, which is roughly four times the model, so half a terabyte of GPU memory. Freeze the base and train a small add-on and all of that extra state disappears. Among the PEFT families, LoRA won for one specific engineering reason: its update can be folded back into the original weights, so inference costs exactly the same as the base model.

---

## 2. LoRA: the math

### Setup

> **In plain language.** Here's the idea before the symbols. Fine-tuning normally means nudging every number in a giant weight matrix. LoRA says: don't touch that matrix, leave it frozen, and instead learn a small correction that you add on top. And make the correction cheap by forcing it to be the product of two skinny matrices, so instead of millions of numbers you learn a few thousand.

For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ (frozen), the fine-tuned weight is:

$$
W = W_0 + \Delta W
$$

LoRA constrains $\Delta W$ to be **low-rank**:

$$
\Delta W = B A, \qquad A \in \mathbb{R}^{r \times k},\ B \in \mathbb{R}^{d \times r},\ r \ll \min(d, k)
$$

Now the forward pass:

$$
y = W x = W_0 x + B A x \qquad \text{(add LoRA contribution)}
$$

> **Saying it out loud.** Fine-tuning normally means learning a full-size update to a weight matrix. LoRA says that update is low rank, so learn it as a product of two thin matrices instead: one that squeezes down to $r$ dimensions and one that expands back up. The forward pass runs the frozen original plus that low-rank branch, and only the two thin matrices get gradients. Because it's a plain addition, you can precompute the sum after training and serve an ordinary dense model, which is the property that made LoRA win.

### Number of trainable parameters

- Full fine-tuning: $d \cdot k$.
- LoRA: $r \cdot (d + k)$.
- For $d = k = 4096$ and $r = 16$: $4096 \times 4096 = 16\text{M}$ (full) vs $16 \times (4096 + 4096) = 130\text{K}$ (LoRA). **~125x fewer parameters.**

> **Saying it out loud.** The parameter count goes from $d$ times $k$ down to $r$ times $d$ plus $k$. On a four-thousand-ninety-six square matrix at rank sixteen that's sixteen million versus a hundred thirty thousand, about a hundred twenty-five fold. Across the model you typically train well under one percent of the parameters. The number that makes it matter operationally: a LoRA checkpoint is tens of megabytes, so you can keep thousands of task adapters around for the price of a single base model.

### Initialization

Critical: $A$ is initialized with random small values (typically $\mathcal{N}(0, \sigma^2)$ or Kaiming); $B$ is initialized to **zero**. So at initialization, $\Delta W = B A = 0$. The model behaves like the base model at the start of training.

> **Saying it out loud.** The initialization is asymmetric on purpose: $A$ random, $B$ exactly zero, so the product is zero and the model at step zero is bit-identical to the pretrained one. If both were random you'd be adding noise to carefully pretrained weights and destroying capability before the first gradient step. If instead you zeroed $A$, the gradient into $B$ would also be zero and nothing would ever move. So the specific arrangement matters, and the failure mode of getting it wrong is a loss spike at step one that may never recover.

### The scaling factor α

The LoRA update is scaled:

$$
y = W_0 x + \frac{\alpha}{r} B A x
$$

$\alpha$ is a hyperparameter (often $\alpha = 2r$ or $\alpha = r$). The $\alpha/r$ scaling makes the LoRA effect strength independent of $r$, so you can change $r$ without re-tuning the learning rate. **Hugging Face PEFT default: $\alpha = 2r$.**

> **Saying it out loud.** Alpha over $r$ is a fixed scaling constant on the LoRA branch, not something you learn. Without it, increasing the rank would automatically increase the size of the update, since you're summing more rank-one pieces, and you'd have to retune the learning rate every time you changed rank. The scaling makes those two hyperparameters separable. HuggingFace defaults to alpha equal to twice the rank, and in practice people tune the ratio rather than the raw number.

### Where to apply LoRA

You don't apply LoRA to every weight matrix. Common choices:

- **Attention Q, K, V projections** (most common).
- **Attention output projection (O).**
- **FFN layers** (less common, but better performance).

LLaMA-style: LoRA on Q, V projections. More aggressive: LoRA on all linear layers.

> **Saying it out loud.** You don't LoRA every matrix, you pick. The classic recipe from the paper is the query and value projections only, which costs about one percent extra parameters. Adding the key and output projections and then the feedforward matrices costs more but performs better, and since the feedforward layers hold most of the parameters in a transformer, ignoring them leaves real quality on the table. The useful empirical rule: at a fixed parameter budget, a low rank spread across all linear layers beats a high rank on just $Q$ and $V$.

---

## 3. Why does low-rank work?

The intrinsic-dimension hypothesis: **fine-tuning updates lie in a low-dimensional subspace of the weight space.** Empirically, full fine-tuning on a downstream task often produces $\Delta W$ with low effective rank. Aghajanyan et al. (2020) showed that fine-tuning trajectories live on a low-dimensional manifold.

LoRA imposes this low-rank structure explicitly. As long as $r$ is large enough to capture the relevant subspace, LoRA matches full fine-tuning quality.

> **Saying it out loud.** The justification is empirical, not theoretical. Aghajanyan and colleagues showed you can constrain fine-tuning to a random low-dimensional subspace and still hit full fine-tuning quality, which means the task's intrinsic dimension is small. The interpretation is that pretraining already learned the features and fine-tuning is mostly reweighting them rather than learning new ones. LoRA just bakes that structure in from the start. And the honest caveat is that this is an observation about typical tasks, which is exactly why it breaks down on genuinely new capabilities like a language the model barely saw.

### How big should $r$ be?

- $r = 4$–$8$: minimal, often sufficient for simple tasks.
- $r = 16$–$32$: standard for most LLM fine-tuning.
- $r = 64$–$128$: for complex tasks or when LoRA underperforms full FT.

In practice: try $r = 16$; if quality is insufficient, increase to 32 or 64. **Empirically, very small $r$ (like 4) often works** — confirming the intrinsic-dimension hypothesis.

> **Saying it out loud.** Rank sixteen is the default you should reach for, with four to eight fine for simple style or format tasks and sixty-four to a hundred twenty-eight reserved for heavy domain shift. Quality saturates fast, so past about sixty-four you're usually burning memory for nothing. Start at sixteen and only go up if the eval says to. And remember rank interacts with dataset size, since a high rank on a thousand training examples just lets you overfit faster.

---

## 4. QLoRA: quantize + LoRA

QLoRA (Dettmers et al. 2023) combines:

1. **Quantize the base model to 4-bit** (NF4 format).
2. **LoRA on top** in fp16.

Result: fine-tune a 70B model on a single GPU (~80GB memory) instead of needing ~16 GPUs.

> **Saying it out loud.** QLoRA is two ideas stacked: squeeze the frozen base model down to four bits, then train normal fp16 LoRA adapters on top of it. Since the base isn't being trained, its precision only has to be good enough for the forward pass, and four bits turns out to be good enough. That takes a seventy-billion-parameter fine-tune from about sixteen GPUs to one. It's probably the single most practically important result in the fine-tuning literature, because it moved this work from labs to laptops.

### Key innovations

**NF4 (NormalFloat 4-bit).** Information-theoretically optimal 4-bit format for normally-distributed weights (which neural network weights approximately are). Better dynamic-range matching than INT4.

**Double quantization.** Quantize the quantization constants too. Saves ~0.4 bits per parameter.

**Paged optimizer.** Store optimizer state in CPU RAM and page to GPU as needed. Avoids OOM on optimizer state.

> **Saying it out loud.** Three innovations. NF4 places its sixteen quantization levels at the quantiles of a normal distribution instead of spacing them evenly, which matters because weights really are roughly Gaussian and uniform INT4 wastes most of its levels out in the empty tails. Double quantization quantizes the per-block scaling constants themselves, saving about another half bit per parameter, which sounds trivial until you multiply by seventy billion. And paged optimizers spill optimizer state to CPU memory so a transient spike doesn't kill a multi-day run.

### Why QLoRA works

The base model is quantized — but **frozen**. Quantization noise is fixed; LoRA learns to compensate (and to learn the new task). Forward pass: dequantize 4-bit weights to fp16 on the fly, add LoRA contribution, output. Backward pass: gradients flow only through LoRA weights (4-bit weights are frozen).

> **Saying it out loud.** Quantization normally hurts because errors compound through training, but here the base is frozen, so the quantization error is a fixed distortion that never changes. And the adapters are trained on top of the already-quantized model, so they learn to compensate for that distortion at the same time as they learn the task. Forward pass dequantizes block by block on the fly, backward pass only touches the adapters. The thing to name as the cost is speed, not quality: dequantizing on every matmul makes QLoRA noticeably slower per step than plain LoRA.

### Quality cost

QLoRA matches full fp16 fine-tuning quality on most tasks. On hardest tasks, slight degradation (1–2 percentage points). For most production use cases, the cost-benefit is overwhelmingly in QLoRA's favor.

> **Saying it out loud.** The quality cost is small, typically within a point or two of fp16 fine-tuning, and it only really shows on the hardest tasks. Given that you're cutting memory roughly fourfold, that trade is overwhelmingly worth it in almost every production setting. The rule I'd state: if the model fits at fp16, use plain LoRA and keep your training throughput; if it doesn't, use QLoRA and accept slower steps.

---

## 5. Adapter modules (Houlsby, Pfeiffer)

The earlier PEFT method. Insert small MLPs ("adapters") into each transformer block:

$$
\mathrm{adapter}(x) = \mathrm{up\_project}(\mathrm{activation}(\mathrm{down\_project}(x)))
$$

$$
\text{output} = x + \mathrm{adapter}(x) \qquad \text{(residual)}
$$

Adapter dimension $r$ (e.g., 64) is much smaller than the model dim. ~0.5–3% of total parameters.

> **Saying it out loud.** An adapter is a small bottleneck MLP dropped into each transformer block: project the hidden state down to a small dimension, apply a nonlinearity, project back up, and add to the residual. Sizing the bottleneck around sixty-four gives you roughly half a percent to three percent of the model's parameters. This was the original PEFT method and it genuinely works. Its structural problem is that it's a new sublayer, not a modification of an existing one, so it can never be folded away.

### Pros

- Conceptually simple.
- Modular: swap adapters for different tasks.
- Can be combined (Pfeiffer's adapter fusion).

> **Saying it out loud.** Adapters have real virtues: the idea is easy to explain, each adapter is a self-contained module you can swap per task, and there's a body of work like Pfeiffer's adapter fusion on composing several of them with a learned attention over modules. That composability is genuinely nicer than adding LoRA deltas together and hoping they don't interfere. So for multi-task and modular research they still have a place. It's serving economics, not quality, that pushed them aside.

### Cons

- **Extra inference latency** (extra matmul per block).
- LoRA is mergeable; adapters are not.

LoRA largely replaced adapter modules. Some research and specific use cases (multi-task, modular composition) still use adapters.

> **Saying it out loud.** The killer downside is latency. An adapter is an extra matmul plus nonlinearity in every block, and it's sequential, sitting on the critical path, so you pay it on every forward pass forever, typically ten to thirty percent. LoRA merges into the base weights and costs exactly zero. In a serving context that argument ends the discussion, which is why adapters survive in research and modular composition work rather than in production stacks.

---

## 6. Prefix tuning and prompt tuning

### Prefix tuning (Li & Liang 2021)

Prepend trainable "virtual tokens" (vectors in embedding space) to every layer's key-value cache. The model attends to these virtual tokens like real tokens, but they're learned per-task.

Each layer:

$$
K = [K_{\text{prefix}}; K_{\text{input}}], \qquad V = [V_{\text{prefix}}; V_{\text{input}}]
$$

Trainable: $K_{\text{prefix}}, V_{\text{prefix}}$ per layer.

> **Saying it out loud.** Prefix tuning prepends trainable vectors to the key and value caches at every layer, so the model attends to them as if they were tokens that were really in the input, except they're free parameters rather than embeddings of any actual word. No weight is ever changed. Two real costs: those virtual tokens consume context window at every layer, and the optimization is famously unstable, usually requiring a reparameterization through an auxiliary MLP just to train. That instability is a big part of why LoRA won.

### Prompt tuning (Lester et al. 2021)

Simpler version: prepend trainable tokens at the **input** layer only. The rest of the model processes these tokens normally.

$$
\text{input-embeddings} = [\text{prompt-embeddings}; \text{word-embeddings}(\text{input-ids})]
$$

Trainable: prompt embeddings only.

> **Saying it out loud.** Prompt tuning is the minimal version: trainable vectors at the input embedding layer only, nothing per layer, sometimes just a few thousand parameters total. It's essentially learning a soft prompt by gradient descent instead of writing one by hand. The headline result is about scale: it's clearly worse than fine-tuning on small models and catches up as you pass ten billion parameters or so. So its viability is a function of model size, which is a genuinely interesting thing to be able to say.

### Trade-offs

- Very few parameters (often <0.1% of total).
- Empirically weaker than LoRA at small to medium model sizes.
- **Catches up with model scale**: at 100B+, prompt tuning matches full fine-tuning.

> **Saying it out loud.** The tradeoff for prompt-based methods is parameters versus reliability. You're training under a tenth of a percent of the model, which is remarkable, but at small and medium scale they're clearly behind LoRA and the optimization is fiddly. The saving grace is that the gap closes as models grow, so at a hundred billion parameters prompt tuning matches full fine-tuning. Today they're niche, useful when you truly cannot afford to store anything per task and you're working with a very large model.

### Status

Niche. LoRA dominates. Prompt tuning is sometimes used for very lightweight task adaptation.

---

## 7. IA³ (Liu et al. 2022)

**Infused Adapter by Inhibiting and Amplifying Inner Activations.** Multiplicatively rescale activations:

$$
K \leftarrow K \cdot \ell_K, \qquad V \leftarrow V \cdot \ell_V
$$

$$
h_{\text{FFN}} \leftarrow h_{\text{FFN}} \cdot \ell_{\text{FF}}
$$

$\ell_K, \ell_V, \ell_{\text{FF}}$ are learned per-layer, per-vector scaling factors. Very few parameters (~0.01% of model).

Reportedly competitive with LoRA on some tasks; less popular in practice.

> **Saying it out loud.** IA-three learns one multiplicative scaling vector each for the keys, the values, and the feedforward intermediate activations, and nothing else. That's about a hundredth of a percent of the model, an order of magnitude below LoRA. It was competitive with LoRA on few-shot benchmarks in the T-Few paper, and being multiplicative it merges into the neighboring weights, so it shares LoRA's zero-overhead property. It's less popular mainly because tooling and community defaults settled on LoRA, not because it fails.

---

## 8. DoRA (Weight-Decomposed LoRA, Liu et al. 2024)

A recent LoRA variant that decomposes weight updates into magnitude and direction:

$$
W = m \cdot \frac{V}{\|V\|}, \qquad V = W_0 + B A
$$

Magnitude $m$ and direction $V / \|V\|$ are updated separately. Empirically beats LoRA at the same rank, especially at low ranks.

Computationally: slightly more expensive per step than LoRA. Status: emerging; some adoption.

> **Saying it out loud.** DoRA came from an observation: if you decompose weight changes into magnitude and direction, full fine-tuning and LoRA move them in noticeably different patterns, so LoRA was constrained in a way that didn't match what fine-tuning actually does. So DoRA learns a separate scalar magnitude per column and applies the low-rank update only to the direction. The payoff is biggest at low rank, where DoRA at four can match LoRA at eight. Costs are a bit more compute per step and slightly messier merging.

---

## 9. Recent variants

### VeRA (Vector-based Random Adaptation, Kopiczko et al. 2024)

Use random fixed $A$ and $B$ matrices shared across layers; only train per-layer scalar/vector scaling. Even fewer parameters than LoRA. Quality is competitive on some tasks.

> **Saying it out loud.** VeRA takes the compression one step further: freeze $A$ and $B$ as random matrices shared across all layers, and learn only small per-layer scaling vectors. It sounds like it shouldn't work, and the reason it does is the same reason random projections work generally, that a random subspace of decent size usually contains a good enough solution. Parameter counts drop by another order of magnitude below LoRA, which matters when you're storing per-user adapters. Quality is competitive on some tasks and not all, so it's a specialist tool.

### GaLore (Gradient Low-Rank Projection, Zhao et al. 2024)

Project the gradient (not the weights) into a low-rank space during optimization. Same memory savings as LoRA but tracks the same trajectory as full fine-tuning. Reportedly closer to full FT quality than LoRA.

> **Saying it out loud.** GaLore moves the low-rank constraint from the weights to the gradients. You still update every parameter, but each gradient gets projected into a low-rank subspace before the optimizer sees it, so it's the optimizer state that shrinks rather than the model's freedom to move. That's a real conceptual difference: LoRA restricts where the model can go, GaLore only compresses the bookkeeping. Consequently GaLore can be used for pretraining, which LoRA fundamentally cannot, and the cost is periodically recomputing the projection subspace with an SVD.

### Tied LoRA

Share $A$ or $B$ matrices across layers/positions. Further reduces parameter count.

> **Saying it out loud.** Tied LoRA shares the $A$ or $B$ matrices across layers so you're not paying for a fresh pair everywhere. It's the same instinct as weight tying in older language models: if the layers are learning related corrections, let them share. Parameter count drops further, expressiveness drops a bit too. It's worth reaching for when you're serving very large numbers of adapters and per-adapter storage is the binding constraint.

---

## 10. Production: serving multi-LoRA

### The setup

You've fine-tuned base model $W_0$ on tasks A, B, C, producing LoRA adapters $(A_A, B_A), (A_B, B_B), (A_C, B_C)$. Now you want to serve them efficiently.

### Approaches

**Option 1: Merge the LoRA**. At deploy time, compute $W = W_0 + B A$ and serve as a regular dense model. **Zero inference overhead.** Cost: one merged model per task.

**Option 2: Multi-LoRA inference (S-LoRA, Punica)**. Keep $W_0$ shared; load multiple LoRA adapters; route requests to the right adapter. Specialized kernels for batched LoRA computation. Used in production systems serving many fine-tunes (e.g., personalized chat).

**Option 3: LoRAX-style hot-swapping.** Swap LoRA adapters on-the-fly per request. Used when you have many LoRAs but only a few active at once.

> **Saying it out loud.** Three serving strategies with different economics. Merge the adapter into the base and deploy a normal dense model: zero overhead, but a whole model's worth of storage per task. Multi-LoRA serving, which is what S-LoRA and Punica do, keeps a single shared base in memory and applies the right small adapter per request using custom batched kernels. Or hot-swap adapters per request, which is simple but adds load latency. The deciding factor is task count: merge for one high-traffic task, multi-LoRA once you have dozens, because it's the only approach where memory doesn't scale with the number of tasks.

### Multi-LoRA challenges

- **Memory.** Many LoRA adapters add up. With $r = 16$ and the modern attention-Q/V LoRA, each is ~10MB; thousands of adapters fit in GPU memory.
- **Throughput.** Batched LoRA computation is non-trivial; specialized kernels needed.
- **Routing.** Decide per-request which LoRA to apply. Often based on user / task / API key.

This is now standard in serving systems for personalized LLMs.

> **Saying it out loud.** Three problems in multi-LoRA serving. Memory, which is the easy one, since a rank-sixteen adapter is around ten megabytes and thousands fit on a GPU. Throughput, which is the hard one, because a batch containing requests for different adapters means a different weight matrix per row, and naive kernels handle that terribly, so you need grouped or segmented GEMMs. And routing, deciding which adapter a request needs, usually by user or API key. The insight that makes the whole thing viable is that the expensive base-model compute is shared across the entire batch and only the tiny LoRA part differs.

---

## 11. When to use what

| Scenario | Recommendation |
|---|---|
| Fine-tune a 7B for a task | LoRA $r = 16$, full fp16 training |
| Fine-tune a 70B without 16x A100s | QLoRA |
| Quick prototyping, tiny task | LoRA $r = 4$–$8$ |
| Many tasks, one base model | LoRA + merging or multi-LoRA serving |
| Need maximum quality, full FT compute available | Full fine-tuning still wins by 1–2 points |
| Multi-task with shared structure | Adapter fusion or LoRA composition |
| Very limited memory and quality tolerance | Prompt tuning or IA³ |

> **Saying it out loud.** The decision tree is short. Seven billion parameters and enough GPU: LoRA at rank sixteen in fp16. Seventy billion without a cluster: QLoRA. Small quick experiment: rank four or eight. Many tasks on one base: multi-LoRA serving. Maximum quality with compute to burn: full fine-tuning still wins by a point or two. And genuinely tiny memory budget on a very large model: prompt tuning or IA-three. Most real projects land on QLoRA at rank sixteen, and that's a completely defensible default.

---

## 12. Common interview gotchas

| Gotcha | Strong answer |
|---|---|
| "Why does low-rank work?" | Fine-tuning trajectories empirically lie in a low-dimensional subspace (Aghajanyan et al.). LoRA imposes this structure explicitly. |
| "Why initialize B to zero?" | So at init, $B A = 0$ and the model is identical to the base model. The LoRA effect grows during training rather than perturbing the model from the start. |
| "What's the role of α/r scaling?" | Decouples LR sensitivity from $r$. With it, you can change $r$ without re-tuning the LR. Default $\alpha = 2r$. |
| "QLoRA — does quantization hurt quality?" | Slightly. NF4 quantization is information-theoretically optimal for Gaussian-distributed weights; LoRA learns to compensate for quantization noise during fine-tuning. Net quality is close to fp16 LoRA. |
| "LoRA vs adapter modules?" | LoRA is mergeable (no inference cost). Adapters add a sublayer per transformer block (extra matmul). LoRA dominates. |
| "When does LoRA underperform full fine-tuning?" | When the task requires updating substantial portions of the model that don't align with the low-rank assumption. Heavy domain shift, very specialized tasks. |
| "Multi-LoRA serving — how?" | Keep base shared; load adapters; route per-request. S-LoRA, Punica systems use specialized batched kernels. |
| "What's DoRA?" | Decompose updates into magnitude and direction. Slightly better than LoRA at low ranks. Recent. |

> **Saying it out loud.** If I compress the gotchas: low rank works because fine-tuning updates empirically have low intrinsic dimension. $B$ starts at zero so the model begins as an exact copy of the base. Alpha over $r$ exists so you can change rank without retuning the learning rate. QLoRA costs about a point of quality because the adapters learn to compensate for a fixed quantization distortion. And LoRA beats adapters on the one thing that matters in serving, which is that it merges away to zero inference cost.

---

## 13. The 10 most-asked LoRA interview questions

1. **What is LoRA?** Add $\Delta W = B A$ to frozen base weight; $B, A$ are low-rank matrices. ~100x fewer parameters than full fine-tuning.
2. **Why does it work?** Fine-tuning updates have low intrinsic dimension. LoRA imposes this.
3. **Initialization?** $A$ random, $B$ zero, so $\Delta W = 0$ at start.
4. **What's α?** Scaling factor $\alpha/r$ decouples LR sensitivity from rank choice.
5. **What's QLoRA?** Quantize base to NF4 (4-bit); train LoRA in fp16 on top. Massive memory savings.
6. **Why is LoRA mergeable?** $W_{\text{new}} = W_0 + B A$ can be computed once; no extra inference latency.
7. **LoRA vs adapter modules?** LoRA: mergeable, no extra latency. Adapters: extra sublayer per block.
8. **Where do you apply LoRA?** Attention Q, V (most common). Also K, O, FFN. Empirically: more matrices = better but more parameters.
9. **What's typical r?** 16–32 for most tasks. Smaller (4–8) for simple tasks; larger (64–128) for complex.
10. **Multi-LoRA serving?** Keep base shared; specialized kernels (S-LoRA, Punica) for batched LoRA inference.

---

## 14. Drill plan

1. Master the LoRA math: $\Delta W = B A$, parameter count $r(d + k)$, init $B = 0$.
2. Know $\alpha/r$ scaling and why it matters.
3. Know QLoRA's three innovations (NF4, double quantization, paged optimizer).
4. Compare LoRA to alternatives (adapters, prefix tuning, IA³).
5. Drill [`INTERVIEW_GRILL.md`](INTERVIEW_GRILL.md).

---

## 15. Further reading

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
- Aghajanyan et al., "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning" (2020).
- Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (Adapter, 2019).
- Li & Liang, "Prefix-Tuning" (2021).
- Lester et al., "The Power of Scale for Parameter-Efficient Prompt Tuning" (2021).
- Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning" (IA³, 2022).
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023).
- Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation" (2024).
- Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters" (2023).
- Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection" (2024).
