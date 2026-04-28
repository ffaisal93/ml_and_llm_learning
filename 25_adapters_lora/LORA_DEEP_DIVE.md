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

---

## 2. LoRA: the math

### Setup

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

### Number of trainable parameters

- Full fine-tuning: $d \cdot k$.
- LoRA: $r \cdot (d + k)$.
- For $d = k = 4096$ and $r = 16$: $4096 \times 4096 = 16\text{M}$ (full) vs $16 \times (4096 + 4096) = 130\text{K}$ (LoRA). **~125x fewer parameters.**

### Initialization

Critical: $A$ is initialized with random small values (typically $\mathcal{N}(0, \sigma^2)$ or Kaiming); $B$ is initialized to **zero**. So at initialization, $\Delta W = B A = 0$. The model behaves like the base model at the start of training.

### The scaling factor α

The LoRA update is scaled:

$$
y = W_0 x + \frac{\alpha}{r} B A x
$$

$\alpha$ is a hyperparameter (often $\alpha = 2r$ or $\alpha = r$). The $\alpha/r$ scaling makes the LoRA effect strength independent of $r$, so you can change $r$ without re-tuning the learning rate. **Hugging Face PEFT default: $\alpha = 2r$.**

### Where to apply LoRA

You don't apply LoRA to every weight matrix. Common choices:

- **Attention Q, K, V projections** (most common).
- **Attention output projection (O).**
- **FFN layers** (less common, but better performance).

LLaMA-style: LoRA on Q, V projections. More aggressive: LoRA on all linear layers.

---

## 3. Why does low-rank work?

The intrinsic-dimension hypothesis: **fine-tuning updates lie in a low-dimensional subspace of the weight space.** Empirically, full fine-tuning on a downstream task often produces $\Delta W$ with low effective rank. Aghajanyan et al. (2020) showed that fine-tuning trajectories live on a low-dimensional manifold.

LoRA imposes this low-rank structure explicitly. As long as $r$ is large enough to capture the relevant subspace, LoRA matches full fine-tuning quality.

### How big should $r$ be?

- $r = 4$–$8$: minimal, often sufficient for simple tasks.
- $r = 16$–$32$: standard for most LLM fine-tuning.
- $r = 64$–$128$: for complex tasks or when LoRA underperforms full FT.

In practice: try $r = 16$; if quality is insufficient, increase to 32 or 64. **Empirically, very small $r$ (like 4) often works** — confirming the intrinsic-dimension hypothesis.

---

## 4. QLoRA: quantize + LoRA

QLoRA (Dettmers et al. 2023) combines:

1. **Quantize the base model to 4-bit** (NF4 format).
2. **LoRA on top** in fp16.

Result: fine-tune a 70B model on a single GPU (~80GB memory) instead of needing ~16 GPUs.

### Key innovations

**NF4 (NormalFloat 4-bit).** Information-theoretically optimal 4-bit format for normally-distributed weights (which neural network weights approximately are). Better dynamic-range matching than INT4.

**Double quantization.** Quantize the quantization constants too. Saves ~0.4 bits per parameter.

**Paged optimizer.** Store optimizer state in CPU RAM and page to GPU as needed. Avoids OOM on optimizer state.

### Why QLoRA works

The base model is quantized — but **frozen**. Quantization noise is fixed; LoRA learns to compensate (and to learn the new task). Forward pass: dequantize 4-bit weights to fp16 on the fly, add LoRA contribution, output. Backward pass: gradients flow only through LoRA weights (4-bit weights are frozen).

### Quality cost

QLoRA matches full fp16 fine-tuning quality on most tasks. On hardest tasks, slight degradation (1–2 percentage points). For most production use cases, the cost-benefit is overwhelmingly in QLoRA's favor.

---

## 5. Adapter modules (Houlsby, Pfeiffer)

The earlier PEFT method. Insert small MLPs ("adapters") into each transformer block:

$$
\operatorname{adapter}(x) = \operatorname{down\_project}(\operatorname{activation}(\operatorname{up\_project}(x)))
$$

$$
\text{output} = x + \operatorname{adapter}(x) \qquad \text{(residual)}
$$

Adapter dimension $r$ (e.g., 64) is much smaller than the model dim. ~0.5–3% of total parameters.

### Pros

- Conceptually simple.
- Modular: swap adapters for different tasks.
- Can be combined (Pfeiffer's adapter fusion).

### Cons

- **Extra inference latency** (extra matmul per block).
- LoRA is mergeable; adapters are not.

LoRA largely replaced adapter modules. Some research and specific use cases (multi-task, modular composition) still use adapters.

---

## 6. Prefix tuning and prompt tuning

### Prefix tuning (Li & Liang 2021)

Prepend trainable "virtual tokens" (vectors in embedding space) to every layer's key-value cache. The model attends to these virtual tokens like real tokens, but they're learned per-task.

Each layer:

$$
K = [K_{\text{prefix}}; K_{\text{input}}], \qquad V = [V_{\text{prefix}}; V_{\text{input}}]
$$

Trainable: $K_{\text{prefix}}, V_{\text{prefix}}$ per layer.

### Prompt tuning (Lester et al. 2021)

Simpler version: prepend trainable tokens at the **input** layer only. The rest of the model processes these tokens normally.

$$
\text{input\_embeddings} = [\text{prompt\_embeddings}; \text{word\_embeddings}(\text{input\_ids})]
$$

Trainable: prompt embeddings only.

### Trade-offs

- Very few parameters (often <0.1% of total).
- Empirically weaker than LoRA at small to medium model sizes.
- **Catches up with model scale**: at 100B+, prompt tuning matches full fine-tuning.

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

---

## 8. DoRA (Weight-Decomposed LoRA, Liu et al. 2024)

A recent LoRA variant that decomposes weight updates into magnitude and direction:

$$
W = m \cdot \frac{V}{\|V\|}, \qquad V = W_0 + B A
$$

Magnitude $m$ and direction $V / \|V\|$ are updated separately. Empirically beats LoRA at the same rank, especially at low ranks.

Computationally: slightly more expensive per step than LoRA. Status: emerging; some adoption.

---

## 9. Recent variants

### VeRA (Vector-based Random Adaptation, Kopiczko et al. 2024)

Use random fixed $A$ and $B$ matrices shared across layers; only train per-layer scalar/vector scaling. Even fewer parameters than LoRA. Quality is competitive on some tasks.

### GaLore (Gradient Low-Rank Projection, Zhao et al. 2024)

Project the gradient (not the weights) into a low-rank space during optimization. Same memory savings as LoRA but tracks the same trajectory as full fine-tuning. Reportedly closer to full FT quality than LoRA.

### Tied LoRA

Share $A$ or $B$ matrices across layers/positions. Further reduces parameter count.

---

## 10. Production: serving multi-LoRA

### The setup

You've fine-tuned base model $W_0$ on tasks A, B, C, producing LoRA adapters $(A_A, B_A), (A_B, B_B), (A_C, B_C)$. Now you want to serve them efficiently.

### Approaches

**Option 1: Merge the LoRA**. At deploy time, compute $W = W_0 + B A$ and serve as a regular dense model. **Zero inference overhead.** Cost: one merged model per task.

**Option 2: Multi-LoRA inference (S-LoRA, Punica)**. Keep $W_0$ shared; load multiple LoRA adapters; route requests to the right adapter. Specialized kernels for batched LoRA computation. Used in production systems serving many fine-tunes (e.g., personalized chat).

**Option 3: LoRAX-style hot-swapping.** Swap LoRA adapters on-the-fly per request. Used when you have many LoRAs but only a few active at once.

### Multi-LoRA challenges

- **Memory.** Many LoRA adapters add up. With $r = 16$ and the modern attention-Q/V LoRA, each is ~10MB; thousands of adapters fit in GPU memory.
- **Throughput.** Batched LoRA computation is non-trivial; specialized kernels needed.
- **Routing.** Decide per-request which LoRA to apply. Often based on user / task / API key.

This is now standard in serving systems for personalized LLMs.

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
5. Drill `INTERVIEW_GRILL.md`.

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
