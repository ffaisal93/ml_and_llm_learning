# LoRA & PEFT — Interview Grill

> 35 questions on parameter-efficient fine-tuning. Drill until you can answer 25+ cold.

---

## A. Foundations

**1. What is PEFT?**
Parameter-Efficient Fine-Tuning. Train a small fraction of parameters while keeping the base model frozen. Saves memory (no optimizer state for frozen weights), enables fine-tuning huge models on modest hardware.

**2. Why not just full fine-tuning?**
Memory. A 70B model needs ~140GB just for weights at fp16; full fine-tuning needs ~3–4x that for gradients + optimizer state. PEFT fits in much less memory.

**3. What are the major PEFT families?**
LoRA (low-rank weight updates), adapter modules (bottleneck MLPs in each layer), prefix/prompt tuning (trainable virtual tokens), IA³ (multiplicative scaling). LoRA dominates.

---

## B. LoRA

**4. What is LoRA?**
Hu et al. 2021. For each weight $W_0$, add a low-rank update $\Delta W = B A$ where $A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}, r \ll \min(d, k)$. New forward pass: $y = W_0 x + B A x$. Train only $A, B$; freeze $W_0$.

**5. How many parameters does LoRA use?**
$r \cdot (d + k)$ per matrix, vs $d \cdot k$ for full FT. For $d = k = 4096, r = 16$: 130K vs 16M (~125x reduction).

**6. How is LoRA initialized?**
$A$: random small values (Kaiming or Gaussian). $B$: zero. So $\Delta W = B A = 0$ at init. The model behaves like the base model at start of training; LoRA effect grows during training.

**7. Why initialize $B$ to zero specifically?**
If both $A$ and $B$ were random, the initial $\Delta W$ would be a random perturbation of $W_0$ — destroying pretrained capabilities at step 0. Zero-init $B$ keeps the base model intact at start.

**8. What's the α scaling factor?**
$y = W_0 x + (\alpha/r) \cdot B A x$. The $\alpha/r$ scaling decouples LR sensitivity from rank choice. Default in HuggingFace PEFT: $\alpha = 2r$.

**9. Why α/r scaling specifically?**
With it, the "magnitude" of the LoRA update is approximately constant in $r$. You can change $r$ without re-tuning the learning rate.

**10. Where do you apply LoRA in a transformer?**
Most common: attention $Q$ and $V$ projections. More aggressive: $K, O$, and FFN. Empirically more matrices = better quality but more parameters. LLaMA-style: $Q, V$ (about 1% extra parameters).

**11. Typical $r$ value?**
16–32 for most tasks. Smaller (4–8) for simple tasks or low memory budget. Larger (64–128) for complex domain shifts.

**12. Why does low-rank work?**
Aghajanyan et al. 2020 showed empirically that fine-tuning trajectories lie on a low-dimensional manifold. LoRA imposes this structure explicitly.

---

## C. QLoRA

**13. What is QLoRA?**
Dettmers et al. 2023. Quantize the base model to 4-bit (NF4); train LoRA in fp16 on top. Forward pass: dequantize on-the-fly for matmul. Backward: gradients flow only through LoRA. Massive memory savings.

**14. Three innovations of QLoRA?**
(a) NF4 quantization — info-theoretically optimal 4-bit for Gaussian weights. (b) Double quantization — quantize the quantization constants. (c) Paged optimizer — store optimizer state on CPU, page to GPU as needed.

**15. What's NF4?**
NormalFloat 4-bit. Quantization buckets chosen to be info-theoretically optimal for normally-distributed weights (which neural network weights approximately are). Better than uniform INT4 because it allocates more buckets to common values.

**16. Why doesn't QLoRA hurt quality much?**
The base is frozen — quantization noise is fixed. LoRA fine-tunes "on top of" the quantized base, learning to compensate for quantization noise while learning the new task. Net quality close to fp16 LoRA.

**17. Memory savings of QLoRA?**
A 70B model: 140GB at fp16. QLoRA: ~35GB (4-bit weights) + small overhead for LoRA adapters + activations. Fits on a single 80GB A100.

---

## D. Other PEFT methods

**18. What are adapter modules?**
Houlsby et al. 2019. Insert small bottleneck MLPs in each transformer block: $\text{down-project} \to \text{activation} \to \text{up-project} + \text{residual}$. ~0.5–3% of total parameters. Replaced by LoRA in production.

**19. LoRA vs adapter — why is LoRA mergeable?**
LoRA's update $B A$ can be added to $W_0$ to form a new dense weight matrix. No extra inference computation. Adapters add a sublayer with its own matmul; mandatory inference latency overhead.

**20. What's prefix tuning?**
Li & Liang 2021. Prepend trainable "virtual tokens" (vectors) to each layer's $K, V$ cache. Model attends to them like real tokens. Trainable: per-layer prefix matrices.

**21. What's prompt tuning?**
Lester et al. 2021. Simpler than prefix tuning: prepend trainable embeddings only at the input layer. Very few parameters. Works well at large model scales.

**22. What's IA³?**
Liu et al. 2022. Infused Adapter by Inhibiting and Amplifying inner activations. Multiplicative scaling on $K, V$, FFN intermediate. Tiny parameter count. Sometimes competitive with LoRA.

**23. What's DoRA?**
Liu et al. 2024. Decompose $W = m \cdot (V / \|V\|)$ where $V = W_0 + B A$. Train magnitude $m$ and direction separately. Beats LoRA at low ranks.

**24. What's GaLore?**
Zhao et al. 2024. Project the gradient into a low-rank space during optimization. Same memory savings as LoRA, but tracks the same trajectory as full FT. Reportedly closer to full FT quality than LoRA.

---

## E. Engineering

**25. How do you serve multiple LoRAs efficiently?**
Three approaches: (a) Merge each LoRA into separate dense models — zero overhead but storage cost per task. (b) Multi-LoRA inference (S-LoRA, Punica) — share base, batch LoRA computations. (c) Hot-swapping — load/unload adapters per request.

**26. What's LoRA merging?**
Compute $W_{\text{new}} = W_0 + B A$ once and serve as a regular dense model. No inference overhead. Cost: separate merged model per task.

**27. Multi-LoRA challenges?**
Memory (many adapters add up), batched throughput (specialized kernels needed), routing (which LoRA per request). S-LoRA / Punica provide production-ready solutions.

**28. Can you compose multiple LoRAs?**
Yes — sum their $\Delta W$ contributions: $W = W_0 + \Delta W_1 + \Delta W_2$. Sometimes useful for multi-task. Quality varies; doesn't always combine cleanly because the LoRAs were trained for different tasks.

**29. What about LoRA dropout?**
Apply dropout on the $B A$ output. Standard regularization in HuggingFace PEFT (`lora_dropout=0.1` typical).

---

## F. When and where

**30. When does LoRA underperform full fine-tuning?**
When the task requires substantial weight updates not captured by low-rank structure. Heavy domain shift, very specialized tasks. Empirically: LoRA matches or comes within 1–2 points of full FT on most tasks.

**31. When is full fine-tuning still preferred?**
When you can afford the compute and need maximum quality. When LoRA's quality gap matters for the application. When you want to deploy without LoRA-merging infrastructure.

**32. When is QLoRA the right choice?**
Default for fine-tuning models > ~13B on consumer GPUs. Default for 70B+ on a single A100/H100. For most production fine-tuning, QLoRA is the workhorse.

**33. When is LoRA wrong?**
Pretraining (no base to LoRA-ify). Tasks requiring deep weight surgery (e.g., teaching new languages from scratch). Cases where you need very fast iteration on small differences (just train fully).

---

## G. Subtleties

**34. Why doesn't $r = 1$ always work?**
A rank-1 matrix has very limited expressive capacity. While intrinsic dimension is low, it's typically larger than 1. $r = 4$–$8$ is usually the practical floor.

**35. What's the relationship between LoRA and matrix factorization?**
LoRA's $\Delta W = B A$ is a rank-$r$ factorization. Mathematically: SVD truncation at $r$ would give the optimal rank-$r$ approximation, but LoRA learns the factorization end-to-end during training rather than computing SVD.

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
