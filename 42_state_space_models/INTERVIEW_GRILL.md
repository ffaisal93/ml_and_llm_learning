# State Space Models — Interview Grill

> 30 questions on SSMs (S4, Mamba, hybrids). Drill until you can answer 22+ cold.

---

## A. Foundations

**1. What is an SSM?**
**It's a linear recurrence** — state evolves, input adds, output is a readout. $h_t = A h_{t-1} + B x_t$, $y_t = C h_t + D x_t$. Same equations as classical state-space ODEs in control theory. The whole appeal: linear → can be computed both as a recurrence (fast inference) *and* a convolution / parallel scan (fast training).

**2. Why are SSMs interesting for LLMs?**
$O(N)$ sequence complexity (vs attention's $O(N^2)$). Constant memory at decode (vs growing KV cache). Empirically competitive quality (Mamba) at long contexts.

**3. What's the recurrent vs convolutional view?**
Recurrent: compute $h_t$ sequentially. Convolutional: unroll to $y = K * x$ where $K_k = C A^k B$ is the kernel. Equivalent for linear time-invariant SSM. SSMs train via convolution (parallel), generate via recurrence (constant memory).

**4. What's the discretization step?**
Continuous SSM $dh/dt = Ah + Bx,\ y = Ch$ discretized by zero-order hold to $h_t = A_d h_{t-1} + B_d x_t$, where $A_d = \exp(A \Delta t)$ and $B_d$ is derived. $\Delta t$ is a step size, often learned.

---

## B. HiPPO and S4

**5. What is HiPPO?**
Gu et al. 2020. Principled initialization for $A$ such that the hidden state is a polynomial approximation of input history. Provides theoretical long-range memory at initialization.

**6. Why does HiPPO matter for ML?**
Random $A$ doesn't naturally remember long history. HiPPO-initialized SSMs do, giving them a meaningful inductive bias for long-range dependencies. Empirically: HiPPO init substantially improves training.

**7. What's S4?**
Gu, Goel, Re 2022. Practical structured SSM. Uses Diagonal-Plus-Low-Rank parameterization of $A$ ($\Lambda + p q^\top$) so that $A^k$ can be computed efficiently, enabling $O(N \log N)$ convolution. First SSM to match transformers on Long Range Arena.

**8. Why DPLR?**
Computing $A^k$ for general $A$ is expensive. With $A = \text{diagonal} + \text{rank-1}$, the computation reduces to structured Cauchy-style operations. Critical for tractable convolution kernels.

---

## C. Mamba

**9. What's the central idea of Mamba?**
**Selectivity.** Make $B, C, \Delta t$ input-dependent (instead of fixed across positions). Each token can choose how much to remember vs forget. Closes the expressiveness gap with attention.

**10. Walk me through Mamba's parameterization.**
**One-liner**: "B, C, and the step size all become functions of the input — so each token decides how much to remember." Mechanics: $B(x), C(x), \Delta t(x) = \mathrm{softplus}(\mathrm{Linear}(x))$ are linear projections. $A$ is diagonal real-valued (S4D-Real init); its discretization via $\Delta t$ becomes input-dependent. State update: $h_t = \bar A(\Delta t_t) h_{t-1} + \bar B_t x_t$.

**11. Why can't Mamba use the convolutional view?**
The kernel $K_k = C A^k B$ depends on input via $\Delta t, B, C$. So there's no single kernel — different per token. Cannot precompute and convolve.

**12. How does Mamba parallelize training without the convolutional view?**
Parallel scan (Blelloch-style). The associative operation $(a_1, b_1) \oplus (a_2, b_2) = (a_2 a_1,\ a_2 b_1 + b_2)$ lets the recurrence be computed in $O(\log N)$ parallel depth. Mamba's CUDA kernel implements this efficiently.

**13. What's selectivity intuitively?**
Some tokens carry information worth remembering (large $\Delta t$ accumulates state); others are noise (small $\Delta t$ fades quickly). The model learns per-token "memory decisions." Without selectivity, all tokens contribute equally — too rigid.

**14. Mamba vs transformer at decode?**
Mamba: $O(d)$ per token, constant memory in $d$. Transformer: $O(\text{seq-len} \cdot d)$ per token, KV cache grows with sequence. For long contexts, Mamba's memory advantage is huge.

**15. Mamba vs transformer at training?**
Mamba: $O(N \cdot d)$ via parallel scan. Transformer: $O(N^2 \cdot d)$ via attention. Mamba's compute scales linearly; transformer's quadratically.

---

## D. Comparing to other models

**16. Mamba vs LSTM?**
Both are RNNs in some sense. LSTM: nonlinear gates, parallel-unfriendly, vanishing gradients with depth. Mamba: linear recurrence, parallel scan, stable gradients via structured $A$ and HiPPO init. Mamba is what LSTMs always wanted to be.

**17. Mamba vs linear attention?**
Both are $O(N)$. Linear attention: constant $K, V$ projections. Mamba: input-dependent $B, C, \Delta t$ — more expressive. Empirically, Mamba beats linear attention on language tasks.

**18. Mamba vs vanilla RNN?**
Vanilla RNN: random init, unstable, can't scale. Mamba: HiPPO-initialized, structured $A$, stable, scales. Different in practice despite similar mathematical form.

**19. Why hasn't Mamba replaced transformers?**
Weaker in-context learning / copy-recall. Less mature ecosystem (FlashAttention, vLLM are transformer-specific). Scaling laws unclear at frontier scale (~100B+). Hybrid models seem to be the practical compromise.

---

## E. Hybrid models

**20. What's a hybrid SSM-transformer?**
Mix attention layers and SSM layers. Attention layers handle copy/recall; SSM layers handle long-range mixing cheaply. Examples: Jamba, Zamba, Bamba, Hymba.

**21. What's Jamba?**
AI21 2024. 7-to-1 SSM-to-attention ratio. 256K+ context. Mamba blocks for cheap long-range; attention blocks for in-context behaviors; MoE on top. First production hybrid.

**22. Why might hybrids beat pure SSM or pure attention?**
Pure SSM: cheap but weak ICL. Pure attention: strong ICL but expensive at long context. Hybrid: cheap at long context (mostly SSM) with attention layers preserving copy/recall.

**23. Open question: Does hybrid beat dense transformer at frontier scale?**
Empirical, debated. Jamba and similar models are competitive but no flagship 100B+ hybrid has clearly beaten a dense transformer of similar compute. Active research.

---

## F. Subtleties

**24. What's Mamba-2?**
Dao & Gu 2024. "Transformers are SSMs": shows attention and SSM are mathematically related. Mamba-2 simplifies the parameterization with this structural understanding. Slightly faster training.

**25. Why is Mamba's HBM bandwidth efficiency important?**
Mamba's CUDA kernel keeps the state in SRAM during the scan (similar to FlashAttention's tiling). This makes the operation memory-bandwidth-efficient on modern GPUs. Without this, the parallel-scan version would be slow.

**26. What's the in-context-learning gap for SSMs?**
Empirically, SSMs are weaker at copying tokens from earlier in the context (the "induction head" behavior). Transformers' attention naturally implements this; SSMs must approximate. Hybrid layers (one attention layer per few SSM) often suffice to close the gap.

**27. Can Mamba do beam search / batched generation efficiently?**
Yes — but the state per beam is $O(d)$, so memory scales with $d \times \text{beam-count}$ not $\text{seq-len} \times \text{beam-count}$. Better than attention's KV cache for batched generation at long context.

---

## G. Practical / implementation

**28. Mamba implementation gotchas?**
The CUDA kernel is non-trivial. Float precision matters (state can drift in fp16; bf16 or fp32 for state recommended). Variable sequence lengths need padding handling.

**29. Where does Mamba fail?**
Tasks heavily reliant on exact copy from earlier in context (some tool-use, table-lookup-style tasks). Tasks where attention-style cross-token interactions are critical. Modern hybrids fix most of these.

**30. Future of SSMs in LLMs?**
Open questions: pure SSMs at frontier scale? Hybrid as new norm? Better selectivity mechanisms? Possibly the answer is "transformers + a few SSM layers" or "SSMs + a few attention layers" — frontier labs are actively exploring.

---

## Quick fire

**31.** *S4 paper?* Gu, Goel, Re 2022.
**32.** *Mamba paper?* Gu & Dao 2023.
**33.** *HiPPO paper?* Gu et al. 2020.
**34.** *Mamba sequence complexity?* $O(N)$.
**35.** *Mamba decode memory?* Constant in seq length.

---

## Self-grading

If you can't answer 1-12, you don't know SSMs. If you can't answer 13-22, you'll struggle on architecture deep-dives. If you can't answer 23-35, frontier-lab interviews will go past you.

Aim for 22+/35 cold.
