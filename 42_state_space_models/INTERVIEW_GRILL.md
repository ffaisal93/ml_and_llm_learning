# State Space Models — Interview Grill

> 30 questions on SSMs (S4, Mamba, hybrids). Drill until you can answer 22+ cold.

---

## A. Foundations

**1. What is an SSM?**
**It's a linear recurrence** — state evolves, input adds, output is a readout. $h_t = A h_{t-1} + B x_t$, $y_t = C h_t + D x_t$. Same equations as classical state-space ODEs in control theory. The whole appeal: linear → can be computed both as a recurrence (fast inference) *and* a convolution / parallel scan (fast training).

> **Saying it out loud.** An SSM is just a running summary that you keep updating. At every step you shrink what you already remember a little, add in whatever the new token brings, and then read an answer out of that summary. It's the same math control engineers use to track a moving object — position and velocity in, updated estimate out. The reason people care is that the update is linear, so you can run it token by token when you're generating, or unroll it and do the whole sequence in parallel when you're training. You get the cheap inference of an RNN without the slow training of an RNN.

**2. Why are SSMs interesting for LLMs?**
$O(N)$ sequence complexity (vs attention's $O(N^2)$). Constant memory at decode (vs growing KV cache). Empirically competitive quality (Mamba) at long contexts.

> **Saying it out loud.** The short version is cost. Attention has to look at every previous token for every new token, so doubling your context quadruples the work, and your KV cache keeps growing. An SSM carries one fixed-size state, so it's linear in sequence length and the memory at decode time is constant no matter how long the context gets. That's the whole pitch: at 100K tokens the transformer is dragging around a huge cache and Mamba is dragging around the same little state it had at token ten.

**3. What's the recurrent vs convolutional view?**
Recurrent: compute $h_t$ sequentially. Convolutional: unroll to $y = K * x$ where $K_k = C A^k B$ is the kernel. Equivalent for linear time-invariant SSM. SSMs train via convolution (parallel), generate via recurrence (constant memory).

> **Saying it out loud.** These are two ways of computing the exact same thing. The recurrent view is step by step: update the state, read out, repeat — great when you're generating one token at a time. The convolutional view says, since the recurrence is linear, I can just expand it out and the output is the input convolved with a fixed kernel, which the GPU can do all at once. So you train in convolution mode for speed and generate in recurrence mode for memory. The catch is that this equivalence only holds if the system is time-invariant — same $A$, $B$, $C$ at every step.

**4. What's the discretization step?**
Continuous SSM $dh/dt = Ah + Bx,\ y = Ch$ discretized by zero-order hold to $h_t = A_d h_{t-1} + B_d x_t$, where $A_d = \exp(A \Delta t)$ and $B_d$ is derived. $\Delta t$ is a step size, often learned.

> **Saying it out loud.** The original state-space equations are written in continuous time, like a physical system evolving smoothly, but tokens arrive in discrete steps. Discretization is how you convert one to the other: you pick a step size $\Delta t$ and ask what the continuous system would do over that interval. Zero-order hold assumes the input is held constant across the step, which gives you $A_d = \exp(A\Delta t)$. The practical thing to remember is that $\Delta t$ is usually learned — a big $\Delta t$ means the state moves a lot per token, a small one means the state barely changes, which is exactly the knob Mamba later makes input-dependent.

---

## B. HiPPO and S4

**5. What is HiPPO?**
Gu et al. 2020. Principled initialization for $A$ such that the hidden state is a polynomial approximation of input history. Provides theoretical long-range memory at initialization.

> **Saying it out loud.** HiPPO is a recipe for how to initialize the $A$ matrix so the state actually remembers things. The idea is that instead of storing raw history, the state stores the coefficients of a polynomial that best fits the history you've seen so far — like keeping a compressed sketch of the curve rather than every point. Gu and colleagues worked out the specific $A$ that makes this happen and proved it's optimal under a given weighting. It's initialization, not architecture, but it's the difference between a model that remembers thousands of steps and one that forgets after fifty.

**6. Why does HiPPO matter for ML?**
Random $A$ doesn't naturally remember long history. HiPPO-initialized SSMs do, giving them a meaningful inductive bias for long-range dependencies. Empirically: HiPPO init substantially improves training.

> **Saying it out loud.** Because a randomly initialized linear recurrence either explodes or forgets almost immediately. If the eigenvalues of $A$ are a bit above one you blow up, a bit below one and your memory decays exponentially — and random init basically never lands on the useful regime. HiPPO puts you in that regime by construction, so long-range memory is there from step zero instead of something the optimizer has to discover. In the S4 papers, swapping HiPPO init for random init is worth tens of points on Long Range Arena — it's not a small tweak.

**7. What's S4?**
Gu, Goel, Re 2022. Practical structured SSM. Uses Diagonal-Plus-Low-Rank parameterization of $A$ ($\Lambda + p q^\top$) so that $A^k$ can be computed efficiently, enabling $O(N \log N)$ convolution. First SSM to match transformers on Long Range Arena.

> **Saying it out loud.** S4 was the first structured state space model that actually worked at scale, from Gu, Goel and Ré in 2022. The problem it solved is computational: to get the convolution kernel you need powers of $A$, and for a general dense matrix that's hopeless. S4 restricts $A$ to a diagonal-plus-low-rank form, which turns those powers into a structured computation you can do in $O(N\log N)$. That's what let it be the first SSM to match transformers on Long Range Arena — and it's still a fairly fiddly parameterization, which is part of why Mamba later simplified to plain diagonal.

**8. Why DPLR?**
Computing $A^k$ for general $A$ is expensive. With $A = \text{diagonal} + \text{rank-1}$, the computation reduces to structured Cauchy-style operations. Critical for tractable convolution kernels.

> **Saying it out loud.** It's purely about making the kernel computable. Raising an arbitrary $N \times N$ matrix to the power $k$ for every $k$ up to sequence length is far too expensive to do inside a training loop. But if $A$ is a diagonal matrix plus a rank-one correction, the whole thing collapses into Cauchy-kernel-style operations you can evaluate fast. The tradeoff is complexity of implementation — DPLR is genuinely painful to write, and later work found that a plain diagonal $A$ with the right initialization gets you most of the quality for a fraction of the pain.

---

## C. Mamba

**9. What's the central idea of Mamba?**
**Selectivity.** Make $B, C, \Delta t$ input-dependent (instead of fixed across positions). Each token can choose how much to remember vs forget. Closes the expressiveness gap with attention.

> **Saying it out loud.** Selectivity — letting the model decide, per token, what's worth remembering. In S4 the dynamics were the same at every position, so a filler word and a key fact got processed identically, which is a weirdly rigid thing for a language model to do. Mamba makes $B$, $C$ and the step size $\Delta t$ functions of the current input, so each token effectively sets its own remember-versus-forget dial. That's what closed most of the quality gap with attention — and the cost is that you lose the convolutional view, since there's no longer one fixed kernel.

**10. Walk me through Mamba's parameterization.**
**One-liner**: "B, C, and the step size all become functions of the input — so each token decides how much to remember." Mechanics: $B(x), C(x), \Delta t(x) = \mathrm{softplus}(\mathrm{Linear}(x))$ are linear projections. $A$ is diagonal real-valued (S4D-Real init); its discretization via $\Delta t$ becomes input-dependent. State update: $h_t = \bar A(\Delta t_t) h_{t-1} + \bar B_t x_t$.

> **Saying it out loud.** The one-line version I'd give is: $B$, $C$ and the step size all become functions of the input, so each token decides how much to remember. Concretely they're just linear projections of the token's hidden vector, with a softplus on $\Delta t$ to keep it positive. The $A$ matrix stays fixed and diagonal — real-valued S4D init — but because it gets discretized through an input-dependent $\Delta t$, the effective decay is input-dependent too. So one shared, cheap $A$ plus three per-token projections buys you all the selectivity.

**11. Why can't Mamba use the convolutional view?**
The kernel $K_k = C A^k B$ depends on input via $\Delta t, B, C$. So there's no single kernel — different per token. Cannot precompute and convolve.

> **Saying it out loud.** Because the convolution trick needs one kernel for the whole sequence, and Mamba doesn't have one. The kernel is built from $C$, powers of $A$ and $B$ — and once those depend on the input, every position has a different kernel. You can't precompute something that changes per token, so the FFT-based convolution is off the table. That's exactly why Mamba needed a hardware-aware parallel scan instead; without that kernel work, selectivity would have made it too slow to be interesting.

**12. How does Mamba parallelize training without the convolutional view?**
Parallel scan (Blelloch-style). The associative operation $(a_1, b_1) \oplus (a_2, b_2) = (a_2 a_1,\ a_2 b_1 + b_2)$ lets the recurrence be computed in $O(\log N)$ parallel depth. Mamba's CUDA kernel implements this efficiently.

> **Saying it out loud.** It uses a parallel scan, the same trick you'd use for a fast prefix sum. The key observation is that composing two consecutive linear updates gives you another linear update of the same shape, and that composition is associative. Associative means you can build a tree instead of a chain, so a sequence of length $N$ takes $\log N$ parallel steps rather than $N$ sequential ones. Mamba ships a custom CUDA kernel that does this while keeping the state in fast SRAM — the algorithm alone isn't enough, the memory-traffic engineering is half the win.

**13. What's selectivity intuitively?**
Some tokens carry information worth remembering (large $\Delta t$ accumulates state); others are noise (small $\Delta t$ fades quickly). The model learns per-token "memory decisions." Without selectivity, all tokens contribute equally — too rigid.

> **Saying it out loud.** Think of it as a volume knob on memory that every token gets to turn. A token that matters — a name, a number, a topic shift — pushes $\Delta t$ up, which means it writes hard into the state and the state persists. A filler token pushes $\Delta t$ down and basically slides past without disturbing anything. Without selectivity every token gets equal weight in the summary, which is fine for signal-processing benchmarks but bad for language, where most tokens are noise. The concrete failure it fixes is selective copying and induction-style tasks that fixed-dynamics S4 simply cannot do.

**14. Mamba vs transformer at decode?**
Mamba: $O(d)$ per token, constant memory in $d$. Transformer: $O(\text{seq-len} \cdot d)$ per token, KV cache grows with sequence. For long contexts, Mamba's memory advantage is huge.

> **Saying it out loud.** At generation time Mamba wins clearly. A transformer has to attend over every cached key and value, so per-token cost grows with how much context you've already produced, and the KV cache grows right along with it. Mamba just updates one fixed-size state, so it's constant work and constant memory per token regardless of whether you're at token 100 or token 100,000. Practically that's the difference between a KV cache in the tens of gigabytes at long context and a state you can hold in a few megabytes.

**15. Mamba vs transformer at training?**
Mamba: $O(N \cdot d)$ via parallel scan. Transformer: $O(N^2 \cdot d)$ via attention. Mamba's compute scales linearly; transformer's quadratically.

> **Saying it out loud.** Training is linear versus quadratic. Attention builds an $N \times N$ score matrix, so cost scales with the square of sequence length; Mamba's scan is linear in $N$. At short context that doesn't matter much — attention's kernels are extremely well optimized and constants dominate — but the curves cross and then diverge fast. The honest caveat is that the crossover is further out than people expect, somewhere in the low thousands of tokens, so for a 2K-context model the theoretical advantage barely shows up.

---

## D. Comparing to other models

**16. Mamba vs LSTM?**
Both are RNNs in some sense. LSTM: nonlinear gates, parallel-unfriendly, vanishing gradients with depth. Mamba: linear recurrence, parallel scan, stable gradients via structured $A$ and HiPPO init. Mamba is what LSTMs always wanted to be.

> **Saying it out loud.** Mamba is basically what LSTMs wanted to be. Both keep a running state, but the LSTM squashes it through nonlinear gates every step, which forces sequential computation and gives you vanishing gradients over long sequences. Mamba keeps the recurrence linear, so it parallelizes with a scan, and it gets its gating from the input-dependent $\Delta t$ instead of from nonlinearities. Plus the structured, HiPPO-style $A$ keeps gradients well-behaved over thousands of steps, where an LSTM starts struggling past a few hundred.

**17. Mamba vs linear attention?**
Both are $O(N)$. Linear attention: constant $K, V$ projections. Mamba: input-dependent $B, C, \Delta t$ — more expressive. Empirically, Mamba beats linear attention on language tasks.

> **Saying it out loud.** They're cousins — both are linear-time and both maintain a fixed-size state instead of a growing cache. The difference is that linear attention's state update is essentially fixed: you accumulate key-value outer products with the same rule every step. Mamba makes the update itself input-dependent through $B$, $C$ and $\Delta t$, which is strictly more expressive, and Mamba-2 later showed these two families are two points on one spectrum. Empirically that extra expressiveness matters — linear attention consistently underperforms on language modeling perplexity while Mamba is roughly transformer-competitive.

**18. Mamba vs vanilla RNN?**
Vanilla RNN: random init, unstable, can't scale. Mamba: HiPPO-initialized, structured $A$, stable, scales. Different in practice despite similar mathematical form.

> **Saying it out loud.** On paper they look almost the same — a state that gets multiplied and added to each step. The difference is entirely in the details that decide whether it trains. A vanilla RNN has a dense random recurrent matrix and a nonlinearity, so gradients explode or vanish and you can't parallelize; Mamba has a structured diagonal $A$ with a principled init, no nonlinearity in the recurrence, and a parallel scan. It's a good reminder that in sequence modeling the math form tells you very little — initialization and hardware mapping are what make something scale.

**19. Why hasn't Mamba replaced transformers?**
Weaker in-context learning / copy-recall. Less mature ecosystem (FlashAttention, vLLM are transformer-specific). Scaling laws unclear at frontier scale (~100B+). Hybrid models seem to be the practical compromise.

> **Saying it out loud.** Three reasons, and only one of them is about quality. The real quality issue is recall: a fixed-size state has to compress everything, so exact copying and lookup from far back in the context is genuinely harder than for attention, which can just point at the token. Then there's ecosystem — FlashAttention, vLLM, paged KV caches, every serving stack is built for transformers, so switching costs are real. And nobody has published a clean frontier-scale result, so labs are hedging with hybrids rather than betting the run on pure SSM.

---

## E. Hybrid models

**20. What's a hybrid SSM-transformer?**
Mix attention layers and SSM layers. Attention layers handle copy/recall; SSM layers handle long-range mixing cheaply. Examples: Jamba, Zamba, Bamba, Hymba.

> **Saying it out loud.** A hybrid just interleaves the two layer types instead of picking a side. Most layers are SSM, which handles the cheap long-range mixing, and every so often you drop in a full attention layer to do the precise lookups that SSMs are bad at. The intuition is that you don't need attention everywhere — you need it in a few places where exact retrieval matters. Jamba, Zamba, Bamba and Hymba are all versions of this idea, and the typical ratio is something like one attention layer per six or seven SSM layers.

**21. What's Jamba?**
AI21 2024. 7-to-1 SSM-to-attention ratio. 256K+ context. Mamba blocks for cheap long-range; attention blocks for in-context behaviors; MoE on top. First production hybrid.

> **Saying it out loud.** Jamba was AI21's 2024 model and the first production-grade hybrid. It runs roughly seven Mamba blocks per attention block, adds mixture-of-experts on top, and supports 256K-plus context. The point of the design is that the attention layers keep in-context learning and copying intact while the Mamba majority keeps the KV cache small enough to actually serve long context. The headline number was that it fits 140K tokens of context on a single 80GB GPU, which a dense transformer of that size flatly cannot do.

**22. Why might hybrids beat pure SSM or pure attention?**
Pure SSM: cheap but weak ICL. Pure attention: strong ICL but expensive at long context. Hybrid: cheap at long context (mostly SSM) with attention layers preserving copy/recall.

> **Saying it out loud.** Because the two failure modes are complementary. Pure SSM is cheap but compresses context into a fixed state, so it fumbles exact recall; pure attention recalls perfectly but pays quadratic compute and a KV cache that eats your memory budget. Mix them and the handful of attention layers restore the induction-head behavior while the SSM majority keeps cost near-linear. The evidence is pretty striking — even one or two attention layers in an otherwise-SSM stack recovers most of the copy-recall gap.

**23. Open question: Does hybrid beat dense transformer at frontier scale?**
Empirical, debated. Jamba and similar models are competitive but no flagship 100B+ hybrid has clearly beaten a dense transformer of similar compute. Active research.

> **Saying it out loud.** Honestly, nobody has shown it cleanly yet. Hybrids like Jamba are competitive at their scale and clearly better on long-context memory footprint, but no 100B-plus hybrid has convincingly beaten a dense transformer trained on matched compute. The complication is that frontier training runs are so expensive nobody does the controlled comparison, so what we have is a lot of suggestive evidence and no decisive experiment. My read is that the win is on inference economics rather than raw quality — and that may be enough to make hybrids the default anyway.

---

## F. Subtleties

**24. What's Mamba-2?**
Dao & Gu 2024. "Transformers are SSMs": shows attention and SSM are mathematically related. Mamba-2 simplifies the parameterization with this structural understanding. Slightly faster training.

> **Saying it out loud.** Mamba-2 came from the 'Transformers are SSMs' paper by Dao and Gu in 2024, and the interesting part is the theory. They showed attention and structured SSMs are both instances of one framework built on semiseparable matrices, which means the two families are far more closely related than anyone assumed. Using that, they simplified the parameterization — a scalar-times-identity structure for $A$ — so the whole thing can be expressed with matrix multiplies and hit the tensor cores. The payoff is practical: two to eight times faster training than Mamba-1 at similar quality, with slightly less expressive $A$ as the tradeoff.

**25. Why is Mamba's HBM bandwidth efficiency important?**
Mamba's CUDA kernel keeps the state in SRAM during the scan (similar to FlashAttention's tiling). This makes the operation memory-bandwidth-efficient on modern GPUs. Without this, the parallel-scan version would be slow.

> **Saying it out loud.** Because on a modern GPU the bottleneck is moving data, not doing math. A naive scan would materialize the full hidden state for every position in high-bandwidth memory, and that traffic dominates the runtime — the arithmetic is basically free by comparison. Mamba's kernel keeps the state in SRAM and only writes out what's needed, exactly the same idea as FlashAttention's tiling. Without that, selectivity would have been a theoretically nice idea that ran slower than the attention it was supposed to replace.

**26. What's the in-context-learning gap for SSMs?**
Empirically, SSMs are weaker at copying tokens from earlier in the context (the "induction head" behavior). Transformers' attention naturally implements this; SSMs must approximate. Hybrid layers (one attention layer per few SSM) often suffice to close the gap.

> **Saying it out loud.** The gap shows up on copying. Transformers form induction heads that literally look back, find where a token appeared before, and read off what came next — attention makes that a one-step operation. An SSM has to reconstruct it out of a fixed-size compressed state, which is doable in principle but fragile and needs more capacity. It's the clearest known weakness of pure SSMs, and the standard fix is architectural rather than algorithmic: put in a couple of attention layers and the induction behavior comes right back.

**27. Can Mamba do beam search / batched generation efficiently?**
Yes — but the state per beam is $O(d)$, so memory scales with $d \times \text{beam-count}$ not $\text{seq-len} \times \text{beam-count}$. Better than attention's KV cache for batched generation at long context.

> **Saying it out loud.** Yes, and this is one of its underrated advantages. Each beam or batch element carries its own state, but that state is a fixed $O(d)$ object rather than a cache that grows with the sequence. So memory scales with batch times state size, not batch times context length times state size. In practice that means you can hold far more concurrent sequences on a GPU at long context — which is exactly the regime where transformer serving falls over from KV cache pressure.

---

## G. Practical / implementation

**28. Mamba implementation gotchas?**
The CUDA kernel is non-trivial. Float precision matters (state can drift in fp16; bf16 or fp32 for state recommended). Variable sequence lengths need padding handling.

> **Saying it out loud.** The main one is that you're depending on a hand-written CUDA kernel, so you're off the beaten path the moment you want something unusual. Numerical precision bites too: the state is accumulated over thousands of steps, so fp16 drifts and you want the state in fp32 or at least bf16 even when the rest of the model is half precision. And variable-length sequences need care, because padding tokens still get integrated into the recurrence unless you explicitly reset the state at document boundaries. That last one is a quiet correctness bug — it doesn't crash, it just makes long-context quality mysteriously worse.

**29. Where does Mamba fail?**
Tasks heavily reliant on exact copy from earlier in context (some tool-use, table-lookup-style tasks). Tasks where attention-style cross-token interactions are critical. Modern hybrids fix most of these.

> **Saying it out loud.** It fails where you need exact retrieval rather than a good summary. Things like copying a long identifier verbatim, looking up a row in a table you were given earlier, or precise tool-call arguments from far back in the prompt — those depend on pointing at a specific past token, and a compressed state can't do that reliably. Needle-in-a-haystack evals show the gap most sharply. Modern hybrids largely patch it, which is itself evidence that the problem is specifically the absence of attention, not something about linear recurrence in general.

**30. Future of SSMs in LLMs?**
Open questions: pure SSMs at frontier scale? Hybrid as new norm? Better selectivity mechanisms? Possibly the answer is "transformers + a few SSM layers" or "SSMs + a few attention layers" — frontier labs are actively exploring.

> **Saying it out loud.** My honest read is that the destination is hybrids, and the only real question is the mixing ratio. Pure SSM keeps stumbling on recall; pure attention keeps being too expensive at long context; and every serious long-context system shipping now has some of both. The open research questions are whether there's a better selectivity mechanism than input-dependent $\Delta t$, and whether SSM scaling laws hold past 100B parameters where nobody has looked. If I had to bet, it's mostly-attention with a few SSM layers at frontier scale and mostly-SSM at the edge, where the constant-memory decode is worth the most.

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
