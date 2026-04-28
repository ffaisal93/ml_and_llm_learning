# Mixture of Experts (MoE): A Frontier-Lab Interview Deep Dive

> **Why this exists.** MoE went from research curiosity to frontier-LLM default in two years (Switch → GShard → Mixtral → DeepSeek-V3 → Llama-4). Interviewers ask: top-k routing math, load balancing loss, expert parallelism, why DeepSeek's auxiliary-loss-free balancing is significant. This document covers every component.

---

## 1. The big picture

A standard transformer FFN does $\mathrm{FFN}(x) = W_2 \cdot \mathrm{activation}(W_1 \cdot x)$. Every FFN layer has the same parameters; every token uses all of them.

An **MoE FFN** replaces this single FFN with $E$ parallel "experts" — separate FFN modules with their own weights. A **router** picks $k$ experts per token (typically $k = 2$, sometimes $k = 1$). Only the selected experts run.

For each token:

$$
\begin{aligned}
\text{scores} &= \text{router}(x) \in \mathbb{R}^E \quad &\text{(one score per expert)} \\
\text{top}_k &= \mathrm{argmax}_k(\text{scores}) \\
\text{weights} &= \mathrm{softmax}(\text{scores}[\text{top}_k]) \\
\text{output} &= \sum_{i \in \text{top}_k} \text{weights}_i \cdot \text{expert}_i(x)
\end{aligned}
$$

The **defining property of MoE**: total parameters scale with $E$, but **per-token compute** scales only with $k$. A 1T-parameter MoE with $k = 2$ out of $E = 64$ experts uses about the same compute per token as a 30B dense model.

---

## 2. Why MoE matters

Three reasons frontier labs went MoE:

### 1. More parameters at fixed compute

For a given training compute budget, MoE gives you a model with more total parameters than a dense equivalent. Empirically, more parameters → better quality (scaling laws), provided you can train them effectively.

### 2. Compute-efficient inference

Inference cost scales with active parameters, not total. Mixtral 8x7B has ~47B total parameters but ~13B active per token — runs at the cost of a 13B dense model.

### 3. Expert specialization

Different experts learn different "skills" (math, code, multilingual, etc.). At inference, the router picks the right experts for each token. This is more efficient than a dense model that has to support all skills with all parameters.

### Caveats

- **Memory cost.** MoE memory scales with total parameters — KV cache + weights of all experts must fit. Memory-bound at scale.
- **Routing instability.** Bad routing → some experts overused, others starved → wasted parameters.
- **Engineering complexity.** Expert parallelism, load balancing, communication patterns are non-trivial.

---

## 3. The routing math

### Top-k softmax routing (Switch Transformer, GShard)

$$
\begin{aligned}
\text{scores} &= W_{\text{router}} \cdot x \in \mathbb{R}^E \\
\text{top}_k\text{\_idx} &= \mathrm{topk}(\text{scores}, k) \\
\text{gates} &= \mathrm{softmax}(\text{scores}[\text{top}_k\text{\_idx}]) \\
\text{output} &= \sum_{i \in \text{top}_k\text{\_idx}} \text{gates}_i \cdot \text{expert}_i(x)
\end{aligned}
$$

$k = 1$ (Switch): cheapest, sometimes unstable.
$k = 2$ (Mixtral, default): better routing quality, slightly more compute.
$k = 4$+ (some research): rarely better than $k = 2$.

### Routing softmax: before or after top-k?

Two flavors:

- **Softmax-then-top-k.** Compute softmax over all $E$ scores; take top $k$; renormalize. Used in some early models.
- **Top-k-then-softmax.** Take top $k$ raw scores; softmax just those. Used in Mixtral, DeepSeek-V3.

The latter has cleaner gradient flow (softmax over a small subset; gradients only flow through chosen experts).

### Sigmoid routing (recent, DeepSeek-V3)

Use sigmoid instead of softmax for routing scores: each expert is selected independently with its own gate. Gives more flexibility (multiple experts can have high weight). Used in DeepSeek-V3.

---

## 4. Load balancing — the central problem

If left unconstrained, the router collapses: a few experts get most tokens, others starve. Wasted parameters, training instability, deployment imbalance.

### The classic auxiliary loss (Switch, GShard)

Add a loss term that encourages balanced expert usage:

$$
f_i = \frac{1}{N} \sum_t \mathbf{1}[\text{expert}_i \in \text{top}_k(t)] \quad \text{(fraction of tokens choosing expert } i\text{)}
$$

$$
P_i = \frac{1}{N} \sum_t \text{softmax\_score}_i(t) \quad \text{(average router prob for expert } i\text{)}
$$

$$
\mathcal{L}_{\text{balance}} = E \cdot \sum_i f_i \cdot P_i
$$

Minimized when $f_i \approx P_i \approx 1/E$ for all $i$. The $E$ factor sets the right scale. Added to total loss with coefficient $\alpha \approx 0.01$.

### Capacity factor

Each expert has a fixed capacity per batch — the maximum tokens it processes. If too many tokens route to one expert, the **excess tokens are dropped** (skipped or sent through a residual). This bounds the work per expert.

$$
\text{capacity} = \text{capacity\_factor} \cdot \frac{\text{batch\_size} \cdot \text{seq\_len}}{E} \cdot k
$$

`capacity_factor = 1.0` is exact balance; `1.25` is common (allows 25% slack); higher reduces dropping but wastes compute.

### Token dropping

When experts overflow, excess tokens skip the expert and pass through unchanged via residual. This is **necessary** for parallelism (fixed shapes) but introduces quality loss. Modern systems try to minimize dropping.

### Auxiliary-loss-free balancing (DeepSeek-V3, 2024)

DeepSeek-V3's contribution: replace the auxiliary loss with **dynamic bias adjustments**. For each expert, maintain a bias $b_i$:

$$
\text{Score}(\text{token}, \text{expert } i) = \text{router\_score}_i + b_i
$$

After each step, adjust $b_i$: increase if expert was underutilized, decrease if overutilized.

Avoids the auxiliary-loss interference with the main loss; balancing emerges naturally. Reportedly produces better expert specialization than aux-loss methods. Frontier interview-relevant.

### Routing collapse

Failure mode: routing concentrates on a few experts permanently. Causes:

- Initialization issue: some experts get a head start.
- Aux loss too weak.
- Capacity factor too high (no penalty for routing imbalance).

Symptoms: training plateau; some experts have huge gradients while others are dormant. Fixes: reset balanced routing, increase aux loss, lower capacity.

---

## 5. Expert design choices

### Number of experts

- **Few experts ($E = 8$, like Mixtral 8x7B):** simpler routing, less specialization potential.
- **Many experts ($E = 64+$ like DeepSeek-V3, GLaM):** more specialization, harder to train.
- **Trade-off:** more experts = more total parameters at same active compute, but routing complexity grows.

### Expert size

- Same as dense FFN size? (Mixtral) — easy implementation.
- Smaller experts? (DeepSeek-MoE: many small fine-grained experts) — better specialization, more compute on routing.

### Shared experts

DeepSeek-MoE introduces **shared experts** that always run for every token, alongside top-k routed experts. Captures common functionality; routed experts specialize. Reportedly improves quality and stability.

### Expert FFN architecture

Standard: same as dense FFN ($W_1$, activation, $W_2$). Some variants use SwiGLU or specialized architectures. Mostly unchanged from dense baseline.

---

## 6. Expert parallelism

For MoE at scale ($E$ experts, large total params), you need to distribute experts across GPUs:

### All-to-all communication

Each token's representation must reach the GPU(s) holding its top-k experts. After expert computation, results return to the original GPU.

1. Each GPU has some tokens and some experts.
2. Routing decides which expert (= which GPU) each token goes to.
3. all-to-all: scatter tokens to their destination GPUs.
4. Experts compute.
5. all-to-all: gather results back.

Two all-to-all communications per MoE layer. **Network bandwidth is often the bottleneck** for MoE training and inference.

### Expert-data parallelism trade-offs

- **Pure expert parallelism:** each GPU has different experts; all data goes through them. Maximum expert capacity per GPU.
- **Expert + data parallel:** experts replicated across some GPUs; tokens split. Combines benefits.
- Modern: **3D parallelism** combines tensor + pipeline + expert + data.

### Token-level vs expert-level routing

- **Token-level (typical):** each token routes independently. Simpler.
- **Sequence-level / batch-level:** route entire sequences. Can improve cache reuse but loses token-specific specialization.

---

## 7. Production MoE models

### Switch Transformer (Google, 2021)

First major MoE LLM. $k = 1$, simplest variant. Established that MoE works at scale.

### GShard (Google, 2020)

$k = 2$, established the load-balancing loss formulation that became standard.

### GLaM (Google, 2021)

1.2T parameters with 64 experts; demonstrated MoE quality matching/beating dense at smaller compute.

### Mixtral 8x7B (Mistral, 2023)

First open-source flagship MoE. 47B total / 13B active. Quality close to LLaMA-2 70B at much lower inference cost. **Defined the modern open MoE template.**

### DeepSeek-MoE (DeepSeek, 2024)

Fine-grained experts (many small) + shared experts. Better specialization. Set the stage for V2.

### DeepSeek-V2/V3 (2024)

236B / 671B total, 21B / 37B active. Auxiliary-loss-free balancing. MLA for KV. Open weights. Frontier-quality MoE.

### Llama-4 (Meta, 2025)

Confirmed MoE as the frontier default. Even Llama abandoned dense.

---

## 8. Why MoE wins on the compute frontier

### Scaling laws

Empirically, MoE scaling laws are *more favorable* than dense: doubling parameters in MoE costs less compute than doubling dense parameters. The compute-quality Pareto frontier shifts.

### Inference economics

Active parameters determine compute (matmul cost). Total parameters determine quality. MoE breaks the link: more total quality, same active compute.

### Training stability

Auxiliary-loss-free MoE (DeepSeek) has matched dense stability. Earlier MoE was less stable; the gap has narrowed.

### What's still hard

- **Memory.** MoE total parameters are huge. Inference at 671B is hard regardless of activation.
- **Communication.** All-to-all bandwidth is critical and sometimes bottleneck.
- **Cold start.** New experts that haven't been used much underperform.

---

## 9. Interview gotchas

| Gotcha | Strong answer |
|---|---|
| "MoE saves compute — does it save memory?" | No. Memory scales with total parameters, not active. MoE saves *compute* per token; memory still scales with $E$. |
| "Why top-2 and not top-1?" | Top-2 (Mixtral) gives more routing flexibility — gradient flows through 2 experts, mixed outputs are richer. Top-1 (Switch) is cheaper but more brittle. |
| "What is routing collapse?" | Router concentrates on a few experts; others starve. Wasted parameters, training instability. Fix with balancing losses or DeepSeek-style bias adjustment. |
| "Why is the load balancing loss $E \cdot \sum f \cdot P$?" | At uniform balance ($f_i = P_i = 1/E$), the loss equals $E \cdot E \cdot (1/E^2) = 1$ regardless of expert count. The leading $E$ keeps the regularizer's strength scale-invariant; without it, minimum would shrink as $1/E$. |
| "What's a capacity factor?" | Maximum tokens per expert per batch. Above it, tokens are dropped via residual. ~1.25 typical. Trade-off: low CF wastes compute (drops); high CF wastes capacity (over-provisioned). |
| "Why is communication the bottleneck?" | All-to-all between GPUs at every MoE layer. Scales with batch × seq × experts. Can dominate over compute. |
| "What's auxiliary-loss-free balancing?" | DeepSeek-V3. Add bias to router scores; adjust biases per-expert based on usage. No aux loss interfering with main loss. |
| "Why fine-grained vs coarse experts?" | Fine-grained (many small): better specialization, more routing overhead. Coarse (few large, like Mixtral): simpler, less specialization. |

---

## 10. The 8 most-asked MoE interview questions

1. **What is MoE?** Replace single FFN with $E$ experts; router picks $k$ per token. Active compute scales with $k$; total parameters scale with $E$.
2. **Walk through routing.** $\text{scores} = W_{\text{router}} \cdot x$; $\mathrm{topk}$; softmax over selected; weighted combine of expert outputs.
3. **What's load balancing for?** Prevent routing collapse where a few experts get all tokens. Aux loss $E \cdot \sum f \cdot P$ or DeepSeek-style bias adjustment.
4. **What's a capacity factor?** Max tokens per expert per batch. Excess dropped via residual.
5. **Top-1 vs top-2?** Top-1 cheaper, top-2 more stable. Mixtral and most modern use top-2.
6. **What's expert parallelism?** Distribute experts across GPUs. All-to-all communication scatters tokens to expert GPUs and gathers results.
7. **Memory vs compute trade-off in MoE?** Compute scales with active params (~$k$ experts). Memory scales with total params (~$E$ experts).
8. **What's DeepSeek's auxiliary-loss-free balancing?** Add per-expert bias to router scores; adjust biases dynamically based on usage. No interfering aux loss.

---

## 11. Drill plan

1. Whiteboard top-k routing including softmax and weighted combination.
2. Memorize the load balancing loss formula and intuition.
3. Know capacity factor and token dropping.
4. Understand all-to-all communication for expert parallelism.
5. Know DeepSeek's auxiliary-loss-free contribution.
6. Drill `INTERVIEW_GRILL.md`.

---

## 12. Further reading

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (2017) — original.
- Fedus et al., "Switch Transformer" (2021).
- Lepikhin et al., "GShard" (2020).
- Du et al., "GLaM" (2021).
- Mistral, "Mixtral 8x7B" (2023).
- DeepSeek, "DeepSeek-MoE: Towards Ultimate Expert Specialization" (2024).
- DeepSeek, "DeepSeek-V3 Technical Report" (2024) — auxiliary-loss-free.
