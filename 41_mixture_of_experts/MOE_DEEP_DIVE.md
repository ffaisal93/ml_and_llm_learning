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

> **Saying it out loud.** In a normal transformer, every token passes through the same feed-forward block, so every parameter does work on every token whether it's relevant or not. MoE says: keep eight or sixty-four copies of that block and add a tiny router that picks two per token. It's a hospital triage desk — reception doesn't walk every patient past every specialist, it reads the case and sends them to the right two. The consequence is that total capacity is set by the number of experts while compute per token is set by $k$, and those numbers are now independent. Mixtral makes it concrete: 46.7 billion parameters total, about 12.9 billion active, running at roughly the cost of a 13B dense model.

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

> **Saying it out loud.** There are three arguments for MoE and one big caveat. Scaling laws say more parameters means better quality, and MoE gets you more parameters without more compute per token, so you move along a better curve for the same training budget. Inference economics follow directly — you serve a 47B-capacity model at 13B cost, and serving cost is what decides whether a model ships. And experts specialize, so you're not asking one set of weights to be equally good at Python and Portuguese. The caveat to name unprompted is memory: every expert has to be resident because the next token might route anywhere, so MoE trades memory for compute. That's a great trade on a cluster and a terrible one on a single GPU.

---

## 3. The routing math

*In plain language:* the router is a single small linear layer. It looks at a token and produces one number per expert, you keep the highest few, and you turn those few numbers into weights that sum to one. The equations below are that, written three slightly different ways, because where you put the softmax relative to the top-k changes how gradients behave.

### Top-k softmax routing (Switch Transformer, GShard)

$$
\begin{aligned}
\text{scores} &= W_{\text{router}} \cdot x \in \mathbb{R}^E \\
\text{top}_k\text{-idx} &= \mathrm{topk}(\text{scores}, k) \\
\text{gates} &= \mathrm{softmax}(\text{scores}[\text{top}_k\text{-idx}]) \\
\text{output} &= \sum_{i \in \text{top}_k\text{-idx}} \text{gates}_i \cdot \text{expert}_i(x)
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

> **Saying it out loud.** Whether you softmax before or after the top-k sounds like a detail and it changes the gradients. Softmax-then-top-k normalizes over all $E$ experts and then throws most of them away, so the weights you keep don't sum to one and the block's output magnitude drifts with router confidence. Top-k-then-softmax takes the raw scores of the winners and normalizes just those, so gates always sum to one and the gradient is concentrated on the experts that actually ran. Mixtral and DeepSeek both do the latter. The thing to say regardless of flavour is that no gradient reaches the *selection* itself — `topk` is discrete, so the router only learns through the gate values it assigned to whatever it happened to pick.

### Sigmoid routing (recent, DeepSeek-V3)

Use sigmoid instead of softmax for routing scores: each expert is selected independently with its own gate. Gives more flexibility (multiple experts can have high weight). Used in DeepSeek-V3.

> **Saying it out loud.** Softmax makes experts compete for a fixed budget: the scores must sum to one, so raising one expert's weight mechanically lowers everyone else's, even when several experts are all genuinely relevant to a token. Sigmoid scores each expert on its own, so the model can say "this token strongly needs experts 3, 17 and 40" without that being self-contradictory. This matters far more when you're picking 8 out of 256 than 2 out of 8, which is why it shows up in fine-grained architectures. It also composes cleanly with bias-based balancing, since adding a bias to an independent score doesn't get redistributed across every other expert the way a softmax would.

---

## 4. Load balancing — the central problem

If left unconstrained, the router collapses: a few experts get most tokens, others starve. Wasted parameters, training instability, deployment imbalance.

### The classic auxiliary loss (Switch, GShard)

*In plain language:* you want a penalty for "some experts are hogging all the tokens." The obstacle is that counting tokens per expert involves `topk`, which is a discrete pick — the counts are piecewise-constant, so differentiating them gives zero and the router would learn nothing. The loss below sidesteps that by multiplying the (non-differentiable) count $f_i$ against the (differentiable) average router probability $P_i$. The gradient flows only through $P_i$, with $f_i$ acting as a per-expert weight, so an overloaded expert gets its router logits pushed down.

Add a loss term that encourages balanced expert usage:

$$
f_i = \frac{1}{N} \sum_t \mathbf{1}[\text{expert}_i \in \text{top}_k(t)] \quad \text{(fraction of tokens choosing expert } i\text{)}
$$

$$
P_i = \frac{1}{N} \sum_t \text{softmax-score}_i(t) \quad \text{(average router prob for expert } i\text{)}
$$

$$
\mathcal{L}_{\text{balance}} = E \cdot \sum_i f_i \cdot P_i
$$

Minimized when $f_i \approx P_i \approx 1/E$ for all $i$. The $E$ factor sets the right scale. Added to total loss with coefficient $\alpha \approx 0.01$.

> **Saying it out loud.** The reason this loss looks odd is that the obvious version doesn't work. You'd like to penalize uneven token counts directly, but counts come out of `topk`, which is discrete, so their gradient is zero and the router would never hear about it. So the loss pairs the hard count $f_i$ with the soft average probability $P_i$: the count is effectively a constant weight, and all the learning signal flows through $P_i$. An expert that's currently overloaded has a large $f_i$, so the gradient pushes its router logits down hard. It bottoms out at 1 when everything is uniform, and the leading $E$ is there so that value is 1 regardless of expert count, which is what lets you use the same coefficient of about 0.01 across model sizes.

### Capacity factor

Each expert has a fixed capacity per batch — the maximum tokens it processes. If too many tokens route to one expert, the **excess tokens are dropped** (skipped or sent through a residual). This bounds the work per expert.

$$
\text{capacity} = \text{capacity-factor} \cdot \frac{\text{batch-size} \cdot \text{seq-len}}{E} \cdot k
$$

`capacity_factor = 1.0` is exact balance; `1.25` is common (allows 25% slack); higher reduces dropping but wastes compute.

### Token dropping

When experts overflow, excess tokens skip the expert and pass through unchanged via residual. This is **necessary** for parallelism (fixed shapes) but introduces quality loss. Modern systems try to minimize dropping.

> **Saying it out loud.** Capacity factor is where the theory meets the hardware. GPUs need fixed tensor shapes, so you have to size each expert's buffer before you know how routing will turn out — you allocate the perfectly-balanced share plus some slack, typically 25 percent. If more tokens than that pick one expert, the overflow gets dropped: those tokens skip the feed-forward block entirely and continue on the residual stream. Nothing errors, nothing logs, they just get less computation than their neighbours. That's the failure mode to name, because it's silent and it correlates with imbalance, so the tokens you drop are the common patterns routed to your most popular expert. And the slack isn't free either — over-provisioned buffers cost memory and get shipped across the network mostly empty.

### Auxiliary-loss-free balancing (DeepSeek-V3, 2024)

DeepSeek-V3's contribution: replace the auxiliary loss with **dynamic bias adjustments**. For each expert, maintain a bias $b_i$:

$$
\text{Score}(\text{token}, \text{expert } i) = \text{router-score}_i + b_i
$$

After each step, adjust $b_i$: increase if expert was underutilized, decrease if overutilized.

Avoids the auxiliary-loss interference with the main loss; balancing emerges naturally. Reportedly produces better expert specialization than aux-loss methods. Frontier interview-relevant.

> **Saying it out loud.** DeepSeek's observation is that the auxiliary loss is structurally in conflict with what you actually want. It's a gradient dragging the router away from "send this token where it'll be handled best" toward "send tokens evenly," so at any coefficient you're trading quality for balance. Their fix is to take balancing out of the loss entirely: give each expert a bias that's added to its routing score for selection purposes only, and after each step nudge those biases based on observed utilization — down for overused, up for underused. It's a thermostat, not a gradient. No interference with the language modeling objective, and the reported result is both better balance and better specialization, because routing is now free to be as opinionated as it likes.

### Routing collapse

Failure mode: routing concentrates on a few experts permanently. Causes:

- Initialization issue: some experts get a head start.
- Aux loss too weak.
- Capacity factor too high (no penalty for routing imbalance).

Symptoms: training plateau; some experts have huge gradients while others are dormant. Fixes: reset balanced routing, increase aux loss, lower capacity.

> **Saying it out loud.** Routing collapse is a rich-get-richer loop. An expert that happens to get slightly more tokens early trains slightly faster, so the router prefers it more, so it gets more tokens, and within a few thousand steps you have two experts doing everything. The important practical point is that it's much easier to prevent than to fix — once the neglected experts have drifted into uselessness, forcing tokens into them makes the loss worse and the router promptly learns to avoid them again. So: adequate balancing from step one, small router initialization so early routing is near-uniform, router z-loss to keep logits from saturating, and float32 for the routing softmax. And log per-expert token counts every step, because that's the only way you'll catch it early.

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

> **Saying it out loud.** The design question that's actually moved is granularity. Mixtral's eight full-size experts are the simple version, and the field has gone the other way — DeepSeek runs 256 small routed experts per layer and picks eight. The argument is combinatorial: 8-choose-2 gives you 28 possible expert combinations, while 64-choose-8 gives you over four billion, so for the same active compute you get a vastly richer space of specializations. Small experts can also afford to be narrow, where a big expert is forced to generalize because too much lands on it. Shared always-on experts complement this by absorbing the general-purpose work every token needs, so the routed ones don't each waste capacity relearning basic syntax. The cost of fine-graining is entirely on the systems side: more routing decisions, more destinations, more communication.

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

> **Saying it out loud.** Expert parallelism means different GPUs hold different experts, which solves the memory problem and creates a communication problem. Every MoE layer needs two all-to-alls: one to dispatch each token to whichever devices hold its chosen experts, one to gather the results back in order. All-to-all is the most expensive collective there is, because unlike an all-reduce it can't be decomposed into a tree — every pair of devices has genuine traffic. And you pay it per layer, so a 60-layer MoE is 120 all-to-alls per forward pass. Since sparsity made the compute *smaller* without making the data movement smaller, the communication-to-compute ratio is far worse than dense. The lever that matters most is topology: keep expert parallelism inside a node where NVLink is fast rather than across nodes on InfiniBand.

### Expert-data parallelism trade-offs

- **Pure expert parallelism:** each GPU has different experts; all data goes through them. Maximum expert capacity per GPU.
- **Expert + data parallel:** experts replicated across some GPUs; tokens split. Combines benefits.
- Modern: **3D parallelism** combines tensor + pipeline + expert + data.

### Token-level vs expert-level routing

- **Token-level (typical):** each token routes independently. Simpler.
- **Sequence-level / batch-level:** route entire sequences. Can improve cache reuse but loses token-specific specialization.

> **Saying it out loud.** Token-level routing is what everyone uses, and it's worth appreciating how fine-grained it really is: every token gets its own decision, at every layer, so a single word takes a different pair of experts at each depth of the network. Sequence-level routing would send an entire sequence to the same experts, which is enormously friendlier to the systems layer — you'd batch cleanly, cut all-to-all traffic, and get expert-weight reuse. The reason nobody does it is that you throw away the specialization that justified MoE in the first place, since the whole point is that different *tokens* need different processing. That tension between routing granularity and communication efficiency is the recurring theme in MoE systems work.

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

> **Saying it out loud.** The lineage is worth being able to recite. Shazeer's 2017 paper introduced sparsely-gated MoE. GShard in 2020 established the recipe everyone still uses — top-2 routing, capacity factor, auxiliary balancing loss. Switch in 2021 simplified to top-1 and proved it trains stably at scale. Mixtral in 2023 was the first genuinely good open-weight MoE, 46.7 billion total and 12.9 billion active — and note that the naive eight-times-seven arithmetic giving 56 billion is wrong, because attention, embeddings and norms are shared across experts and counted once. DeepSeek-V3 in 2024 pushed the sparsity ratio to 671 billion total on 37 billion active with auxiliary-loss-free balancing. And Llama-4 going MoE is the signal that dense frontier models are over.

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

> **Saying it out loud.** MoE wins because it breaks the link between capacity and per-token compute, and per-token compute is what you pay on every request forever. That's the whole economic argument, and it's why the Pareto frontier moved. The stability gap that made people nervous early on has largely closed — auxiliary-loss-free balancing plus router z-loss gets you dense-like training behaviour. What's genuinely still hard is worth stating plainly: memory, because you need all 671 billion parameters resident to serve 37 billion of them; communication, because all-to-all bandwidth caps how far you can shard; and cold start, because an expert that's barely been routed to is an expert that hasn't learned much, which is the tail end of the same collapse dynamic.

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

> **Saying it out loud.** If there's one correction to have ready, it's that MoE does *not* save memory. It saves compute per token; every expert still has to be in memory because the next token might route anywhere. The second is that top-2 beats top-1 not on quality of output but on stability — two experts means gradients reach two experts per token and the output varies smoothly as the router shifts weights, where top-1's gate is barely more than a scalar. The third is the load balancing loss: the leading $E$ is there so the balanced value is 1 regardless of expert count, which is what lets a coefficient of 0.01 transfer across model sizes. Getting those three right is most of what this topic is graded on.

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
