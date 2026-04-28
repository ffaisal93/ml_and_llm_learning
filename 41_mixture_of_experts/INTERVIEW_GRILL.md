# Mixture of Experts — Interview Grill

> 35 questions on MoE. Drill until you can answer 25+ cold.

---

## A. Architecture

**1. What is MoE?**
Replace dense FFN with $E$ parallel experts; a router picks $k$ experts per token; only chosen experts run. Total parameters scale with $E$; active compute scales with $k$. Decouples capacity from compute.

**2. Walk me through MoE routing.**

$$
\begin{aligned}
\text{scores} &= W_{\text{router}} \cdot x \in \mathbb{R}^E \\
\text{top}_k\text{\_idx} &= \operatorname{topk}(\text{scores}, k) \\
\text{gates} &= \operatorname{softmax}(\text{scores}[\text{top}_k\text{\_idx}]) \\
\text{output} &= \sum_i \text{gates}_i \cdot \text{expert}_i(x)
\end{aligned}
$$

Expert outputs weighted-sum into final output.

**3. Top-1 vs top-2 routing?**
Top-1 (Switch): each token uses exactly one expert. Simplest, cheapest. Top-2 (Mixtral, modern default): each token uses two experts with weighted combination. More stable, slightly more compute.

**4. Why is MoE compute-efficient?**
Inference cost scales with active parameters, not total. Mixtral 8x7B: 47B total but ~13B active per token — runs at the cost of a 13B dense model.

**5. Does MoE save memory?**
No. Total parameters are still in memory. MoE saves *compute*, not memory. KV cache + all expert weights must fit. Memory-bound at scale.

**6. What does MoE replace in a transformer?**
The FFN sublayer in each transformer block. Attention is unchanged. Modern MoE replaces every FFN; some early designs alternated MoE and dense layers.

---

## B. Load balancing

**7. Why is load balancing critical?**
Without it, the router collapses: a few experts get most tokens, others starve. Wasted parameters, training instability, uneven inference cost.

**8. What's the auxiliary loss formula?**
$\mathcal{L}_{\text{balance}} = E \cdot \sum_i f_i \cdot P_i$ where $f_i$ is the fraction of tokens routing to expert $i$ and $P_i$ is the average router probability for expert $i$. Minimized when both are uniform $1/E$.

**9. Why multiply by $E$ in the aux loss?**
Sets the right scale: at perfect balance ($f_i = P_i = 1/E$ for all $i$), $E \cdot \sum_{i=1}^E (1/E)(1/E) = E \cdot E \cdot (1/E^2) = 1$, a constant independent of $E$. Without the leading $E$, the minimum would be $1/E$ and shrink with more experts — making the regularizer weaker as $E$ grows.

**10. What's the typical aux loss coefficient?**
$\alpha \approx 0.01$ (very small). Strong enough to prevent collapse but weak enough to not interfere with the main loss.

**11. What's a capacity factor?**
Maximum tokens an expert can process per batch. $\text{capacity} = \text{capacity\_factor} \cdot (\text{batch} \cdot \text{seq} / E) \cdot k$. CF $= 1.0$ is exact balance; CF $= 1.25$ is common with 25% slack.

**12. What happens to overflow tokens?**
They're dropped — skipped via residual connection. Necessary for fixed shape compute. Quality cost.

**13. What's auxiliary-loss-free balancing?**
DeepSeek-V3. Add a per-expert bias $b_i$ to router scores. Adjust each $b_i$ dynamically: increase if underused, decrease if overused. No aux loss to interfere with main loss. Reportedly produces better specialization.

**14. What's routing collapse?**
Router permanently concentrates on a few experts. Causes: weak aux loss, bad initialization, capacity factor too high. Fix: stronger balancing, restart with balanced routing.

---

## C. Expert design

**15. How many experts is typical?**
Mixtral: $E = 8$. DeepSeek-V3: $E = 64+$. GLaM: 64. Trade-off: more experts = more total params at same active compute, but routing harder.

**16. Why fine-grained experts?**
DeepSeek-MoE introduces many small experts (vs few large). Smaller per-expert capacity → better specialization. More routing overhead but quality gains.

**17. What are shared experts?**
Always-active experts that run for every token alongside top-k routed experts. Capture common functionality; routed experts specialize. DeepSeek innovation; reportedly improves stability.

**18. Where do MoE layers go in the transformer?**
Replace the FFN in each transformer block. Attention stays dense. Some research alternates MoE and dense FFN layers; pure-MoE-FFN is the modern norm.

**19. Are attention layers ever MoE?**
Rare in mainstream models. Attention is already capacity-rich; conversion to MoE has not shown clear gains. FFN-MoE is the dominant pattern.

---

## D. Expert parallelism

**20. What's expert parallelism?**
Distribute experts across GPUs. Each GPU holds some experts; tokens route to whichever GPU has their assigned expert.

**21. What's all-to-all communication?**
At every MoE layer: each token's representation is sent to the GPU(s) holding its top-k experts. After expert computation, results return. Two all-to-all per layer.

**22. Why is communication a bottleneck for MoE?**
All-to-all scales with batch × seq × top-k. Often dominates compute. Network bandwidth (NVLink within node, IB across) is the binding constraint at scale.

**23. How does MoE combine with other parallelism?**
3D parallelism: tensor parallel (within expert), data parallel (across batches), expert parallel (different experts on different GPUs). Pipeline parallel may also be added. Modern frontier: 4D+ parallelism configurations.

**24. Inference parallelism for MoE?**
Same patterns as training. Expert parallelism is essential for large-MoE inference. Vector + KV cache must fit; expert weights must be reachable.

---

## E. Production MoE models

**25. What was Switch Transformer's contribution?**
Google 2021. First major MoE LLM. Simple top-1 routing. Established that MoE works at scale and trains stably with aux loss.

**26. What was GShard's contribution?**
Google 2020. Top-2 routing + capacity factor + load balancing loss. Defined the standard MoE recipe that Mixtral and many followers use.

**27. What's special about Mixtral 8x7B?**
First open-source flagship MoE. ~47B total / ~13B active. Quality near LLaMA-2 70B at much lower inference cost. Top-2 routing, 8 experts. Defined the open MoE template.

**28. What's special about DeepSeek-V3?**
671B total / 37B active. Auxiliary-loss-free balancing. Many fine-grained experts + shared experts. MLA for KV cache. Open weights. Frontier-quality with ~10% the inference cost of similar dense models.

**29. Why has every major lab gone MoE?**
Better scaling laws (more params for same compute). Better inference economics (active $\ll$ total). Better expert specialization. The compute-quality Pareto frontier shifted.

---

## F. Subtleties

**30. Why is training MoE less stable than dense?**
Routing decisions are non-differentiable (top-k); gradients flow through chosen experts only. Imbalance amplifies. Capacity factors create token dropping. Aux loss can interfere. Modern systems have largely solved this — DeepSeek-V3 is as stable as dense.

**31. Why does sigmoid routing (DeepSeek-V3) help?**
Each expert is selected independently with its own gate. More flexibility — multiple experts can have high weight without being forced into a softmax simplex. Plays well with bias-based balancing.

**32. What's expert specialization?**
Different experts learn different "skills" — some experts handle math, others code, others multilingual. Empirical observation; encouraged by routing patterns. Better with fine-grained experts (DeepSeek-MoE).

**33. What's the relationship between MoE and ensembling?**
MoE is a learned soft ensemble: the router decides per-token which "members" (experts) contribute. Differs from ensembles in that experts are jointly trained and only $k$ are active per token.

**34. Why does MoE often produce more diverse outputs than dense?**
Different routing per token can specialize behavior. Dense models smooth across all skills; MoE can specialize. Empirically: MoE's outputs sometimes more varied.

**35. What's the future of MoE?**
Open questions: even more fine-grained experts (1000+)? Hybrid attention-MoE? Better balancing without bias hack? More active experts ($k = 4+$) at scale? Field is moving fast; the answer might be different in a year.

---

## Quick fire

**36.** *Switch Transformer's $k$?* 1.
**37.** *Mixtral's $k$?* 2.
**38.** *Default load balance coefficient?* ~0.01.
**39.** *Mixtral total / active params?* 47B / 13B.
**40.** *DeepSeek-V3 total / active params?* 671B / 37B.

---

## Self-grading

If you can't answer 1-15, you don't know MoE. If you can't answer 16-30, you'll struggle on architecture interviews. If you can't answer 31-40, frontier-lab interviews will go past you.

Aim for 25+/40 cold.
