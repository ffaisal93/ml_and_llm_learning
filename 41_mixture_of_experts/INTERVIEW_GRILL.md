# Mixture of Experts — Interview Grill

> 35 questions on MoE. Drill until you can answer 25+ cold.

---

## A. Architecture

**1. What is MoE?**
Replace dense FFN with $E$ parallel experts; a router picks $k$ experts per token; only chosen experts run. Total parameters scale with $E$; active compute scales with $k$. Decouples capacity from compute.

> **Saying it out loud.** Mixture of experts is the idea that a model shouldn't have to use all of itself on every token. You take the feed-forward block in each transformer layer and replace it with eight or sixty-four copies, then add a tiny router that looks at each token and picks two of them. It's like a hospital: reception doesn't send every patient past every specialist, it reads the case and routes to the right two. So total capacity grows with the number of experts while the work per token stays flat. The number that makes it concrete is Mixtral — 46.7 billion parameters total, about 12.9 billion active per token, so it costs about what a 13B dense model costs to run.

**2. Walk me through MoE routing.**

$$
\begin{aligned}
\text{scores} &= W_{\text{router}} \cdot x \in \mathbb{R}^E \\
\text{top}_k\text{-idx} &= \mathrm{topk}(\text{scores}, k) \\
\text{gates} &= \mathrm{softmax}(\text{scores}[\text{top}_k\text{-idx}]) \\
\text{output} &= \sum_i \text{gates}_i \cdot \text{expert}_i(x)
\end{aligned}
$$

Expert outputs weighted-sum into final output.

> **Saying it out loud.** Routing is four lines. A small linear layer maps the token to one score per expert. You take the top $k$ scores. You softmax just those $k$ so the gates sum to one — which matters, because otherwise the block's output magnitude would swing with how confident the router happened to be. Then you run those experts and take the gated weighted sum. Two details worth adding: the decision is per token, not per sequence, and it's recomputed at every layer, so a single word takes a different pair of experts at each depth. And the `topk` is discrete, so no gradient flows through *which* experts got picked — only through the gates assigned to the ones that did.

**3. Top-1 vs top-2 routing?**
Top-1 (Switch): each token uses exactly one expert. Simplest, cheapest. Top-2 (Mixtral, modern default): each token uses two experts with weighted combination. More stable, slightly more compute.

> **Saying it out loud.** Top-1 is Switch Transformer's choice: cheapest possible, one expert per token, and it proved MoE could train at scale at all. The problem is brittleness — with a single expert selected, the gate is essentially a scalar rescale and the router gets a very thin learning signal, so it's more prone to collapse. Top-2 gives you a genuine blend of two experts, so gradients reach two of them per token and the output varies smoothly as the router shifts its weights. You're paying double the FFN compute for that. Top-2 is the default because that stability is worth the compute, and current frontier models like DeepSeek-V3 push $k$ higher still with many smaller experts.

**4. Why is MoE compute-efficient?**
Inference cost scales with active parameters, not total. Mixtral 8x7B: 47B total but ~13B active per token — runs at the cost of a 13B dense model.

> **Saying it out loud.** FLOPs per token depend on how many parameters actually participate, and in a top-2-of-8 MoE that's a quarter of the feed-forward weights. So you get the knowledge capacity of a much bigger model at the compute cost of a small one. Mixtral is the clean example: 46.7 billion total, 12.9 billion active, matching or beating a 70B dense model while costing about what a 13B costs to serve. Worth noting the arithmetic isn't naive — eight times seven would be 56, but attention, embeddings, and norms are shared across all experts and counted once, so only the FFN blocks are replicated. Getting that right is a good signal that you know where MoE actually lives in the architecture.

**5. Does MoE save memory?**
No. Total parameters are still in memory. MoE saves *compute*, not memory. KV cache + all expert weights must fit. Memory-bound at scale.

> **Saying it out loud.** No, and this is the misconception worth correcting immediately. Every expert has to be resident in memory, because the next token might route anywhere, so you're holding all 46.7 billion of Mixtral's parameters even though you compute with 12.9 billion. MoE trades memory for compute, full stop. That's why it's a great fit for a well-provisioned inference fleet and a bad fit for a single consumer GPU, where the binding constraint is exactly the thing MoE makes worse. It also means MoE inference is even more memory-bandwidth-bound than dense inference: you're moving a lot of weights around relative to the arithmetic you do with them.

**6. What does MoE replace in a transformer?**
The FFN sublayer in each transformer block. Attention is unchanged. Modern MoE replaces every FFN; some early designs alternated MoE and dense layers.

> **Saying it out loud.** It replaces the feed-forward sublayer, and only that — attention stays fully dense. There's a good reason: the FFN is where most of the parameters live, roughly two-thirds of a transformer block, so that's where you get leverage from sparsity. Attention is comparatively parameter-light and it's the part that has to mix information across positions, which routing would disrupt. Early designs like GShard alternated, putting an MoE layer every other block to keep communication costs down; modern models generally convert every FFN. It's a useful thing to say precisely, because it's what makes the parameter arithmetic come out right.

---

## B. Load balancing

**7. Why is load balancing critical?**
Without it, the router collapses: a few experts get most tokens, others starve. Wasted parameters, training instability, uneven inference cost.

> **Saying it out loud.** Because MoE has a rich-get-richer failure built into it. An expert that happens to receive slightly more tokens early gets slightly more gradient, becomes slightly better, so the router prefers it more, so it gets even more tokens. Within a few thousand steps you can have two experts doing all the work and six that never learned anything but that you're still paying to store and shard. It also wrecks throughput in a distributed setting, because the GPU holding the popular expert becomes the bottleneck while the others idle. So balancing isn't a nicety — without it, MoE degenerates into a small dense model with expensive dead weight.

**8. What's the auxiliary loss formula?**
$\mathcal{L}_{\text{balance}} = E \cdot \sum_i f_i \cdot P_i$ where $f_i$ is the fraction of tokens routing to expert $i$ and $P_i$ is the average router probability for expert $i$. Minimized when both are uniform $1/E$.

> **Saying it out loud.** The formula is expert count times the sum over experts of the hard fraction times the mean router probability, and the *reason it's built that way* is the interesting part. You'd like to just penalize uneven token counts, but counts come from `topk`, which is discrete, so they have zero gradient and would teach the router nothing. So the loss pairs the count $f_i$, which acts as a constant weight, with $P_i$, the mean softmax probability, which is fully differentiable. All the gradient flows through $P_i$, scaled by how overloaded that expert currently is — so a popular expert has a big $f_i$ and the loss pushes its router logits down. It's minimized when both are uniform at $1/E$.

**9. Why multiply by $E$ in the aux loss?**
Sets the right scale: at perfect balance ($f_i = P_i = 1/E$ for all $i$), $E \cdot \sum_{i=1}^E (1/E)(1/E) = E \cdot E \cdot (1/E^2) = 1$, a constant independent of $E$. Without the leading $E$, the minimum would be $1/E$ and shrink with more experts — making the regularizer weaker as $E$ grows.

> **Saying it out loud.** It's a normalization so the loss means the same thing regardless of how many experts you have. Without the leading $E$, perfectly balanced routing gives you a loss of $1/E$ — so with 8 experts the floor is 0.125 and with 256 experts it's 0.004, and your fixed coefficient of 0.01 would be doing wildly different amounts of work in the two cases. Multiply by $E$ and the balanced value is exactly 1 no matter what, so the coefficient transfers across model sizes. It's a small detail but it's exactly the kind of thing that shows you've read the loss rather than memorized it.

**10. What's the typical aux loss coefficient?**
$\alpha \approx 0.01$ (very small). Strong enough to prevent collapse but weak enough to not interfere with the main loss.

> **Saying it out loud.** About 0.01, and it's genuinely a knife edge in both directions. Too small and you get collapse — the balancing term never overcomes the rich-get-richer dynamic and most of your experts die. Too large and something subtler goes wrong: the router starts optimizing for even distribution rather than sending tokens where they'd be handled best, so you get perfect balance and worse quality, because you've effectively made the routing random. That inherent tension between balance and quality is precisely what DeepSeek's auxiliary-loss-free approach was designed to escape.

**11. What's a capacity factor?**
Maximum tokens an expert can process per batch. $\text{capacity} = \text{capacity-factor} \cdot (\text{batch} \cdot \text{seq} / E) \cdot k$. CF $= 1.0$ is exact balance; CF $= 1.25$ is common with 25% slack.

> **Saying it out loud.** Capacity factor exists because GPUs need fixed tensor shapes. You have to allocate a buffer per expert before you know how routing will turn out, so you size it as the perfectly-balanced share times some slack — 1.25 means each expert can take 25 percent more than its fair share. Set it too low and you drop a lot of tokens. Set it too high and you're allocating and communicating buffers that sit mostly empty, which costs both memory and all-to-all bandwidth. It's the concrete engineering reason load balancing matters: with perfect balance you could run at 1.0 and waste nothing.

**12. What happens to overflow tokens?**
They're dropped — skipped via residual connection. Necessary for fixed shape compute. Quality cost.

> **Saying it out loud.** They get dropped, which means the token skips the feed-forward block entirely and just passes through on the residual stream. Nothing crashes, nothing warns you — the token simply gets less computation than its neighbours. That's the failure mode I'd emphasize, because it's invisible unless you explicitly log the drop rate, and a few percent of tokens quietly bypassing every MoE layer is a real quality hit. It also correlates with imbalance, so the tokens most likely to be dropped are the ones routed to your most popular expert, which is to say the common patterns. Inference typically runs with a higher capacity factor or dropless kernels precisely to avoid it.

**13. What's auxiliary-loss-free balancing?**
DeepSeek-V3. Add a per-expert bias $b_i$ to router scores. Adjust each $b_i$ dynamically: increase if underused, decrease if overused. No aux loss to interfere with main loss. Reportedly produces better specialization.

> **Saying it out loud.** DeepSeek's insight is that the auxiliary loss is fighting the main objective — it's a gradient pulling the router away from good routing and toward even routing. So they remove it entirely and balance with a control loop instead. Each expert gets a bias added to its routing score, used only for the top-k selection, not for the gate weights. After each step you look at utilization and nudge the biases: overused expert, lower its bias; underused, raise it. No gradient, no interference with the language modeling loss, just a thermostat. The result is both better balance and better expert specialization, since routing is now free to be as opinionated as it wants and balance is enforced outside the loss.

**14. What's routing collapse?**
Router permanently concentrates on a few experts. Causes: weak aux loss, bad initialization, capacity factor too high. Fix: stronger balancing, restart with balanced routing.

> **Saying it out loud.** Routing collapse is the state where the router has picked its favourites and won't come back. It's self-reinforcing, and once it's set in it's usually not recoverable by turning up the balancing coefficient mid-run, because the neglected experts have drifted so far from useful that even if you force tokens into them the loss gets worse and the router learns to avoid them again. So it's a problem you prevent rather than fix: adequate balancing coefficient from step one, small router initialization so early routing is near-uniform, and router z-loss to stop the logits growing large and saturating. The diagnostic is trivial and everyone forgets it — log per-expert token counts every step and watch for the distribution going bimodal early.

---

## C. Expert design

**15. How many experts is typical?**
Mixtral: $E = 8$. DeepSeek-V3: $E = 64+$. GLaM: 64. Trade-off: more experts = more total params at same active compute, but routing harder.

> **Saying it out loud.** Eight is the small end and modern frontier models have moved well past it — DeepSeek-V3 runs 256 routed experts per layer plus shared ones. More experts means more total capacity at unchanged active compute, which sounds free, and the costs are all on the systems side: routing gets harder as the router has to discriminate among more options, load balance gets harder, and all-to-all communication grows. There's also a per-expert data hunger issue — with 256 experts each one sees a fraction of a percent of your tokens, so you need enormous training corpora to train them all well. The trend is clearly toward many small experts rather than few large ones.

**16. Why fine-grained experts?**
DeepSeek-MoE introduces many small experts (vs few large). Smaller per-expert capacity → better specialization. More routing overhead but quality gains.

> **Saying it out loud.** Fine-grained means you split the same total parameter budget into many more, smaller experts and select more of them per token. The argument is combinatorial: with 8 experts choosing 2, there are 28 possible combinations, but with 64 experts choosing 8 there are over four billion. That's a vastly richer space of specializations available for the same compute. And each small expert can afford to be narrowly specialized, where a large expert is forced to be a generalist because too many different token types land on it. The cost is routing and communication overhead, since you're now making a finer-grained decision and shipping tokens to more destinations per layer.

**17. What are shared experts?**
Always-active experts that run for every token alongside top-k routed experts. Capture common functionality; routed experts specialize. DeepSeek innovation; reportedly improves stability.

> **Saying it out loud.** Shared experts run on every token unconditionally, in addition to whichever routed experts get selected. The reasoning is about redundancy: without them, every specialized expert has to independently relearn the general-purpose stuff that all tokens need — basic syntax, common patterns — which wastes capacity across all of them. Give the model one always-on expert to absorb that common knowledge and the routed experts are free to be genuinely specialized. It also stabilizes training, because every token now has a guaranteed compute path that doesn't depend on the router being sensible, which softens the damage from early bad routing and from dropped tokens.

**18. Where do MoE layers go in the transformer?**
Replace the FFN in each transformer block. Attention stays dense. Some research alternates MoE and dense FFN layers; pure-MoE-FFN is the modern norm.

> **Saying it out loud.** They go where the FFNs are, and modern models convert every one of them. Early designs like GShard and Switch alternated — MoE every second or fourth layer — mainly to limit how many all-to-all communication rounds you pay per forward pass, since each MoE layer costs two. As interconnects improved that constraint relaxed. One pattern that persists is keeping the first layer or two dense, on the theory that the earliest layers do generic low-level processing where there's nothing to specialize on, so routing there is wasted overhead.

**19. Are attention layers ever MoE?**
Rare in mainstream models. Attention is already capacity-rich; conversion to MoE has not shown clear gains. FFN-MoE is the dominant pattern.

> **Saying it out loud.** There's research on it but nothing mainstream, and the reasons are both structural and practical. Attention's job is mixing information *across* positions, so routing different tokens to different attention experts fragments exactly the thing attention exists to do. Attention is also parameter-light relative to the FFN, so the capacity payoff is much smaller. And there's a memory problem people miss: expert-specific key and value projections would multiply your KV cache, which is already the binding constraint on serving long contexts. The trend has actually gone the other way — DeepSeek's MLA compresses the KV cache rather than expanding it.

---

## D. Expert parallelism

**20. What's expert parallelism?**
Distribute experts across GPUs. Each GPU holds some experts; tokens route to whichever GPU has their assigned expert.

> **Saying it out loud.** Expert parallelism is the obvious sharding strategy for MoE: the experts are independent, so you put different ones on different GPUs. With 64 experts across 8 GPUs, each device holds 8, and no single device ever needs the whole model's FFN weights. What makes it different from tensor parallelism is that the *data* has to move rather than being replicated — a token computed on GPU 0 whose expert lives on GPU 5 has to physically travel there and come back. So expert parallelism converts a memory problem into a communication problem, and whether that's a good trade depends entirely on your interconnect.

**21. What's all-to-all communication?**
At every MoE layer: each token's representation is sent to the GPU(s) holding its top-k experts. After expert computation, results return. Two all-to-all per layer.

> **Saying it out loud.** All-to-all is the collective where every GPU sends a different slice of data to every other GPU. In an MoE layer you need it twice: once to dispatch each token to whichever devices hold its chosen experts, and once to gather the results back so the tokens are reassembled in their original order. It's the most expensive collective there is, because unlike an all-reduce it can't be decomposed into a tree — every pair of devices has genuine traffic. And you pay it per MoE layer, so a 60-layer model with MoE everywhere is 120 all-to-alls per forward pass.

**22. Why is communication a bottleneck for MoE?**
All-to-all scales with batch × seq × top-k. Often dominates compute. Network bandwidth (NVLink within node, IB across) is the binding constraint at scale.

> **Saying it out loud.** Because the volume scales with batch times sequence length times hidden dimension times $k$, and it happens twice per layer, while the expert computation itself got *smaller* thanks to sparsity. So the ratio of communication to compute is much worse than in a dense model — you made the arithmetic cheaper without making the data movement cheaper. The number that decides your architecture is the bandwidth cliff: NVLink inside a node is on the order of hundreds of gigabytes per second, InfiniBand across nodes is several times slower, so keeping expert parallelism within a node whenever possible is the single biggest systems lever. That's also why capacity factor matters — every unit of slack is buffer you're shipping across the wire empty.

**23. How does MoE combine with other parallelism?**
3D parallelism: tensor parallel (within expert), data parallel (across batches), expert parallel (different experts on different GPUs). Pipeline parallel may also be added. Modern frontier: 4D+ parallelism configurations.

> **Saying it out loud.** You compose them along different axes. Data parallelism splits the batch, pipeline parallelism splits the layers across devices, tensor parallelism splits individual matrices, and expert parallelism splits the experts. They're orthogonal, so a frontier run uses all four at once and the configuration is a real optimization problem. The rule that drives the layout is topology: put the chattiest dimensions — tensor and expert parallelism — inside a node where NVLink is fast, and put the cheaper ones, data and pipeline, across nodes. Getting that mapping wrong is a common way to lose half your throughput without any bug in the model code.

**24. Inference parallelism for MoE?**
Same patterns as training. Expert parallelism is essential for large-MoE inference. Vector + KV cache must fit; expert weights must be reachable.

> **Saying it out loud.** Inference has the same sharding options and a different bottleneck. Training is compute-bound and batches are huge, so all-to-all overhead amortizes; serving is memory-bandwidth-bound with small batches, so the fixed cost of shipping tokens around is much harder to hide. That's why MoE serving really wants large batches — with enough concurrent requests, every expert gets enough tokens to be worth activating, whereas a batch of one wakes up two experts per layer and wastes everything else. The other constraint is simply that all expert weights must be resident and reachable somewhere in the cluster, which is why a 671B-total model needs a multi-GPU deployment even though only 37B is active.

---

## E. Production MoE models

**25. What was Switch Transformer's contribution?**
Google 2021. First major MoE LLM. Simple top-1 routing. Established that MoE works at scale and trains stably with aux loss.

> **Saying it out loud.** Switch Transformer's contribution was mostly proof and simplification. The MoE idea predates it by years, but Switch showed you could route to a *single* expert — which everyone had assumed would be too unstable — and still train reliably at scale, provided you added the auxiliary balancing loss, used a capacity factor, and cast the router in float32 for numerical stability. That last detail is a real one: the router's softmax is sensitive, and doing it in bfloat16 causes divergence. The headline result was a claimed 7x pretraining speedup at matched quality, which is what made everyone take MoE seriously.

**26. What was GShard's contribution?**
Google 2020. Top-2 routing + capacity factor + load balancing loss. Defined the standard MoE recipe that Mixtral and many followers use.

> **Saying it out loud.** GShard came first and it's really where the standard recipe comes from — top-2 routing, the capacity factor with token dropping, and the auxiliary load balancing loss are all GShard. It was built for a 600-billion-parameter translation model, and just as importantly it introduced the annotation-based sharding API that made this kind of parallelism expressible at all rather than something you hand-wrote. Switch gets more citations because it simplified to top-1 and framed the results better, but if someone asks where the modern MoE template originated, the honest answer is GShard.

**27. What's special about Mixtral 8x7B?**
First open-source flagship MoE. ~47B total / ~13B active. Quality near LLaMA-2 70B at much lower inference cost. Top-2 routing, 8 experts. Defined the open MoE template.

> **Saying it out loud.** Mixtral mattered because it was the first MoE with open weights that was actually good. Eight experts per FFN layer, top-2 routing, 46.7 billion parameters total and about 12.9 billion active, matching or beating Llama-2 70B while costing roughly what a 13B dense model costs to serve. The detail I'd volunteer is the parameter count, because the naive eight-times-seven arithmetic gives 56 billion and that's wrong — attention, embeddings, and normalization layers are shared across all experts and counted once, so only the FFN blocks are actually replicated. It also demonstrated the awkward part of MoE economics: you need 47 billion parameters' worth of memory to run a 13B-speed model.

**28. What's special about DeepSeek-V3?**
671B total / 37B active. Auxiliary-loss-free balancing. Many fine-grained experts + shared experts. MLA for KV cache. Open weights. Frontier-quality with ~10% the inference cost of similar dense models.

> **Saying it out loud.** DeepSeek-V3 is the model to cite for where MoE actually is now: 671 billion total parameters with only 37 billion active, so a ratio of about 18 to 1 versus Mixtral's 3.6 to 1. Three ideas do the work. Fine-grained experts, 256 routed plus shared ones per layer, for a combinatorially richer specialization space. Auxiliary-loss-free balancing via per-expert bias adjustment, which removes the gradient that was fighting the main objective. And Multi-head Latent Attention, which compresses the KV cache — important because with sparsity making the FFN cheap, the KV cache becomes your serving bottleneck. The number that made everyone pay attention is the training cost, reported around 2.8 million GPU-hours, which is a small fraction of what comparable dense models cost.

**29. Why has every major lab gone MoE?**
Better scaling laws (more params for same compute). Better inference economics (active $\ll$ total). Better expert specialization. The compute-quality Pareto frontier shifted.

> **Saying it out loud.** Because it moves the Pareto frontier, and the argument is economic rather than aesthetic. Scaling laws say quality improves with both parameters and data, but dense models tie parameter count directly to compute per token, so you're paying for capacity on every single token whether you need it or not. MoE breaks that coupling — you can grow capacity roughly an order of magnitude while holding compute per token flat. Since serving cost is what actually determines whether a model is deployable at scale, that's the whole ballgame. The counterweight, and it's the thing to name, is that you now need enough memory to hold parameters you mostly don't use, plus the engineering to shard them and the all-to-all traffic to reach them.

---

## F. Subtleties

**30. Why is training MoE less stable than dense?**
Routing decisions are non-differentiable (top-k); gradients flow through chosen experts only. Imbalance amplifies. Capacity factors create token dropping. Aux loss can interfere. Modern systems have largely solved this — DeepSeek-V3 is as stable as dense.

> **Saying it out loud.** There are four sources of instability and they compound. The `topk` is discrete, so no gradient reaches the routing decision itself — only the gate weights. Gradients reach only the selected experts, so a neglected expert stays neglected. Token dropping means some tokens silently skip the layer. And the auxiliary loss is an extra objective pulling against the main one. Each is manageable, but early MoE runs would diverge in ways dense runs didn't. The fixes are known now — router z-loss to keep logits from exploding, float32 routing, careful initialization, and DeepSeek's bias-based balancing that removes the competing gradient entirely — and the honest current answer is that a well-configured MoE trains about as reliably as a dense model.

**31. Why does sigmoid routing (DeepSeek-V3) help?**
Each expert is selected independently with its own gate. More flexibility — multiple experts can have high weight without being forced into a softmax simplex. Plays well with bias-based balancing.

> **Saying it out loud.** With softmax gating, the expert scores compete — they're forced onto a simplex summing to one, so raising one expert's weight mechanically lowers everyone else's, even when several experts are genuinely all relevant. Sigmoid scores each expert independently, so "this token needs experts 3, 17, and 40, all strongly" is directly expressible. That matters much more when you're selecting eight out of 256 than when you're selecting two out of eight. It also composes cleanly with bias-based balancing, since you can add a bias to an independent score without the softmax redistributing that change across every other expert in a way that's hard to reason about.

**32. What's expert specialization?**
Different experts learn different "skills" — some experts handle math, others code, others multilingual. Empirical observation; encouraged by routing patterns. Better with fine-grained experts (DeepSeek-MoE).

> **Saying it out loud.** Specialization is the thing MoE is supposed to buy you, and the honest answer is that it's real but messier than the marketing. Nobody assigns experts topics — it emerges from routing. When people actually inspect the patterns in coarse MoEs like Mixtral, specialization looks more syntactic than semantic: experts pick up on token types, punctuation, numbers, code delimiters, rather than clean domains like "math" versus "biology." Fine-grained architectures get closer to the intuitive story, because a small expert can afford to be narrow while a large one is forced to generalize. The tradeoff to name is that balancing pressure actively works against specialization — pushing routing toward uniform is pushing it toward less opinionated, which is exactly why DeepSeek moved balancing out of the loss.

**33. What's the relationship between MoE and ensembling?**
MoE is a learned soft ensemble: the router decides per-token which "members" (experts) contribute. Differs from ensembles in that experts are jointly trained and only $k$ are active per token.

> **Saying it out loud.** There's a family resemblance and the differences are the interesting part. A classical ensemble trains models independently and averages all of them at inference, so you pay full cost for every member and you get variance reduction from their independence. MoE trains all experts jointly as one model, so they co-adapt rather than being independent, and it runs only $k$ of them per token — so it's conditional computation, not averaging. The gain isn't variance reduction, it's capacity: an ensemble of $N$ models costs $N$ times as much to run, while an MoE of $N$ experts costs about $k/N$ of its total size. Same intuition about diverse sub-models, opposite cost structure.

**34. Why does MoE often produce more diverse outputs than dense?**
Different routing per token can specialize behavior. Dense models smooth across all skills; MoE can specialize. Empirically: MoE's outputs sometimes more varied.

> **Saying it out loud.** The intuition is that a dense model has to represent every skill in the same shared weights, so everything gets averaged toward a compromise, while an MoE can keep genuinely distinct behaviours in separate experts without them interfering. That's the same reason people say MoE handles multilingual and multi-domain training better — less destructive interference between very different data distributions. I'd hold this one loosely, though, and say so: it's more of a plausible mechanism with anecdotal support than a well-measured effect, and diversity in outputs is confounded by sampling settings and post-training. The better-supported version of the claim is the interference one, not the diversity one.

**35. What's the future of MoE?**
Open questions: even more fine-grained experts (1000+)? Hybrid attention-MoE? Better balancing without bias hack? More active experts ($k = 4+$) at scale? Field is moving fast; the answer might be different in a year.

> **Saying it out loud.** The direction is clear even if the details aren't: finer-grained experts, higher sparsity ratios, and balancing that doesn't fight the main objective. DeepSeek already went from Mixtral's 3.6-to-1 total-to-active ratio up to 18-to-1, and there's no obvious wall yet. The open problems worth naming are on the systems side rather than the modeling side — all-to-all communication is the real ceiling, and better balancing mechanisms and topology-aware routing are where the wins are. Personally I'd flag the memory-versus-compute asymmetry as the thing that shapes what comes next: MoE keeps making compute cheaper while making memory requirements worse, so techniques that let you page experts in and out, or route with locality in mind, matter more than another architectural tweak.

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
