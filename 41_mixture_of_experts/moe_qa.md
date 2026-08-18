# Mixture of Experts: Interview Q&A

## Q1: What is Mixture of Experts? How does it work?

**Answer:**

**Mixture of Experts (MoE):**
- Architecture with multiple expert networks
- Router decides which experts to activate
- Only subset of experts process each input
- Enables models with trillions of parameters

**How It Works:**

**1. Multiple Experts:**
- 8-128 feed-forward networks (experts)
- Each expert is independent
- All experts have same architecture

**2. Router/Gating:**
- Takes input, outputs expert scores
- Computes probability distribution over experts
- Selects top-k experts with highest scores

**3. Sparse Activation:**
- Only k experts activated per token
- Typically k=1 or k=2
- Most experts remain inactive

**4. Weighted Combination:**
- Process input through selected experts
- Weighted combination of expert outputs
- Weights from router probabilities

**Mathematical Formulation:**
```
scores = Router(x)  # Expert scores
probs = softmax(scores)  # Probabilities
top_k_indices = topk(probs, k)  # Select k experts
output = sum(probs[i] * Expert[i](x) for i in top_k_indices)
```

**Key Insight:**
- Total parameters: num_experts × params_per_expert (large)
- Active parameters: k × params_per_expert (small)
- Enables scaling without proportional compute increase

> **Saying it out loud.** Mixture of experts is the idea that a model doesn't need to use all of itself on every input. Instead of one big feed-forward block that every token passes through, you keep eight or sixty-four smaller ones and a little router that picks two per token. Think of a hospital: you don't run every patient past every specialist — reception looks at the case and sends them to two of them. So the model's total knowledge scales with the number of experts while the work per token stays roughly flat. The tradeoff to name up front is that you still have to hold every expert in memory even though you only compute with two, so MoE buys you compute efficiency, not memory efficiency.

---

## Q2: How does MoE reduce computation compared to dense models?

**Answer:**

**Dense Model:**
- All parameters used for every input
- Computation: O(d_model²) per token
- Example: 7B parameters, all active

**MoE Model:**
- Total parameters: num_experts × params_per_expert
- Active parameters: k × params_per_expert
- Computation: O(k × d_model²) per token

**Example: Mixtral-8x7B**
- 8 experts in each FFN layer; only the FFN blocks are replicated
- Total: **46.7B** parameters — *not* 8 × 7B = 56B, because attention layers, embeddings and norms are **shared** across all experts and counted once
- Active: k=2, so **12.9B** parameters per token (again less than 2 × 7B, for the same sharing reason)
- Computation: Only ~12.9B parameters active (not 46.7B!)

**Efficiency:**
- Total capacity: 46.7B parameters
- Computation: Only ~12.9B parameters
- ~3.6× more parameters, but similar computation to a ~13B dense model

**Memory:**
- During training: Need all expert parameters (46.7B)
- During inference: Can load only active experts (~12.9B)
- KV cache: Same as dense model (not affected by MoE)

**Reduction:**
- Computation: (num_experts / k)× reduction
- Example: 8 experts, k=2 → 4× reduction in computation
- But total parameters: num_experts× more

> **Saying it out loud.** The saving is real but it's narrower than people assume. Compute per token scales with *active* parameters, so a top-2-of-8 model does roughly the work of a model a quarter the size — Mixtral is 46.7 billion total and about 12.9 billion active, and it runs at roughly 13B speed. Note that the arithmetic isn't naive multiplication: attention layers, embeddings, and norms are shared across all experts and counted once, which is why 8 times 7 doesn't give you 56. And the thing that doesn't shrink at all is memory, because you must have every expert resident to serve any token. So the honest framing is that MoE trades memory for compute, and it only pays off when you're memory-rich and compute-bound.

---

## Q3: What is the routing mechanism? How does top-k routing work?

**Answer:**

**Routing Mechanism:**
- Router (gating network) decides which experts to use
- Takes input, outputs scores for each expert
- Selects experts based on scores

**Top-k Routing Algorithm:**

**1. Compute Scores:**
```python
scores = Router(x)  # (num_experts,) - logits
probs = softmax(scores)  # Probabilities
```

**2. Select Top-k:**
```python
top_k_probs, top_k_indices = torch.topk(probs, k)
# Select k experts with highest probabilities
```

**3. Renormalize:**
```python
top_k_probs = top_k_probs / top_k_probs.sum()
# Renormalize so probabilities sum to 1
```

**4. Weighted Combination:**
```python
output = 0
for i, expert_idx in enumerate(top_k_indices):
    expert_output = Expert[expert_idx](x)
    output += top_k_probs[i] * expert_output
```

**Example:**
- 8 experts, k=2
- Router scores: [0.1, 0.3, 0.05, 0.2, 0.15, 0.1, 0.05, 0.05]
- Top-2: experts 1 and 3 (scores 0.3 and 0.2)
- Renormalize: [0.6, 0.4] (for experts 1 and 3)
- Output: 0.6 × Expert1(x) + 0.4 × Expert3(x)

**Why Top-k?**
- Hard routing: Only use k experts (efficient)
- Soft routing: Use all experts with weights (less efficient)
- Top-k balances efficiency and flexibility

> **Saying it out loud.** Routing is a tiny linear layer that maps each token to one score per expert, and you take the top few. Two details matter. First, the routing decision is per token, not per sequence, so different words in the same sentence go to different experts — and the router runs at every MoE layer, so a token's path through the network is a different pair of experts at each depth. Second, you renormalize the selected probabilities so they sum to one, otherwise the block's output magnitude varies with how confident the router happened to be. The problem to name is that `topk` is a discrete operation, so there's no gradient through *which* experts were chosen — the router only learns through the weights it assigned to the ones it picked.

---

## Q4: What is load balancing? Why is it important?

*In plain language:* left alone, the router develops favourites — a couple of experts get picked constantly and the rest are dead weight you're still paying to store. Load balancing is an extra term in the loss whose only job is to punish that. The tricky part, spelled out below, is that "how many tokens went to expert 3" is a counting operation with no gradient, so the loss has to be built out of the router's soft probabilities if it's going to teach the router anything.

**Answer:**

**Load Balancing Problem:**
- Without balancing, router might always select same experts
- Some experts never used (waste of parameters)
- Others overloaded (bottleneck)
- Expert collapse: Only few experts ever used

**Load Balancing Solution:**
- Encourage uniform expert usage
- Ensure all experts are utilized
- Prevent expert collapse

**Load Balancing Loss:**

A loss written purely in terms of hard token counts has **no gradient with respect to the router** — `topk` is a discrete operation, so counts are piecewise-constant and differentiate to zero. The Switch Transformer / GShard auxiliary loss avoids this by pairing the (constant) hard fraction with the (differentiable) mean router probability:

```
f_i = fraction of tokens whose top-k set contains expert i   # hard count, no gradient
P_i = mean over tokens of softmax(router_logits)_i           # soft, differentiable

L_balance = num_experts * sum_i (f_i * P_i)
```

The gradient flows only through `P_i`, using `f_i` as a per-expert weight: an overloaded expert has a large `f_i`, so the loss pushes its router logits *down*. Minimized when `f_i ≈ P_i ≈ 1/num_experts`, where the loss equals 1 regardless of expert count (that is what the leading `num_experts` factor is for).

Where load_i is fraction of tokens routed to expert i.

**Goal:**
- Minimize variance of expert usage
- Distribute tokens evenly across experts
- All experts should be used roughly equally

**Why Important:**
- Without balancing: Experts 0-2 always used, 3-7 never used
- With balancing: All experts used roughly equally
- Better parameter utilization
- Prevents expert collapse

**Training:**
- Add load balancing loss to total loss
- L_total = L_main + α * L_balance
- Encourages router to distribute tokens

> **Saying it out loud.** Without a balancing term, MoE collapses. It's a rich-get-richer loop: an expert that gets slightly more tokens early trains slightly faster, so the router likes it more, so it gets more tokens — and within a few thousand steps you have two experts doing everything and six dead ones you're still paying to store. The fix is an auxiliary loss, and the subtle part is making it differentiable. Counting how many tokens went to expert three gives you a number with zero gradient, because `topk` is discrete. So Switch Transformer multiplies that hard count by the *mean softmax probability* the router assigned to that expert — the count is a constant weight and the gradient flows through the probability, pushing an overloaded expert's logits down. Weight it around 0.01: too low and you collapse, too high and the router balances at the cost of routing sensibly.

---

## Q5: Compare MoE with dense models. What are the trade-offs?

**Answer:**

**Comparison:**

| Aspect | Dense Model | MoE Model |
|--------|-------------|-----------|
| **Total Parameters** | P | num_experts × P |
| **Active Parameters** | P (all) | k × P |
| **Computation** | O(P) | O(k × P) |
| **Memory (Training)** | P | num_experts × P |
| **Memory (Inference)** | P | k × P (can load only active) |
| **Quality** | Baseline | Similar (slight trade-off) |
| **Training** | Simple | Complex (need balancing) |

**Trade-offs:**

**MoE Advantages:**
- Can have many more parameters (trillions)
- Only use subset per input (efficient)
- Experts can specialize
- Better for diverse inputs

**MoE Disadvantages:**
- More complex training (load balancing)
- Higher memory during training
- Routing overhead (small)
- Slight quality trade-off (often negligible)

**When to Use:**
- **Dense**: Small-medium models, simplicity
- **MoE**: Large models, need efficiency, diverse inputs

> **Saying it out loud.** The comparison comes down to what resource you're short of. A dense model uses every parameter on every token, which is simple, trains stably, and is easy to serve. An MoE gets you far more total parameters at the same compute per token, which is why every frontier lab uses one. What you pay is threefold: memory, since every expert has to be resident; engineering, since experts get sharded across devices and now every forward pass involves an all-to-all communication step; and training stability, since you've added a router that can collapse. The rule of thumb worth stating is that MoE wins when you're memory-rich and compute-bound — a big training cluster or a well-provisioned inference fleet — and loses on a single GPU where you simply can't fit the parameters you never use.

---

## Q6: How is MoE used in modern LLMs like GPT-4 and Mixtral?

**Answer:**

**GPT-4 (Rumored):**
- Uses MoE architecture (exact details not public)
- Multiple experts
- Top-k routing
- Enables very large model (trillions of parameters)

**Mixtral-8x7B:**
- 8 experts per FFN layer (the name refers to 8 experts sized like a 7B model's FFN, not 8 independent 7B models)
- Total: **46.7B** parameters — attention, embeddings and norms are shared across experts, so the naive 8 × 7B = 56B is wrong
- Top-2 routing (k=2)
- Active: **12.9B** parameters per token

**Architecture:**
- Replace standard FFN with MoE-FFN
- Each transformer block has MoE layer
- Router decides which experts per token

**Efficiency:**
- Total capacity: 46.7B parameters
- Computation: Only ~12.9B parameters active
- Similar computation to a ~13B dense model
- But ~3.6× more capacity

**Quality:**
- Achieves quality of larger dense models
- With computation of smaller models
- Best of both worlds

> **Saying it out loud.** Mixtral is the model to use as the concrete example. Eight experts per feed-forward layer, top-2 routing, 46.7 billion parameters total, about 12.9 billion active per token. The number people get wrong is that total — the naive eight times seven gives 56 billion, but attention, embeddings, and layer norms are shared across all experts and counted once, so only the FFN blocks are actually replicated. That correction is worth volunteering, because it shows you understand where MoE is actually applied in the architecture. GPT-4 is widely believed to be an MoE but nothing is confirmed, so I'd flag it as rumor rather than state it. The headline result is that Mixtral matches or beats a 70B dense model at roughly 13B inference cost.

---

## Q7: What are the challenges in training MoE models?

**Answer:**

**1. Expert Collapse:**
- Router might always select same experts
- Other experts never trained
- Solution: Load balancing loss

**2. Gradient Flow:**
- Only active experts receive gradients
- Inactive experts don't learn
- Solution: Expert sampling, auxiliary losses

**3. Routing Instability:**
- Router decisions can be unstable
- Experts might not converge
- Solution: Temperature annealing, regularization

**4. Load Imbalance:**
- Uneven expert usage
- Some experts overloaded
- Solution: Load balancing loss, expert capacity limits

**5. Memory:**
- Need to store all expert parameters
- Higher memory than dense
- Solution: Expert sharding, gradient checkpointing

**Training Techniques:**
- Load balancing loss
- Expert sampling (random experts sometimes)
- Temperature annealing (soft → hard routing)
- Gradient clipping
- Careful initialization

> **Saying it out loud.** Almost every MoE training problem is the router. Collapse is the headline one — the rich-get-richer feedback loop that leaves most of your experts untrained — and it's why the auxiliary loss exists. Related is that gradients only reach the experts that were selected, so a dead expert stays dead and there's no automatic recovery. Then there's capacity: in a real distributed implementation each expert has a fixed buffer per batch, so if too many tokens route to one expert, the overflow gets *dropped* and skips the layer entirely, which is a silent quality loss you'll only see if you log it. Add router z-loss to stop the logits from drifting large, keep the balance coefficient near 0.01, and log expert utilization every step — that last habit is what separates people who've trained one from people who've read about it.

---

## Summary

Mixture of Experts enables training models with trillions of parameters while keeping computation efficient. By activating only a subset of experts for each input, MoE achieves the capacity of very large models with the computation of much smaller models. Key components include multiple expert networks, a routing mechanism for expert selection, and load balancing to ensure all experts are utilized. Modern models like GPT-4 and Mixtral-8x7B (46.7B total / 12.9B active) use MoE to achieve unprecedented scale and efficiency.

