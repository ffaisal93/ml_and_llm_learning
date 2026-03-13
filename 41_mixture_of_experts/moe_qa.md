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
- 8 experts, each 7B parameters
- Total: 8 × 7B = 56B parameters
- Active: k=2, so 2 × 7B = 14B parameters per token
- Computation: Only 14B parameters active (not 56B!)

**Efficiency:**
- Total capacity: 56B parameters
- Computation: Only 14B parameters
- 4× more parameters, but similar computation to 14B dense model

**Memory:**
- During training: Need all expert parameters (56B)
- During inference: Can load only active experts (14B)
- KV cache: Same as dense model (not affected by MoE)

**Reduction:**
- Computation: (num_experts / k)× reduction
- Example: 8 experts, k=2 → 4× reduction in computation
- But total parameters: num_experts× more

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

---

## Q4: What is load balancing? Why is it important?

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
```
L_balance = (1/num_experts) * sum(load_i)²
```

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

---

## Q6: How is MoE used in modern LLMs like GPT-4 and Mixtral?

**Answer:**

**GPT-4 (Rumored):**
- Uses MoE architecture (exact details not public)
- Multiple experts
- Top-k routing
- Enables very large model (trillions of parameters)

**Mixtral-8x7B:**
- 8 experts, each 7B parameters
- Total: 56B parameters
- Top-2 routing (k=2)
- Active: 14B parameters per token

**Architecture:**
- Replace standard FFN with MoE-FFN
- Each transformer block has MoE layer
- Router decides which experts per token

**Efficiency:**
- Total capacity: 56B parameters
- Computation: Only 14B parameters active
- Similar computation to 14B dense model
- But 4× more capacity

**Quality:**
- Achieves quality of larger dense models
- With computation of smaller models
- Best of both worlds

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

---

## Summary

Mixture of Experts enables training models with trillions of parameters while keeping computation efficient. By activating only a subset of experts for each input, MoE achieves the capacity of very large models with the computation of much smaller models. Key components include multiple expert networks, a routing mechanism for expert selection, and load balancing to ensure all experts are utilized. Modern models like GPT-4 and Mixtral-8x7B use MoE to achieve unprecedented scale and efficiency.

