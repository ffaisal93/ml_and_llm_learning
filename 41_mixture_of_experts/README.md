# Topic 41: Mixture of Experts (MoE)

> 🔥 **For interviews, read these first:**
> - **`MOE_DEEP_DIVE.md`** — frontier-lab interview deep dive: top-k routing, load balancing loss derivation, capacity factor / token dropping, expert parallelism + all-to-all, Switch/Mixtral/DeepSeek-V3, auxiliary-loss-free balancing, fine-grained vs coarse experts.
> - **`INTERVIEW_GRILL.md`** — 40 active-recall questions.

## What You'll Learn

This topic teaches you Mixture of Experts comprehensively:
- What is MoE and how it works
- Sparse activation and routing
- Training procedures
- Gating mechanisms
- Memory and computation efficiency
- Real-world applications

## Why We Need This

### Interview Importance
- **Hot topic**: MoE is used in modern LLMs (GPT-4, Mixtral)
- **Efficiency**: Enables scaling to larger models
- **Understanding**: Key architecture for large-scale models

### Real-World Application
- **Large models**: GPT-4, Mixtral-8x7B use MoE
- **Efficiency**: Activate only subset of parameters
- **Scaling**: Train models with trillions of parameters

## Industry Use Cases

### 1. **Large Language Models**
**Use Case**: GPT-4, Mixtral-8x7B
- Scale to trillions of parameters
- Activate only subset during inference
- Efficient training and serving

### 2. **Efficient Inference**
**Use Case**: Production serving
- Reduce computation per token
- Lower latency
- Cost-effective serving

### 3. **Specialized Models**
**Use Case**: Domain-specific experts
- Different experts for different tasks
- Better specialization
- Improved quality

## Core Intuition

Mixture of Experts tries to scale parameter count without paying full dense-compute cost on every token.

The key idea is:
- have many experts available
- route each token to only a few of them

That means the model can be large in total capacity while staying sparse in computation.

### Experts

Experts are separate subnetworks, often feed-forward blocks.

### Router

The router decides which experts a token should use.

That means MoE performance depends not only on expert quality, but also on routing quality.

## Technical Details Interviewers Often Want

### Capacity vs Compute

This is the central MoE trade-off.

MoE increases parameter capacity, but each token only uses a subset of that capacity.

### Load Balancing Matters

If the router sends too many tokens to the same experts:
- some experts overload
- others under-train
- efficiency and quality both suffer

That is why load-balancing losses matter in MoE training.

### Sparse Activation Does Not Mean Free Scaling

MoE improves compute efficiency, but it introduces:
- routing complexity
- communication overhead
- expert imbalance risk

## Common Failure Modes

- explaining MoE as just "more parameters" without sparse routing
- ignoring load balancing
- assuming MoE always lowers latency in practice
- forgetting communication overhead in distributed settings

## Edge Cases and Follow-Up Questions

1. Why does MoE increase capacity without full dense computation?
2. Why is router quality so important?
3. Why do load-balancing losses matter?
4. Why can MoE introduce systems complexity even if math looks simple?
5. Why is sparse activation not the same as free scaling?

## What to Practice Saying Out Loud

1. The difference between total parameters and active parameters
2. Why routing quality is central to MoE
3. Why MoE is a capacity-compute trade-off, not just a bigger model trick

## Theory

### What is Mixture of Experts?

Mixture of Experts is an architecture where multiple expert networks are trained, but only a subset is activated for each input. A gating network (router) decides which experts to use, enabling models with many parameters while keeping computation efficient.

### Key Concepts

**Experts:**
- Multiple feed-forward networks (experts)
- Each expert is a complete neural network
- Typically 8-128 experts in modern models

**Router/Gating:**
- Decides which experts to activate
- Outputs probability distribution over experts
- Top-k routing: activate k experts with highest scores

**Sparse Activation:**
- Only k experts activated per token
- Most experts remain inactive
- Reduces computation significantly

## Industry-Standard Boilerplate Code

See detailed files for complete implementations:
- `moe_theory.md`: Complete theoretical foundation
- `moe_code.py`: Full implementation
- `moe_training.py`: Training procedures
- `moe_qa.md`: Comprehensive interview Q&A

## Exercises

1. Implement MoE layer
2. Implement routing mechanism
3. Train MoE model
4. Compare MoE vs dense models

## Next Steps

- Review transformer architecture
- Compare with dense models
- Explore state space models
