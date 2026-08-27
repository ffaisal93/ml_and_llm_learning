# RL Alignment Interview Q&A: Detailed Answers

## Q1: Explain the RLHF (Reinforcement Learning from Human Feedback) pipeline in detail.

**Answer:**

RLHF is a three-stage process used to align language models with human preferences. Here's the detailed pipeline:

**Stage 1: Supervised Fine-Tuning (SFT)**
- **Purpose**: Create a baseline model that can follow instructions
- **Data**: Human-written demonstrations (prompt-response pairs)
- **Training**: Standard supervised learning (cross-entropy loss)
- **Result**: Model that can generate reasonable responses but may not align with human preferences

**Stage 2: Reward Model Training**
- **Purpose**: Learn a function that scores how good a response is
- **Data**: Human preference comparisons (chosen response vs rejected response)
- **Training**: Binary classification - learn to rank chosen > rejected
- **Loss**: Binary cross-entropy on preference pairs
- **Result**: Reward model r(x, y) that scores response quality

**Mathematical Formulation:**
```
P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

Where:
- y_w: Winning (chosen) response
- y_l: Losing (rejected) response
- σ: Sigmoid function
```

**Stage 3: RL Optimization (PPO)**
- **Purpose**: Optimize policy to maximize reward while staying close to reference
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Objective**: Maximize E[r(x, y)] - β * KL(π_θ || π_ref)
- **Result**: Aligned model that generates preferred responses

**Why this works:**
- SFT gives model capability
- Reward model captures human preferences
- RL optimization aligns model with preferences

**Challenges:**
- Need large amounts of human feedback
- Reward model may have biases
- RL optimization can be unstable
- Cost: Expensive to collect human preferences

> **Saying it out loud.** RLHF is three stages and each adds a different signal. SFT teaches the model the shape of being an assistant — you're just doing cross-entropy on human-written answers. Then you train a reward model on pairwise comparisons, so instead of "here's the right answer" it learns "this one beats that one." Then you run RL, usually PPO, to push the policy toward high reward while a KL penalty keeps it from wandering off the SFT model. The one-line summary: SFT teaches format, the reward model captures taste, and RL is what actually moves the distribution — and the KL term is what stops the whole thing degenerating into reward hacking.

---

## Q2: How does DPO differ from RLHF? When would you use each?

**Answer:**

**DPO (Direct Preference Optimization):**

**Key Difference:**
- **RLHF**: Needs separate reward model, uses RL (PPO) to optimize
- **DPO**: No reward model, directly optimizes policy on preferences

**DPO Mathematical Formulation:**
```
L_DPO = -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))

Where:
- y_w: Chosen response
- y_l: Rejected response
- π_θ: Current policy
- π_ref: Reference policy (frozen)
- β: Temperature parameter
```

**How DPO Works:**
1. Uses reference model instead of reward model
2. Directly optimizes policy to prefer chosen over rejected
3. KL penalty prevents deviation from reference
4. No RL needed - just supervised learning on preferences

**Comparison:**

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Reward Model** | Required | Not needed |
| **Optimization** | RL (PPO) | Supervised learning |
| **Complexity** | High (3 stages) | Lower (2 stages) |
| **Flexibility** | Can use any reward | Limited to preferences |
| **Stability** | Can be unstable | More stable |
| **Data Needs** | Preference + demonstrations | Just preferences |

**When to Use RLHF:**
- Need flexible reward shaping
- Have complex reward structure
- Want to iterate on reward model
- Have resources for complex pipeline

**When to Use DPO:**
- Want simpler pipeline
- Have preference data but no demonstrations
- Need faster training
- Want more stable optimization

**Trade-off:**
- DPO is simpler but less flexible
- RLHF is more complex but more powerful

> **Saying it out loud.** The core difference is that DPO deletes the reward model and the RL loop. Rafailov showed that the RLHF objective has a closed-form optimum, so you can rewrite the reward as the log-ratio between your policy and the frozen reference, and when you plug that into the preference model the intractable constant cancels — what's left is an ordinary supervised loss on preference pairs. So DPO is two models in memory instead of four and no rollouts at all. I'd use DPO when I have good preference pairs and limited compute, and PPO or GRPO when the reward is verifiable, I can afford to generate, or I need to hand-shape the reward.

> **Contested, not settled.** Whether DPO genuinely matches online RLHF is an open argument and shouldn't be stated as fact. The DPO paper and most open recipes report rough parity on chat benchmarks, while Xu et al. 2024 ("Is DPO Superior to PPO for LLM Alignment?") and Tajwar et al. 2024 report a well-tuned PPO ahead on harder, more out-of-distribution settings. Safest framing: DPO is competitive and much cheaper on static preference data; on-policy methods still lead where the policy has to move far from that data — which is exactly why iterative/online DPO exists.

---

## Q3: Explain PPO (Proximal Policy Optimization) in detail. Why is it used in RLHF?

**Answer:**

**What is PPO?**
PPO is a policy gradient algorithm that prevents large policy updates by clipping the objective function.

**The Four Models in PPO/RLHF:**

**1. Policy Model (π_θ):**
- Generates responses/actions
- Outputs probability distribution: π_θ(a|s)
- Being optimized during training
- Used for: generation, policy gradient computation

**2. Critic Model (V_φ):**
- Estimates state value: V_φ(s) = E[R | s]
- Predicts expected future return
- Used for: advantage computation (A = Q - V), baseline for variance reduction
- Trained with: value loss L^VF = (V_φ(s) - R)^2

**3. Reference Model (π_ref):**
- Frozen copy of policy before RL training
- Typically the SFT (Supervised Fine-Tuned) model
- Used for: KL penalty computation, importance sampling ratio
- Mathematical role: KL(π_θ || π_ref) = E[log(π_θ/π_ref)]

**4. Reward Model (r_ψ):**
- Scores responses: r_ψ(x, y)
- Trained on human preferences before RL
- Used for: computing rewards during RL training
- Typically frozen during RL (can be updated)

**Mathematical Formulation:**

**Standard Policy Gradient:**
```
L_PG = E[r(θ) * A]

Where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s) (importance sampling ratio)
- A: Advantage estimate
```

**Problem with Standard PG:**
- Large updates can destabilize training
- Policy can change too quickly
- Can lead to poor performance

**PPO Solution - Clipped Objective:**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

Where:
- ε: Clipping parameter (typically 0.1-0.3)
- clip(r(θ), 1-ε, 1+ε): Clips ratio to [1-ε, 1+ε]
- min: Takes pessimistic estimate
```

**Why Clipping Works:**
1. **Prevents large updates**: Ratio is clipped, so updates are bounded
2. **Pessimistic**: Taking minimum prevents over-optimization
3. **Stable**: Policy changes gradually
4. **Sample efficient**: Can use same data multiple times

**PPO Algorithm:**
```
1. Collect trajectories with current policy
2. Compute advantages A(s,a)
3. For K epochs:
   a. Compute r(θ) = π_θ(a|s) / π_θ_old(a|s)
   b. Compute clipped objective
   c. Update policy
4. Update old policy: π_θ_old = π_θ
```

**Why PPO in RLHF:**
1. **Stability**: Language models are sensitive - need stable updates
2. **Sample efficiency**: Human feedback is expensive - reuse data
3. **KL constraint**: Keeps policy close to reference (prevents mode collapse)
4. **Proven**: Works well in practice (ChatGPT, Claude)

**PPO Loss Components:**
```
L_PPO = L^CLIP + c_v * L^VF + β * KL(π_θ || π_ref)

Where:
- L^CLIP: Clipped policy loss (uses Policy Model π_θ)
- L^VF: Value function loss (uses Critic Model V_φ)
- KL: KL penalty (uses Reference Model π_ref)
- Rewards: From Reward Model r_ψ
- c_v, β: Coefficients
```

**How All Four Models Work Together:**

**Training Loop:**
1. **Generate**: Policy Model π_θ generates responses
2. **Score**: Reward Model r_ψ scores responses → rewards
3. **Evaluate**: Critic Model V_φ estimates values → V(s)
4. **Compare**: Reference Model π_ref provides logprobs → KL penalty
5. **Compute**: Advantages A = returns - V(s)
6. **Update**: Policy π_θ and Critic V_φ (Reference π_ref and Reward r_ψ frozen)

**Mathematical Flow:**
```
responses = π_θ.generate(prompts)
rewards = r_ψ(prompts, responses)
values = V_φ(prompts)
policy_logprobs = log π_θ(responses | prompts)
ref_logprobs = log π_ref(responses | prompts)

advantages = returns - values
ratio = exp(policy_logprobs - ref_logprobs)

L = min(ratio*A, clip(ratio)*A) + c_v*(V-R)² + β*KL(π_θ||π_ref)
```

See [`ppo_models_detailed.md`](ppo_models_detailed.md) for complete mathematical details!


> **Saying it out loud.** PPO is a policy-gradient method with a safety belt. The problem it solves is that you're learning from your own samples, so one bad batch can shove the policy somewhere it can't recover from — PPO stops that by computing the ratio of new to old probability for each token and clipping it to roughly 0.8 to 1.2, so no single update can move things too far. That's what lets you reuse a batch of expensive rollouts for several gradient steps instead of one. The thing to have ready is the four-models answer: policy, frozen reference for KL, reward model, and critic — which for a 70B policy is about a terabyte of weights, and is exactly the cost GRPO and DPO were invented to cut.

---

## Q4: What is GRPO (Group Relative Policy Optimization)? When is it useful?

**Answer:**

> **Correction (2026-08-18).** The original text of this answer described GRPO as optimizing across *demographic or user groups* (age groups, regions, skill levels). That is wrong and would be a costly thing to say in an interview. In GRPO, a "group" is a set of $G$ sampled completions for the **same prompt**. The answer below is the corrected version.

**What is GRPO?**
GRPO (Shao et al. 2024, introduced in DeepSeekMath and popularized by DeepSeek-R1) is PPO with the **value network removed**. PPO needs a learned critic to estimate the baseline "how good did we expect this to be?" GRPO estimates that baseline empirically instead: for each prompt, sample a *group* of $G$ completions from the current policy, score them all, and use the group's own mean reward as the baseline.

**Mathematical Formulation:**

For each prompt $x$, sample $G$ completions $y_1, \dots, y_G \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid x)$ and score them to get $r_1, \dots, r_G$. The advantage of completion $i$ is the reward **standardized within its own group**:

$$
\hat A_i = \frac{r_i - \mu_{\mathrm{group}}}{\sigma_{\mathrm{group}}},
\qquad
\mu_{\mathrm{group}} = \frac{1}{G}\sum_{j=1}^{G} r_j,
\qquad
\sigma_{\mathrm{group}} = \mathrm{std}(r_1, \dots, r_G)
$$

This single scalar $\hat A_i$ is shared by every token of completion $i$. The rest is standard PPO:

$$
\mathcal{L}_{\mathrm{GRPO}} = -\mathbb{E}\!\left[\frac{1}{|y_i|}\sum_t \min\!\big(\rho_t \hat A_i,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_i\big)\right] + \beta\, \mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})
$$

where $\rho_t = \pi_\theta(y_t \mid x, y_{<t}) / \pi_{\theta_{\mathrm{old}}}(y_t \mid x, y_{<t})$ is the usual importance ratio.

**Why GRPO?**
- **No critic**: removes a full-size value network from memory and from the list of things that can destabilize training.
- **Better baseline for LLMs**: the critic is hard to fit when there is one sparse scalar reward per long response; a measured group mean is a low-variance, unbiased-enough substitute.
- **Natural fit for sampling-heavy pipelines**: if you are already sampling many candidates per prompt (math, code), the group comes for free.
- **Prevents over-optimization**: the KL anchor to $\pi_{\mathrm{ref}}$ is retained, exactly as in PPO-RLHF.

**Use Cases:**
1. **Verifiable-reward tasks**: math (exact-match on the final answer), code (unit tests pass), format compliance (valid JSON, correct tool call).
2. **Reasoning RL at scale**: DeepSeek-R1, Qwen QwQ and similar systems train long chain-of-thought this way.
3. **Any setting where sampling $G$ candidates is cheap and grading them is cheap and deterministic.**
4. **Memory-constrained RLHF**: dropping the critic is a large, immediate saving.

**Example:**
- Prompt: "What is the remainder when $7^{100}$ is divided by 13?"
- Sample $G = 8$ completions from the current policy.
- Grade each against the gold answer: rewards $= [1, 0, 1, 0, 0, 1, 0, 0]$.
- Group mean $= 0.375$. So the three correct completions get positive advantage and the five wrong ones get negative advantage — all measured against *how hard this particular prompt turned out to be*, with no critic involved.
- Degenerate case: if all 8 were correct (or all 8 wrong), $\sigma_{\mathrm{group}} = 0$, the advantage is undefined/zero, and the prompt contributes no learning signal. DAPO's "dynamic sampling" trick exists to drop exactly these prompts.

**How it differs from PPO:**
- **PPO**: baseline comes from a *learned* value network $V_\phi(s)$; four models in memory (policy, reference, reward model, critic).
- **GRPO**: baseline comes from the *measured mean of $G$ sibling completions of the same prompt*; three models in memory (policy, reference, reward model or verifier).
- **PPO**: one rollout per prompt is enough; **GRPO**: needs $G$ rollouts per prompt (typically 8–64), so it trades critic compute for generation compute.
- **PPO**: per-token advantages via GAE; **GRPO**: one advantage per completion, broadcast to all its tokens (cruder credit assignment).

**Implementation:**
```python
# For ONE prompt: G completions sampled from the current policy
completions = [policy.sample(prompt) for _ in range(G)]
rewards = torch.tensor([verifier(prompt, c) for c in completions])   # [G]

# Group-relative advantage: standardize WITHIN this prompt's group
advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)     # [G]

# Same scalar advantage for every token of a given completion
ratio = torch.exp(policy_logprobs - old_logprobs)                    # [G, L]
surr1 = ratio * advantages.unsqueeze(-1)
surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantages.unsqueeze(-1)
loss = -torch.min(surr1, surr2).mean() + beta * kl_to_reference
```


> **Saying it out loud.** GRPO is PPO with the value network deleted. PPO needs a critic to answer "was this better than expected?", and for language that critic is miserable to train — one sparse reward at the end of a thousand-token response, and a target that keeps moving as the policy improves. GRPO answers the same question by measuring instead of predicting: sample $G$ completions for the *same prompt*, grade them all, and each one's advantage is its reward minus the group mean over the group's standard deviation. Be precise about "group" — it means a group of completions for one prompt, never a group of users. The tradeoff is clean: you drop a whole model and its instability, and you pay in generation, which is why GRPO wins exactly where grading is cheap and deterministic, like math and code.
---

## Q5: What are the main challenges in RL alignment? How do you address them?

**Answer:**

**Challenge 1: Reward Hacking**
- **Problem**: Model finds ways to maximize reward that don't align with intent
- **Example**: Model generates "I can't answer" to avoid negative reward
- **Solution**: 
  - Careful reward design
  - Multiple reward signals
  - Human evaluation
  - Regularization (KL penalty)

**Challenge 2: Distribution Shift**
- **Problem**: Policy changes, but reward model trained on old distribution
- **Solution**:
  - Retrain reward model periodically
  - Use on-policy data
  - Regularization to prevent large shifts

**Challenge 3: Mode Collapse**
- **Problem**: Policy collapses to single response pattern
- **Solution**:
  - KL penalty (keeps policy diverse)
  - Entropy bonus
  - Diverse training data

**Challenge 4: Instability**
- **Problem**: Training can be unstable, performance can degrade
- **Solution**:
  - PPO clipping (prevents large updates)
  - Gradient clipping
  - Learning rate scheduling
  - Checkpointing and rollback

**Challenge 5: Human Feedback Quality**
- **Problem**: Inconsistent or biased human feedback
- **Solution**:
  - Multiple annotators
  - Quality control
  - Bias detection
  - Diverse annotator pool

**Challenge 6: Scalability**
- **Problem**: Need large amounts of human feedback
- **Solution**:
  - Active learning (prioritize important examples)
  - Synthetic data generation
  - Transfer learning
  - Few-shot learning

**Challenge 7: Evaluation**
- **Problem**: Hard to measure alignment
- **Solution**:
  - Multiple metrics (helpfulness, harmlessness, honesty)
  - Human evaluation
  - Red teaming
  - Real-world testing


> **Saying it out loud.** If I had to name the hard parts in order, it's reward hacking, distribution shift, and evaluation. Reward hacking is the model getting better at the score than at the job, because the reward model is an approximation and optimisation finds approximation errors. Distribution shift is the same problem from the other side — as the policy improves it leaves the region where the reward model was trained, so the scores stop meaning anything, which is why people refresh the reward model or re-collect preferences. And evaluation is genuinely unsolved: loss curves tell you nothing, so you're stuck with expensive human comparison as ground truth and cheap LLM judges as a proxy. The unifying answer to most of these is the KL anchor plus early stopping, because the true-quality curve peaks well before the reward curve does.
---

## Q6: How do you prevent reward hacking in RLHF?

**Answer:**

**What is Reward Hacking?**
Model finds unintended ways to maximize reward that don't align with human intent.

**Examples:**
- Always says "I can't answer" to avoid negative reward
- Generates very long responses (more tokens = higher reward)
- Repeats high-reward phrases
- Exploits reward model biases

**Prevention Strategies:**

**1. Careful Reward Design**
- Multiple reward signals (not just one)
- Penalize obvious hacks (length, repetition)
- Reward diversity
- Use human evaluation as ground truth

**2. Regularization**
- **KL Penalty**: Prevents policy from deviating too much
  ```
  L = E[r(θ)A] - β * KL(π_θ || π_ref)
  ```
- Keeps policy reasonable
- Prevents extreme behaviors

**3. Reward Model Robustness**
- Train on diverse data
- Detect and remove biases
- Regular updates
- Multiple reward models (ensemble)

**4. Monitoring**
- Track reward distribution
- Detect anomalies (sudden spikes)
- Monitor response patterns
- Human spot checks

**5. Constrained Optimization**
- Hard constraints (max length, no repetition)
- Soft constraints (penalties)
- Multi-objective optimization

**6. Iterative Refinement**
- Start with simple reward
- Identify hacks
- Refine reward
- Repeat

**Example Implementation:**
```python
def robust_reward(response, base_reward):
    # Base reward from reward model
    reward = base_reward
    
    # Penalize hacks
    if is_too_long(response):
        reward -= 0.1
    if has_repetition(response):
        reward -= 0.1
    if is_evasive(response):
        reward -= 0.2
    
    # Encourage diversity
    if is_diverse(response):
        reward += 0.05
    
    return reward
```


> **Saying it out loud.** You can't eliminate reward hacking, only bound it, and the KL anchor is the main way you bound it. Beyond that it's layered defence: shape the reward to kill the hacks you know about (length, repetition, evasive refusals), use an ensemble of reward models so there's no single weakness to exploit, and grade with a held-out reward model or humans so you can see the gap open up. The detection signal is the important one to say out loud — if reward-model score climbs while human win-rate flattens or drops, you're hacking, and that's Goodhart's law. In practice the cheapest effective mitigation is early stopping on a KL budget, because the true-quality peak comes well before the reward stops improving.
---

## Q7: Explain the KL penalty in RLHF. Why is it important?

**Answer:**

**What is KL Penalty?**
KL (Kullback-Leibler) divergence measures how different two probability distributions are. In RLHF, we penalize the policy for deviating from a reference policy.

**Mathematical Formulation:**
```
KL(π_θ || π_ref) = E[log(π_θ(a|s) / π_ref(a|s))]

In practice:
KL_penalty = β * (log π_θ - log π_ref)
```

**Why KL Penalty?**

**1. Prevents Mode Collapse**
- Without KL: Policy might collapse to single response
- With KL: Keeps policy diverse (similar to reference)

**2. Prevents Reward Hacking**
- Without KL: Model finds hacks to maximize reward
- With KL: Constrains model to reasonable behaviors

**3. Maintains Capabilities**
- Reference model has good capabilities (from SFT)
- KL penalty preserves these capabilities
- Prevents catastrophic forgetting

**4. Stability**
- Prevents large policy changes
- More stable training
- Gradual optimization

**5. Trust Region**
- KL penalty creates trust region
- Policy can't deviate too far
- Similar to PPO clipping

**How to Choose β (KL Coefficient):**
- **Too small (β < 0.01)**: Policy can deviate too much, risk of hacks
- **Too large (β > 1.0)**: Policy can't learn, stays too close to reference
- **Typical (β = 0.1-0.5)**: Balance between learning and stability

**In Practice:**
```python
# RLHF loss with KL penalty
ratio = exp(policy_logprob - reference_logprob)
policy_loss = -ratio * reward
kl_penalty = beta * (policy_logprob - reference_logprob)
total_loss = policy_loss + kl_penalty
```

**Monitoring KL:**
- Track KL during training
- If KL too high: Increase β
- If KL too low: Decrease β
- Target: KL ≈ 0.1-0.5 nats per token


> **Saying it out loud.** The KL penalty is a leash back to the SFT model, and it's doing more work than the word "regularisation" suggests. It bounds reward hacking, because you can only exploit the reward model within a limited radius. It preserves capabilities the preference data never touches, so you don't lose coding ability while optimising for politeness. And it keeps you where the reward model was actually trained, which is the only place its scores are trustworthy. Beta is the knob — worth noting most published RLHF recipes sit at 0.01 to 0.1 rather than higher — and the practical failure mode is that too small a beta gives you gibberish and capability collapse, while too large a beta gives you a model that barely changed.
---

## Q8: How would you implement a complete RLHF pipeline?

**Answer:**

**Complete Implementation Steps:**

**Step 1: Supervised Fine-Tuning**
```python
# Train on human demonstrations
def train_sft(model, demonstrations):
    for prompt, response in demonstrations:
        outputs = model(prompt)
        loss = cross_entropy(outputs, response)
        loss.backward()
        optimizer.step()
```

**Step 2: Train Reward Model**
```python
# Train on preference pairs
def train_reward_model(reward_model, preferences):
    for prompt, chosen, rejected in preferences:
        chosen_score = reward_model(prompt, chosen)
        rejected_score = reward_model(prompt, rejected)
        
        # Binary classification: chosen > rejected
        loss = -log_sigmoid(chosen_score - rejected_score)
        loss.backward()
        optimizer.step()
```

**Step 3: RL Optimization (PPO)**
```python
def rlhf_training(policy, reference, reward_model, preferences):
    optimizer = Adam(policy.parameters())
    
    for epoch in range(num_epochs):
        # Generate responses
        responses = policy.generate(prompts)
        
        # Score with reward model
        rewards = reward_model(prompts, responses)
        
        # Get logprobs
        policy_logprobs = policy.get_logprobs(prompts, responses)
        ref_logprobs = reference.get_logprobs(prompts, responses)
        
        # Compute advantages
        advantages = compute_advantages(rewards)
        
        # PPO loss with KL penalty
        ratio = exp(policy_logprobs - ref_logprobs)
        policy_loss = -min(ratio * advantages, 
                          clip(ratio, 1-ε, 1+ε) * advantages)
        kl_penalty = beta * (policy_logprobs - ref_logprobs)
        
        loss = policy_loss + kl_penalty
        loss.backward()
        optimizer.step()
```

**Key Components:**
1. **Data**: Demonstrations + preferences
2. **Models**: Policy, reference, reward model
3. **Training**: SFT → Reward → RL
4. **Monitoring**: Reward, KL, human evaluation


> **Saying it out loud.** If someone asks me to build it, I'd walk the three stages and be explicit about what's frozen at each one. SFT is plain cross-entropy on demonstrations with the loss masked so you never grade the model on the user's own tokens — that mask is the bug interviewers look for. Then the reward model: initialise from the SFT checkpoint, swap the LM head for a scalar head, train with logistic loss on preference pairs. Then PPO, holding four models — policy, frozen reference, reward model, critic — with token-level reward being the reward-model score at the end of the sequence plus a per-token KL penalty. And I'd say what I'd monitor, because that's the part people forget: reward, KL to reference, and output entropy, with a held-out human eval, because reward alone will happily go up while the model gets worse.
---

## Summary

These questions cover:
- RLHF pipeline (detailed)
- DPO vs RLHF
- PPO (mathematical details)
- GRPO (group-based optimization)
- Challenges and solutions
- Reward hacking prevention
- KL penalty importance
- Complete implementation

All with detailed explanations, mathematical formulations, and code examples!

---

## Additional Resources for Interview Preparation

**For detailed paragraph-style explanations suitable for interviews, see:**

- **[`ppo_process_explanation.md`](ppo_process_explanation.md)**: Complete process explanations of:
  - PPO training process (full paragraph style)
  - GRPO training process (full paragraph style)
  - DPO training process (full paragraph style)
  - When to use each approach
  - Complete mathematical flow in narrative form

- **[`rlhf_pipeline_explanation.md`](rlhf_pipeline_explanation.md)**: Complete three-stage RLHF pipeline:
  - Stage 1: Supervised Fine-Tuning (detailed process)
  - Stage 2: Reward Model Training (detailed process)
  - Stage 3: RL Optimization with PPO (detailed process)
  - Challenges and solutions
  - Evaluation and iteration

These documents provide comprehensive, flowing explanations that you can use directly in interviews to explain the complete processes from start to finish.

