# PPO Models: Detailed Explanation of All Components

## Overview

In PPO (Proximal Policy Optimization) used for RLHF, there are four key models/components that work together. This document explains each one in detail: what they are, their mathematical role, how they're used, and where they appear in the training pipeline.

---

## Part 1: The Four Models in PPO/RLHF

### Model 1: Policy Model (π_θ)

**What it is:**
- The main model being trained
- Generates responses/actions
- Outputs probability distribution over actions
- This is what we're optimizing

**Mathematical Role:**
```
π_θ(a|s): Probability of action a given state s
```

**In Language Models:**
```
π_θ(y|x): Probability of generating response y given prompt x
```

**Outputs:**
- Log probabilities: `log π_θ(a|s)`
- Action probabilities: `π_θ(a|s)`
- Can also include value estimate (if using actor-critic architecture)

**Where it's used:**
1. **Generation**: Generate responses during training
2. **Loss computation**: Compute policy gradient
3. **Importance sampling**: Compute ratio r(θ) = π_θ / π_θ_old

**Mathematical Formulation in PPO:**
```
L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

Where:
r(θ) = π_θ(a|s) / π_θ_old(a|s)
```

**Key Point:**
- This is the model we're training
- It learns to maximize reward
- Constrained by KL penalty to stay close to reference

---

### Model 2: Critic Model (Value Function V_φ)

**What it is:**
- Estimates the value of a state
- Predicts expected future return
- Used to compute advantages
- Can be separate model or shared with policy

**Mathematical Role:**
```
V_φ(s): Expected return from state s
V_φ(s) = E[∑_{t=0}^∞ γ^t r_t | s_0 = s]
```

**In Language Models:**
```
V_φ(x): Expected reward for prompt x
```

**Outputs:**
- Value estimate: `V(s)` (scalar)
- Used to compute advantages: `A(s,a) = Q(s,a) - V(s)`

**Where it's used:**
1. **Advantage computation**: A = Q - V
2. **Value loss**: L^VF = (V_φ(s) - R)^2
3. **Baseline**: Reduces variance in policy gradient

**Mathematical Formulation:**
```
Advantage: A(s,a) = Q(s,a) - V(s)

Where:
Q(s,a) = E[∑_{t=0}^∞ γ^t r_t | s_0 = s, a_0 = a]
V(s) = E[∑_{t=0}^∞ γ^t r_t | s_0 = s]
```

**Value Loss:**
```
L^VF = E[(V_φ(s) - R)^2]

Where:
R = actual return (discounted sum of rewards)
```

**Key Point:**
- Estimates how good a state is
- Used to compute advantages (how much better than average)
- Trained with MSE loss against actual returns

**Architecture Options:**
1. **Separate Critic**: Independent model V_φ(s)
2. **Shared Base**: Policy and critic share base layers, separate heads
3. **Actor-Critic**: Single model with policy and value heads

---

### Model 3: Reference Model (π_ref)

**What it is:**
- Frozen copy of policy before RL training
- Used to compute KL penalty
- Prevents policy from deviating too much
- Typically the SFT (Supervised Fine-Tuned) model

**Mathematical Role:**
```
π_ref(a|s): Reference policy (frozen)
```

**In Language Models:**
```
π_ref(y|x): Reference model's probability of response y
```

**Outputs:**
- Log probabilities: `log π_ref(a|s)`
- Used to compute KL divergence

**Where it's used:**
1. **KL penalty computation**: KL(π_θ || π_ref)
2. **Importance sampling ratio**: r(θ) = π_θ / π_ref
3. **Regularization**: Prevents policy collapse

**Mathematical Formulation:**
```
KL Penalty: β * KL(π_θ || π_ref)

Where:
KL(π_θ || π_ref) = E[log(π_θ(a|s) / π_ref(a|s))]
                 = E[log π_θ(a|s) - log π_ref(a|s)]
```

**In PPO Loss:**
```
L_total = L^CLIP + β * KL(π_θ || π_ref)

Where:
L^CLIP = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
r(θ) = π_θ(a|s) / π_ref(a|s)
```

**Key Point:**
- Frozen (not trained)
- Provides stability and prevents mode collapse
- Ensures policy doesn't forget SFT capabilities

**Why Important:**
1. **Prevents mode collapse**: Keeps policy diverse
2. **Prevents reward hacking**: Constrains policy
3. **Maintains capabilities**: Preserves SFT knowledge
4. **Stability**: Prevents large policy changes

---

### Model 4: Reward Model (r_ψ)

**What it is:**
- Predicts reward for a response
- Trained on human preferences
- Scores how good a response is
- Used to compute rewards during RL training

**Mathematical Role:**
```
r_ψ(x, y): Reward for response y to prompt x
```

**Outputs:**
- Reward score: `r(x, y)` (scalar)
- Higher = better response

**Where it's used:**
1. **Reward computation**: Score generated responses
2. **Return computation**: R = ∑ γ^t r_t
3. **Advantage computation**: A = Q - V

**Mathematical Formulation:**
```
Reward: r_t = r_ψ(x_t, y_t)

Return: R = ∑_{t=0}^T γ^t r_t

Advantage: A(s,a) = Q(s,a) - V(s)
          = E[R | s, a] - V(s)
```

**Training (Before RL):**
```
L_reward = -log σ(r_ψ(x, y_w) - r_ψ(x, y_l))

Where:
- y_w: Chosen (winning) response
- y_l: Rejected (losing) response
- σ: Sigmoid function
```

**Key Point:**
- Trained separately before RL
- Captures human preferences
- Used to score responses during RL training
- Can be frozen or updated during RL

**Why Important:**
1. **Human preferences**: Encodes what humans want
2. **Reward signal**: Provides learning signal for policy
3. **Quality assessment**: Measures response quality

---

## Part 2: How They Work Together in PPO Training

### Complete PPO Training Loop

**Step 1: Generate Responses**
```
Using Policy Model π_θ:
responses = π_θ.generate(prompts)
```

**Step 2: Score with Reward Model**
```
Using Reward Model r_ψ:
rewards = r_ψ(prompts, responses)
```

**Step 3: Get Log Probabilities**
```
Using Policy Model π_θ:
policy_logprobs = log π_θ(responses | prompts)

Using Reference Model π_ref:
ref_logprobs = log π_ref(responses | prompts)
```

**Step 4: Compute Returns**
```
returns = compute_discounted_returns(rewards)
```

**Step 5: Compute Values**
```
Using Critic Model V_φ:
values = V_φ(prompts)
```

**Step 6: Compute Advantages**
```
advantages = returns - values  # A = Q - V
```

**Step 7: Compute PPO Loss**
```
ratio = exp(policy_logprobs - ref_logprobs)
unclipped = ratio * advantages
clipped = clip(ratio, 1-ε, 1+ε) * advantages
policy_loss = -min(unclipped, clipped)

value_loss = (values - returns)^2

kl_penalty = β * (policy_logprobs - ref_logprobs)

total_loss = policy_loss + c_v * value_loss + kl_penalty
```

**Step 8: Update Models**
```
Update Policy π_θ: optimize total_loss
Update Critic V_φ: optimize value_loss
Reference π_ref: frozen (no update)
Reward r_ψ: typically frozen (can update)
```

---

## Part 3: Mathematical Details for Each Model

### Policy Model (π_θ) - Detailed Mathematics

**Forward Pass:**
```
Input: prompt x
Output: response y with probability π_θ(y|x)

For each token:
  logits = π_θ(x, y_{<t})
  probs = softmax(logits)
  y_t ~ Categorical(probs)
```

**Log Probability:**
```
log π_θ(y|x) = ∑_{t=1}^T log π_θ(y_t | x, y_{<t})
```

**Policy Gradient:**
```
∇_θ L = E[r(θ) * A * ∇_θ log π_θ(a|s)]

Where:
r(θ) = π_θ(a|s) / π_θ_old(a|s)
A = advantage
```

**PPO Clipping:**
```
L^CLIP = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

Prevents:
- Large policy updates
- Over-optimization
- Training instability
```

---

### Critic Model (V_φ) - Detailed Mathematics

**Value Function:**
```
V_φ(s) = E_π[∑_{t=0}^∞ γ^t r_t | s_0 = s]

Where:
- γ: discount factor
- r_t: reward at time t
- π: current policy
```

**Bellman Equation:**
```
V_φ(s) = E[r + γ V_φ(s') | s]

In practice:
V_φ(s) ≈ r + γ V_φ(s')
```

**Value Loss:**
```
L^VF = E[(V_φ(s) - R)^2]

Where:
R = actual return = ∑_{t=0}^T γ^t r_t
```

**Gradient:**
```
∇_φ L^VF = E[2(V_φ(s) - R) * ∇_φ V_φ(s)]
```

**Why Value Function:**
1. **Baseline**: Reduces variance in policy gradient
2. **Advantages**: A = Q - V (how much better than average)
3. **Stability**: More stable than raw returns

---

### Reference Model (π_ref) - Detailed Mathematics

**KL Divergence:**
```
KL(π_θ || π_ref) = E_{π_θ}[log(π_θ(a|s) / π_ref(a|s))]
                 = E_{π_θ}[log π_θ(a|s) - log π_ref(a|s)]
```

**In Practice:**
```
KL_penalty = β * E[(log π_θ - log π_ref)]
```

**Properties:**
- KL ≥ 0 (always non-negative)
- KL = 0 if π_θ = π_ref
- Asymmetric: KL(π_θ || π_ref) ≠ KL(π_ref || π_θ)

**Why KL Penalty:**
1. **Trust region**: Keeps policy close to reference
2. **Prevents collapse**: Maintains diversity
3. **Stability**: Prevents large changes
4. **Capability preservation**: Keeps SFT knowledge

**Typical Values:**
- β = 0.1 - 0.5 (KL coefficient)
- Target KL: 0.1 - 0.5 nats per token
- If KL too high: increase β
- If KL too low: decrease β

---

### Reward Model (r_ψ) - Detailed Mathematics

**Reward Function:**
```
r_ψ(x, y): R → R

Maps (prompt, response) to scalar reward
```

**Training Objective:**
```
L_reward = -log σ(r_ψ(x, y_w) - r_ψ(x, y_l))

Where:
- y_w: chosen (winning) response
- y_l: rejected (losing) response
- σ: sigmoid function
```

**Interpretation:**
```
P(y_w > y_l | x) = σ(r_ψ(x, y_w) - r_ψ(x, y_l))

Probability that chosen is better than rejected
```

**During RL:**
```
For generated response y:
  reward = r_ψ(x, y)
  
Used to compute:
  - Returns: R = ∑ γ^t r_t
  - Advantages: A = Q - V
```

**Reward Shaping (Optional):**
```
r_total = r_ψ(x, y) + r_KL(x, y) + r_length(x, y)

Where:
- r_KL: KL penalty (can be in reward or loss)
- r_length: Length penalty
```

---

## Part 4: Architecture Details

### Policy Model Architecture

**Option 1: Separate Policy Network**
```python
class PolicyModel(nn.Module):
    def __init__(self):
        self.base = Transformer(...)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        hidden = self.base(x)
        logits = self.head(hidden)
        return logits
```

**Option 2: Actor-Critic (Shared Base)**
```python
class ActorCritic(nn.Module):
    def __init__(self):
        self.base = Transformer(...)  # Shared
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        hidden = self.base(x)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden)
        return logits, values
```

### Critic Model Architecture

**Option 1: Separate Critic**
```python
class CriticModel(nn.Module):
    def __init__(self):
        self.base = Transformer(...)
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x):
        hidden = self.base(x)
        value = self.head(hidden)
        return value
```

**Option 2: Shared with Policy (Actor-Critic)**
- Same as above, but shares base with policy

### Reference Model Architecture

**Same as Policy Model:**
- Copy of policy before RL training
- Frozen (no gradients)
- Used only for log probability computation

```python
# Initialize reference model
reference_model = copy.deepcopy(policy_model)
reference_model.eval()  # Freeze
for param in reference_model.parameters():
    param.requires_grad = False
```

### Reward Model Architecture

**Typical Architecture:**
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.base = base_model  # Can use policy base
        self.head = nn.Linear(d_model, 1)
    
    def forward(self, x, y):
        # Concatenate prompt and response
        input_ids = concat(x, y)
        hidden = self.base(input_ids)
        # Use last token or mean pooling
        reward = self.head(hidden[-1] or mean(hidden))
        return reward
```

---

## Part 5: Training Phases

### Phase 1: Supervised Fine-Tuning (SFT)

**Models Used:**
- Policy Model π_θ (being trained)

**Objective:**
```
L_SFT = -log π_θ(y | x)

Standard language modeling loss
```

**Result:**
- Policy model that can follow instructions
- This becomes the reference model π_ref

---

### Phase 2: Reward Model Training

**Models Used:**
- Reward Model r_ψ (being trained)

**Data:**
- Preference pairs: (x, y_w, y_l)

**Objective:**
```
L_reward = -log σ(r_ψ(x, y_w) - r_ψ(x, y_l))
```

**Result:**
- Reward model that scores responses
- Trained to prefer chosen over rejected

---

### Phase 3: RL Optimization (PPO)

**Models Used:**
- Policy Model π_θ (being trained)
- Critic Model V_φ (being trained)
- Reference Model π_ref (frozen)
- Reward Model r_ψ (typically frozen)

**Objective:**
```
L_PPO = L^CLIP + c_v * L^VF + β * KL(π_θ || π_ref)

Where:
L^CLIP = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
L^VF = E[(V_φ(s) - R)^2]
KL = E[log π_θ - log π_ref]
```

**Training Loop:**
1. Generate responses with π_θ
2. Score with r_ψ
3. Get logprobs from π_θ and π_ref
4. Compute values with V_φ
5. Compute advantages
6. Update π_θ and V_φ

**Result:**
- Aligned policy model
- Better at generating preferred responses

---

## Part 6: Summary Table

| Model | Role | Trained? | Used For | Mathematical Form |
|-------|------|----------|----------|-------------------|
| **Policy π_θ** | Generate responses | Yes | Generation, loss | π_θ(a\|s) |
| **Critic V_φ** | Estimate state value | Yes | Advantages | V_φ(s) = E[R\|s] |
| **Reference π_ref** | Regularization | No (frozen) | KL penalty | π_ref(a\|s) |
| **Reward r_ψ** | Score responses | Before RL | Rewards | r_ψ(x, y) |

**Key Relationships:**
- **Advantage**: A = Q - V = (R - V_φ)
- **Ratio**: r(θ) = π_θ / π_ref
- **KL**: KL = E[log π_θ - log π_ref]
- **Reward**: r = r_ψ(x, y)

**Training:**
- **SFT**: Train π_θ
- **Reward**: Train r_ψ
- **RL**: Train π_θ and V_φ (π_ref and r_ψ frozen)

---

## Conclusion

Understanding these four models is crucial for PPO/RLHF:

1. **Policy Model**: What we're optimizing, generates responses
2. **Critic Model**: Estimates values, computes advantages
3. **Reference Model**: Provides stability, prevents collapse
4. **Reward Model**: Scores responses, provides learning signal

Each has a specific mathematical role and is used at different stages of training. Together, they enable stable and effective RLHF training.

