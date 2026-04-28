# PPO Models: Detailed Explanation of All Components

## Overview

In PPO (Proximal Policy Optimization) used for RLHF, there are four key models/components that work together. This document explains each one in detail: what they are, their mathematical role, how they're used, and where they appear in the training pipeline.

---

## Part 1: The Four Models in PPO/RLHF

### Model 1: Policy Model ($\pi_\theta$)

**What it is:**

- The main model being trained
- Generates responses/actions
- Outputs probability distribution over actions
- This is what we're optimizing

**Mathematical role.** $\pi_\theta(a \mid s)$ is the probability of action $a$ given state $s$.

**In language models.** $\pi_\theta(y \mid x)$ is the probability of generating response $y$ given prompt $x$.

**Outputs:**

- Log probabilities: $\log \pi_\theta(a \mid s)$
- Action probabilities: $\pi_\theta(a \mid s)$
- Can also include value estimate (if using actor-critic architecture)

**Where it's used:**

1. **Generation:** generate responses during training.
2. **Loss computation:** compute policy gradient.
3. **Importance sampling:** compute the ratio $r(\theta) = \pi_\theta / \pi_{\theta_{\text{old}}}$.

**Mathematical formulation in PPO:**

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}\!\left[\min\!\big(r(\theta)\, A,\; \mathrm{clip}(r(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A\big)\right]
$$

where

$$
r(\theta) \;=\; \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)}.
$$

**Key point.** This is the model we're training. It learns to maximize reward, constrained by a KL penalty to stay close to the reference.

---

### Model 2: Critic Model (Value Function $V_\phi$)

**What it is:**

- Estimates the value of a state
- Predicts expected future return
- Used to compute advantages
- Can be a separate model or share parameters with the policy

**Mathematical role.**

$$
V_\phi(s) \;=\; \mathbb{E}\!\left[\sum_{t=0}^{\infty} \gamma^{t}\, r_t \;\Big|\; s_0 = s\right].
$$

**In language models.** $V_\phi(x)$ is the expected reward for prompt $x$.

**Outputs:**

- Scalar value estimate $V(s)$.
- Used to compute advantages $A(s, a) = Q(s, a) - V(s)$.

**Where it's used:**

1. **Advantage computation:** $A = Q - V$.
2. **Value loss:** $L^{\text{VF}} = (V_\phi(s) - R)^2$.
3. **Baseline:** reduces variance in the policy gradient.

**Mathematical formulation.**

$$
A(s, a) \;=\; Q(s, a) - V(s),
$$

with

$$
Q(s, a) \;=\; \mathbb{E}\!\left[\sum_{t=0}^{\infty} \gamma^{t}\, r_t \;\Big|\; s_0 = s,\, a_0 = a\right], \qquad
V(s) \;=\; \mathbb{E}\!\left[\sum_{t=0}^{\infty} \gamma^{t}\, r_t \;\Big|\; s_0 = s\right].
$$

**Value loss.**

$$
L^{\text{VF}} \;=\; \mathbb{E}\!\left[\big(V_\phi(s) - R\big)^2\right],
$$

where $R$ is the actual return (discounted sum of rewards).

**Key point.** Estimates how good a state is, used to compute advantages (how much better than average), trained with MSE loss against actual returns.

**Architecture options:**

1. **Separate critic:** independent model $V_\phi(s)$.
2. **Shared base:** policy and critic share base layers, separate heads.
3. **Actor-critic:** single model with policy and value heads.

---

### Model 3: Reference Model ($\pi_{\text{ref}}$)

**What it is:**

- Frozen copy of the policy before RL training
- Used to compute the KL penalty
- Prevents the policy from deviating too much
- Typically the SFT (supervised fine-tuned) model

**Mathematical role.** $\pi_{\text{ref}}(a \mid s)$ is the (frozen) reference policy. For language models, $\pi_{\text{ref}}(y \mid x)$ is the reference model's probability of response $y$.

**Outputs:**

- Log probabilities $\log \pi_{\text{ref}}(a \mid s)$.
- Used to compute KL divergence.

**Where it's used:**

1. **KL penalty computation:** $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$.
2. **Importance sampling ratio:** $r(\theta) = \pi_\theta / \pi_{\text{ref}}$.
3. **Regularization:** prevents policy collapse.

**Mathematical formulation.**

$$
\text{KL penalty} \;=\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big),
$$

where

$$
\mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
\;=\; \mathbb{E}_{\pi_\theta}\!\left[\log \frac{\pi_\theta(a \mid s)}{\pi_{\text{ref}}(a \mid s)}\right]
\;=\; \mathbb{E}_{\pi_\theta}\!\left[\log \pi_\theta(a \mid s) - \log \pi_{\text{ref}}(a \mid s)\right].
$$

**In the PPO loss.**

$$
L_{\text{total}} \;=\; L^{\text{CLIP}} \;+\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big),
$$

with

$$
L^{\text{CLIP}} \;=\; \mathbb{E}\!\left[\min\!\big(r(\theta)\, A,\; \mathrm{clip}(r(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A\big)\right], \qquad
r(\theta) \;=\; \frac{\pi_\theta(a \mid s)}{\pi_{\text{ref}}(a \mid s)}.
$$

**Key point.** Frozen (not trained); provides stability, prevents mode collapse, and ensures the policy doesn't forget SFT capabilities.

**Why important:**

1. **Prevents mode collapse:** keeps the policy diverse.
2. **Prevents reward hacking:** constrains the policy.
3. **Maintains capabilities:** preserves SFT knowledge.
4. **Stability:** prevents large policy changes.

---

### Model 4: Reward Model ($r_\psi$)

**What it is:**

- Predicts a reward for a response
- Trained on human preferences
- Scores how good a response is
- Used to compute rewards during RL training

**Mathematical role.** $r_\psi(x, y)$ is the scalar reward for response $y$ to prompt $x$. Higher means better response.

**Where it's used:**

1. **Reward computation:** score generated responses.
2. **Return computation:** $R = \sum_t \gamma^{t} r_t$.
3. **Advantage computation:** $A = Q - V$.

**Mathematical formulation.**

$$
r_t = r_\psi(x_t, y_t), \qquad
R = \sum_{t=0}^{T} \gamma^{t}\, r_t, \qquad
A(s, a) = Q(s, a) - V(s) = \mathbb{E}[R \mid s, a] - V(s).
$$

**Training (before RL).** Bradley–Terry preference loss:

$$
L_{\text{reward}} \;=\; -\,\log \sigma\!\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big),
$$

where $y_w$ is the chosen (winning) response, $y_l$ is the rejected (losing) response, and $\sigma$ is the sigmoid function.

**Key point.** Trained separately before RL; captures human preferences; used to score responses during RL training; can be frozen or updated during RL.

**Why important:**

1. **Human preferences:** encodes what humans want.
2. **Reward signal:** provides the learning signal for the policy.
3. **Quality assessment:** measures response quality.

---

## Part 2: How They Work Together in PPO Training

### Complete PPO Training Loop

**Step 1 — Generate responses.** Using policy model $\pi_\theta$:

$$
\text{responses} \;=\; \pi_\theta.\text{generate}(\text{prompts}).
$$

**Step 2 — Score with reward model.** Using reward model $r_\psi$:

$$
\text{rewards} \;=\; r_\psi(\text{prompts}, \text{responses}).
$$

**Step 3 — Get log probabilities.**

$$
\text{policy\_logprobs} \;=\; \log \pi_\theta(\text{responses} \mid \text{prompts}), \qquad
\text{ref\_logprobs} \;=\; \log \pi_{\text{ref}}(\text{responses} \mid \text{prompts}).
$$

**Step 4 — Compute returns.**

$$
\text{returns} \;=\; \text{compute\_discounted\_returns}(\text{rewards}).
$$

**Step 5 — Compute values.** Using critic model $V_\phi$:

$$
\text{values} \;=\; V_\phi(\text{prompts}).
$$

**Step 6 — Compute advantages.**

$$
\text{advantages} \;=\; \text{returns} - \text{values}, \qquad A = Q - V.
$$

**Step 7 — Compute PPO loss.**

$$
\begin{aligned}
\text{ratio} &= \exp(\text{policy\_logprobs} - \text{ref\_logprobs}), \\
\text{unclipped} &= \text{ratio} \cdot \text{advantages}, \\
\text{clipped} &= \mathrm{clip}(\text{ratio}, 1-\epsilon, 1+\epsilon) \cdot \text{advantages}, \\
\text{policy\_loss} &= -\min(\text{unclipped}, \text{clipped}), \\
\text{value\_loss} &= (\text{values} - \text{returns})^2, \\
\text{kl\_penalty} &= \beta \cdot (\text{policy\_logprobs} - \text{ref\_logprobs}), \\
\text{total\_loss} &= \text{policy\_loss} + c_v \cdot \text{value\_loss} + \text{kl\_penalty}.
\end{aligned}
$$

**Step 8 — Update models.**

- Update policy $\pi_\theta$: optimize $\text{total\_loss}$.
- Update critic $V_\phi$: optimize $\text{value\_loss}$.
- Reference $\pi_{\text{ref}}$: frozen (no update).
- Reward $r_\psi$: typically frozen (can be updated).

---

## Part 3: Mathematical Details for Each Model

### Policy Model ($\pi_\theta$) — detailed mathematics

**Forward pass.** Input prompt $x$, output response $y$ with probability $\pi_\theta(y \mid x)$. For each token:

$$
\text{logits} = \pi_\theta(x, y_{<t}), \qquad
\text{probs} = \mathrm{softmax}(\text{logits}), \qquad
y_t \sim \mathrm{Categorical}(\text{probs}).
$$

**Log probability.**

$$
\log \pi_\theta(y \mid x) \;=\; \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t}).
$$

**Policy gradient.**

$$
\nabla_\theta L \;=\; \mathbb{E}\!\left[r(\theta) \cdot A \cdot \nabla_\theta \log \pi_\theta(a \mid s)\right],
$$

where $r(\theta) = \pi_\theta(a \mid s) / \pi_{\theta_{\text{old}}}(a \mid s)$ and $A$ is the advantage.

**PPO clipping.**

$$
L^{\text{CLIP}} \;=\; \mathbb{E}\!\left[\min\!\big(r(\theta)\, A,\; \mathrm{clip}(r(\theta),\, 1-\epsilon,\, 1+\epsilon)\, A\big)\right].
$$

This prevents large policy updates, over-optimization, and training instability.

---

### Critic Model ($V_\phi$) — detailed mathematics

**Value function.**

$$
V_\phi(s) \;=\; \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma^{t}\, r_t \;\Big|\; s_0 = s\right],
$$

where $\gamma$ is the discount factor, $r_t$ the reward at time $t$, and $\pi$ the current policy.

**Bellman equation.**

$$
V_\phi(s) \;=\; \mathbb{E}\!\left[r + \gamma\, V_\phi(s') \,\big|\, s\right] \;\;\;\Longrightarrow\;\;\; V_\phi(s) \;\approx\; r + \gamma\, V_\phi(s').
$$

**Value loss.**

$$
L^{\text{VF}} \;=\; \mathbb{E}\!\left[\big(V_\phi(s) - R\big)^2\right], \qquad R \;=\; \sum_{t=0}^{T} \gamma^{t}\, r_t.
$$

**Gradient.**

$$
\nabla_\phi L^{\text{VF}} \;=\; \mathbb{E}\!\left[2\,(V_\phi(s) - R) \cdot \nabla_\phi V_\phi(s)\right].
$$

**Why a value function:**

1. **Baseline:** reduces variance in the policy gradient.
2. **Advantages:** $A = Q - V$ — how much better than average.
3. **Stability:** more stable than raw returns.

---

### Reference Model ($\pi_{\text{ref}}$) — detailed mathematics

**KL divergence.**

$$
\mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
\;=\; \mathbb{E}_{\pi_\theta}\!\left[\log \frac{\pi_\theta(a \mid s)}{\pi_{\text{ref}}(a \mid s)}\right]
\;=\; \mathbb{E}_{\pi_\theta}\!\left[\log \pi_\theta(a \mid s) - \log \pi_{\text{ref}}(a \mid s)\right].
$$

**In practice.**

$$
\text{KL\_penalty} \;=\; \beta \cdot \mathbb{E}\!\left[\log \pi_\theta - \log \pi_{\text{ref}}\right].
$$

**Properties.**

- $\mathrm{KL} \ge 0$ (always non-negative).
- $\mathrm{KL} = 0$ iff $\pi_\theta = \pi_{\text{ref}}$.
- Asymmetric: $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}}) \neq \mathrm{KL}(\pi_{\text{ref}} \,\|\, \pi_\theta)$.

**Why a KL penalty:**

1. **Trust region:** keeps the policy close to the reference.
2. **Prevents collapse:** maintains diversity.
3. **Stability:** prevents large changes.
4. **Capability preservation:** keeps SFT knowledge.

**Typical values.** $\beta \in [0.1, 0.5]$; target KL $\in [0.1, 0.5]$ nats per token. If KL is too high, increase $\beta$; if too low, decrease $\beta$.

---

### Reward Model ($r_\psi$) — detailed mathematics

**Reward function.** $r_\psi(x, y) \colon \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ — maps (prompt, response) to a scalar reward.

**Training objective (Bradley–Terry).**

$$
L_{\text{reward}} \;=\; -\,\log \sigma\!\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big),
$$

where $y_w$ is the chosen (winning) response, $y_l$ the rejected (losing) response, and $\sigma$ the sigmoid function.

**Interpretation.**

$$
P(y_w \succ y_l \mid x) \;=\; \sigma\!\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big),
$$

the probability that the chosen response is preferred over the rejected one.

**During RL.** For a generated response $y$:

$$
\text{reward} = r_\psi(x, y),
$$

used to compute returns $R = \sum_t \gamma^{t} r_t$ and advantages $A = Q - V$.

**Reward shaping (optional).**

$$
r_{\text{total}} \;=\; r_\psi(x, y) \;+\; r_{\text{KL}}(x, y) \;+\; r_{\text{length}}(x, y),
$$

where $r_{\text{KL}}$ is a KL penalty (can live in the reward or in the loss) and $r_{\text{length}}$ is a length penalty.

---

## Part 4: Architecture Details

### Policy Model Architecture

**Option 1 — separate policy network:**

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

**Option 2 — actor–critic (shared base):**

```python
class ActorCritic(nn.Module):
    def __init__(self):
        self.base = Transformer(...)              # shared
        self.policy_head = nn.Linear(d_model, vocab_size)
        self.value_head  = nn.Linear(d_model, 1)

    def forward(self, x):
        hidden = self.base(x)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden)
        return logits, values
```

### Critic Model Architecture

**Option 1 — separate critic:**

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

**Option 2 — shared with policy (actor–critic):** same as above, but shares the base with the policy.

### Reference Model Architecture

Same as the policy model — a copy of the policy before RL training. Frozen (no gradients), used only for log-probability computation.

```python
# Initialize reference model
reference_model = copy.deepcopy(policy_model)
reference_model.eval()  # freeze
for param in reference_model.parameters():
    param.requires_grad = False
```

### Reward Model Architecture

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        self.base = base_model        # can use policy base
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, y):
        # Concatenate prompt and response
        input_ids = concat(x, y)
        hidden = self.base(input_ids)
        # Use last token or mean pooling
        reward = self.head(hidden[-1])  # or mean(hidden)
        return reward
```

---

## Part 5: Training Phases

### Phase 1: Supervised Fine-Tuning (SFT)

**Models used.** Policy model $\pi_\theta$ (being trained).

**Objective (standard language modeling loss).**

$$
L_{\text{SFT}} \;=\; -\,\log \pi_\theta(y \mid x).
$$

**Result.** A policy model that can follow instructions; this becomes the reference model $\pi_{\text{ref}}$.

---

### Phase 2: Reward Model Training

**Models used.** Reward model $r_\psi$ (being trained).

**Data.** Preference pairs $(x, y_w, y_l)$.

**Objective.**

$$
L_{\text{reward}} \;=\; -\,\log \sigma\!\big(r_\psi(x, y_w) - r_\psi(x, y_l)\big).
$$

**Result.** A reward model that scores responses, trained to prefer chosen over rejected.

---

### Phase 3: RL Optimization (PPO)

**Models used.**

- Policy model $\pi_\theta$ (being trained).
- Critic model $V_\phi$ (being trained).
- Reference model $\pi_{\text{ref}}$ (frozen).
- Reward model $r_\psi$ (typically frozen).

**Objective.**

$$
L_{\text{PPO}} \;=\; L^{\text{CLIP}} \;+\; c_v \cdot L^{\text{VF}} \;+\; \beta \cdot \mathrm{KL}\!\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big),
$$

with

$$
\begin{aligned}
L^{\text{CLIP}} &= \mathbb{E}\!\left[\min\!\big(r(\theta)\, A,\; \mathrm{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\, A\big)\right], \\
L^{\text{VF}}   &= \mathbb{E}\!\left[(V_\phi(s) - R)^2\right], \\
\mathrm{KL}     &= \mathbb{E}\!\left[\log \pi_\theta - \log \pi_{\text{ref}}\right].
\end{aligned}
$$

**Training loop.**

1. Generate responses with $\pi_\theta$.
2. Score with $r_\psi$.
3. Get logprobs from $\pi_\theta$ and $\pi_{\text{ref}}$.
4. Compute values with $V_\phi$.
5. Compute advantages.
6. Update $\pi_\theta$ and $V_\phi$.

**Result.** An aligned policy model, better at generating preferred responses.

---

## Part 6: Summary Table

| Model | Role | Trained? | Used for | Mathematical form |
|-------|------|----------|----------|-------------------|
| **Policy $\pi_\theta$** | Generate responses | Yes | Generation, loss | $\pi_\theta(a \mid s)$ |
| **Critic $V_\phi$** | Estimate state value | Yes | Advantages | $V_\phi(s) = \mathbb{E}[R \mid s]$ |
| **Reference $\pi_{\text{ref}}$** | Regularization | No (frozen) | KL penalty | $\pi_{\text{ref}}(a \mid s)$ |
| **Reward $r_\psi$** | Score responses | Before RL | Rewards | $r_\psi(x, y)$ |

**Key relationships.**

- **Advantage:** $A = Q - V = R - V_\phi$.
- **Ratio:** $r(\theta) = \pi_\theta / \pi_{\text{ref}}$.
- **KL:** $\mathrm{KL} = \mathbb{E}\!\left[\log \pi_\theta - \log \pi_{\text{ref}}\right]$.
- **Reward:** $r = r_\psi(x, y)$.

**Training.**

- **SFT:** train $\pi_\theta$.
- **Reward:** train $r_\psi$.
- **RL:** train $\pi_\theta$ and $V_\phi$ ($\pi_{\text{ref}}$ and $r_\psi$ frozen).

---

## Conclusion

Understanding these four models is crucial for PPO/RLHF:

1. **Policy model:** what we're optimizing; generates responses.
2. **Critic model:** estimates values; computes advantages.
3. **Reference model:** provides stability; prevents collapse.
4. **Reward model:** scores responses; provides the learning signal.

Each has a specific mathematical role and is used at different stages of training. Together, they enable stable and effective RLHF training.
