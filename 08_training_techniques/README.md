# Topic 8: Training Techniques (RLHF, DPO, PPO, GRPO)

> 🔥 **For interviews, read these first:**
> - **`ALIGNMENT_DEEP_DIVE.md`** — frontier-lab interview deep dive: full RLHF math, Bradley-Terry preference model, complete DPO derivation (whiteboard-ready), the alphabet soup (IPO/KTO/ORPO/SimPO/GRPO), Constitutional AI, RLAIF, process vs outcome supervision, reward hacking, KL blowup, mode collapse, sycophancy, alignment tax, Goodhart curves.
> - **`INTERVIEW_GRILL.md`** — 60 active-recall questions with strong answers covering the full post-training stack. Drill until you can answer 40+ cold.
>
> The README below is the conceptual overview. The two files above are where the interview-grade depth lives.

## What You'll Learn

This topic teaches you advanced LLM training techniques:
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)
- Theory and implementations

## Why We Need This

### Interview Importance
- **Hot topic**: RLHF/DPO are cutting-edge
- **Understanding**: Shows deep LLM knowledge
- **Implementation**: May ask to implement

### Real-World Application
- **ChatGPT training**: Uses RLHF
- **Model alignment**: Make models helpful, harmless
- **Preference learning**: Learn from human preferences

## Industry Use Cases

### 1. **RLHF**
**Use Case**: ChatGPT, Claude
- Align models with human preferences
- Make models helpful and safe
- Improve response quality

### 2. **DPO**
**Use Case**: Modern LLM training
- Simpler than RLHF
- Direct optimization
- No reward model needed

### 3. **PPO**
**Use Case**: Reinforcement learning
- Stable policy updates
- Used in RLHF pipeline
- General RL algorithm

## Core Intuition

These training techniques exist because next-token prediction alone does not fully capture desired assistant behavior.

A base language model can be:
- fluent
- knowledgeable
- still not aligned with user preferences

Alignment methods try to push the model toward preferred behavior without letting it drift arbitrarily far from a useful reference policy.

### RLHF

RLHF breaks the problem into stages:
1. supervised fine-tuning on demonstrations
2. reward modeling from preference data
3. policy optimization using the reward signal

The intuition is:
- first learn how to answer at all
- then learn what humans prefer
- then optimize behavior against that preference signal

### DPO

DPO skips the explicit reward-model-plus-RL loop and directly optimizes relative preference between chosen and rejected responses.

That makes it easier to train and reason about in many settings.

### PPO

PPO is mainly about making policy updates stable.

Its clipping mechanism is trying to stop the new policy from moving too far in one step.

That is why PPO appears in RLHF even though RLHF is the bigger pipeline.

## Technical Details Interviewers Often Want

### Why a Reference Model Matters

Without a reference or KL-style constraint, the policy can drift too far:
- exploit reward model quirks
- become unstable
- collapse into weird outputs

The reference model acts like an anchor.

### Why Reward Models Are Risky

A reward model is only an approximation of human preference.

That means the policy can learn to:
- game the reward
- sound good without being correct
- optimize style more than truth

This is one of the most important conceptual follow-ups in alignment interviews.

### Why DPO Is Attractive

DPO is appealing because it:
- avoids explicit on-policy RL optimization
- is simpler to implement and train
- often works well with preference pairs directly

But it is not "strictly better" in all cases. It just changes the optimization setup.

## Common Failure Modes

- reward hacking
- over-optimizing preference style while hurting factuality
- too-weak KL control causing drift
- too-strong KL control preventing meaningful improvement
- claiming DPO and RLHF are identical when they are not

## Edge Cases and Follow-Up Questions

1. Why can preference optimization hurt factual accuracy?
2. Why do we need a reference model or KL penalty?
3. Why might DPO be simpler than PPO-based RLHF?
4. What happens if the reward model is misspecified?
5. Why is SFT still needed before preference optimization in many pipelines?

## What to Practice Saying Out Loud

1. The three-stage RLHF pipeline
2. The conceptual difference between SFT, reward modeling, and PPO
3. Why DPO is simpler but not universally better
4. Why alignment metrics must be paired with truthfulness and robustness checks

## Industry-Standard Boilerplate Code

### RLHF (Simplified)

```python
"""
RLHF: Reinforcement Learning from Human Feedback
Simplified implementation
"""
import numpy as np
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """
    Reward Model: Predicts how good a response is
    Trained on human preferences
    """
    def __init__(self, model_dim: int):
        super().__init__()
        self.linear = nn.Linear(model_dim, 1)
    
    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)

def rlhf_loss(policy_logprobs: torch.Tensor, 
              reference_logprobs: torch.Tensor,
              rewards: torch.Tensor,
              beta: float = 0.1) -> torch.Tensor:
    """
    RLHF Loss (PPO-style)
    
    Args:
        policy_logprobs: Log probabilities from current policy
        reference_logprobs: Log probabilities from reference model
        rewards: Reward from reward model
        beta: KL penalty coefficient
    """
    # Ratio: how much more/less likely is policy vs reference
    ratio = torch.exp(policy_logprobs - reference_logprobs)
    
    # Policy gradient term
    policy_loss = -ratio * rewards
    
    # KL penalty: prevent policy from deviating too much
    kl_penalty = beta * (policy_logprobs - reference_logprobs)
    
    # Total loss
    loss = policy_loss + kl_penalty
    
    return loss.mean()
```

### DPO (Direct Preference Optimization)

```python
"""
DPO: Direct Preference Optimization
Simpler alternative to RLHF
No reward model needed
"""
import torch
import torch.nn.functional as F

def dpo_loss(policy_logprobs_chosen: torch.Tensor,
             policy_logprobs_rejected: torch.Tensor,
             reference_logprobs_chosen: torch.Tensor,
             reference_logprobs_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """
    DPO Loss
    
    Directly optimizes policy to prefer chosen over rejected responses
    No reward model needed - uses reference model instead
    
    Args:
        policy_logprobs_chosen: Log probs of chosen response (from policy)
        policy_logprobs_rejected: Log probs of rejected response (from policy)
        reference_logprobs_chosen: Log probs of chosen (from reference)
        reference_logprobs_rejected: Log probs of rejected (from reference)
        beta: Temperature parameter
    """
    # Log ratio for chosen
    log_ratio_chosen = policy_logprobs_chosen - reference_logprobs_chosen
    
    # Log ratio for rejected
    log_ratio_rejected = policy_logprobs_rejected - reference_logprobs_rejected
    
    # DPO objective: maximize (chosen - rejected)
    # With KL penalty to prevent deviation from reference
    loss = -F.logsigmoid(
        beta * (log_ratio_chosen - log_ratio_rejected)
    )
    
    return loss.mean()
```

### PPO (Proximal Policy Optimization)

**Detailed Implementation:** See `ppo.py` for basic implementation and `ppo_complete.py` for complete version with all four models.

**Complete Guide:** See `ppo_models_detailed.md` for comprehensive explanation of all four models:
- **Policy Model (π_θ)**: Generates responses, being optimized
- **Critic Model (V_φ)**: Estimates values, computes advantages
- **Reference Model (π_ref)**: Frozen, provides KL penalty
- **Reward Model (r_ψ)**: Scores responses, provides learning signal

**Key Concepts:**
- **Clipping**: Prevents large policy updates
- **Importance Sampling**: Reuse old data
- **Advantages**: How much better than average
- **Why used in RLHF**: Stable, sample-efficient

**Mathematical Details:**
- Policy gradient: ∇_θ L = E[r(θ) * A * ∇_θ log π_θ]
- PPO clipping: L^CLIP = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
- Value loss: L^VF = E[(V_φ(s) - R)^2]
- KL penalty: β * KL(π_θ || π_ref)

### GRPO (Group Relative Policy Optimization)

**Detailed Implementation:** See `grpo.py` for complete implementation with:
- Group-based optimization
- Relative rewards
- Multi-group handling

**Key Concepts:**
- **Relative optimization**: Better than baseline, not absolute
- **Multiple groups**: Different preferences per group
- **Fairness**: All groups improve relative to average
- **Use case**: When you have multiple user segments

## Theory

### RLHF Pipeline
1. **Supervised Fine-tuning**: Train on human demonstrations
2. **Reward Model**: Train on human preferences
3. **RL Optimization**: Use PPO to optimize policy with reward model

### DPO vs RLHF
- **RLHF**: Needs reward model, more complex
- **DPO**: No reward model, simpler, direct optimization
- **Trade-off**: DPO simpler but RLHF more flexible

## Code Files

- **`rlhf_dpo.py`**: RLHF and DPO loss implementations
- **`ppo.py`**: Basic PPO implementation with detailed explanations
- **`ppo_complete.py`**: Complete PPO with all four models (Policy, Critic, Reference, Reward)
- **`ppo_models_detailed.md`**: Comprehensive guide explaining all four models, their roles, and mathematical details
- **`ppo_process_explanation.md`**: **NEW** - Complete paragraph-style explanations of PPO, GRPO, and DPO processes for interviews
- **`rlhf_pipeline_explanation.md`**: **NEW** - Complete paragraph-style explanation of the three-stage RLHF pipeline
- **`grpo.py`**: GRPO implementation for group-based optimization
- **`rl_alignment_qa.md`**: Detailed interview Q&A on RL alignment

## Exercises

1. Implement DPO loss
2. Compare RLHF vs DPO
3. Implement PPO clipping
4. Test GRPO on multi-group data
5. Understand KL penalty role

## Next Steps

- **Topic 9**: Sampling techniques
- **Topic 10**: Optimizers
