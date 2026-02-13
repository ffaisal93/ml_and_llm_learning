# Topic 8: Training Techniques (RLHF, DPO, PPO, GRPO)

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

```python
"""
PPO: Proximal Policy Optimization
Used in RLHF pipeline
"""
import torch

def ppo_loss(old_logprobs: torch.Tensor,
             new_logprobs: torch.Tensor,
             advantages: torch.Tensor,
             clip_epsilon: float = 0.2) -> torch.Tensor:
    """
    PPO Loss with clipping
    
    Clips policy updates to prevent large changes
    More stable than vanilla policy gradient
    
    Args:
        old_logprobs: Log probs from old policy
        new_logprobs: Log probs from new policy
        advantages: Advantage estimates
        clip_epsilon: Clipping parameter
    """
    # Importance sampling ratio
    ratio = torch.exp(new_logprobs - old_logprobs)
    
    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    
    # Take minimum (pessimistic)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages)
    
    return loss.mean()
```

## Theory

### RLHF Pipeline
1. **Supervised Fine-tuning**: Train on human demonstrations
2. **Reward Model**: Train on human preferences
3. **RL Optimization**: Use PPO to optimize policy with reward model

### DPO vs RLHF
- **RLHF**: Needs reward model, more complex
- **DPO**: No reward model, simpler, direct optimization
- **Trade-off**: DPO simpler but RLHF more flexible

## Exercises

1. Implement DPO loss
2. Compare RLHF vs DPO
3. Implement PPO clipping
4. Test on preference data

## Next Steps

- **Topic 9**: Sampling techniques
- **Topic 10**: Optimizers

