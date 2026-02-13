"""
GRPO (Group Relative Policy Optimization) from Scratch
Detailed implementation with explanations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def grpo_loss(policy_logprobs: torch.Tensor,
              reference_logprobs: torch.Tensor,
              group_rewards: torch.Tensor,
              beta: float = 0.1) -> torch.Tensor:
    """
    GRPO: Group Relative Policy Optimization
    
    What is GRPO?
    - Extension of PPO for group-based preferences
    - Optimizes policy relative to group performance
    - Used when you have multiple groups with different preferences
    
    Mathematical Formulation:
    L_GRPO = -E[r(θ) * (R_group - R_baseline)] + β * KL(π_θ || π_ref)
    
    Where:
    - r(θ) = π_θ(a|s) / π_ref(a|s) (importance sampling ratio)
    - R_group: Reward for this group
    - R_baseline: Baseline reward (average across groups)
    - β: KL penalty coefficient
    
    Why GRPO?
    - Handles multiple preference groups
    - Relative optimization (better than baseline)
    - Prevents over-optimization with KL penalty
    
    Args:
        policy_logprobs: Log probabilities from current policy
        reference_logprobs: Log probabilities from reference model
        group_rewards: Rewards for each group
        beta: KL penalty coefficient
    """
    # Importance sampling ratio
    ratio = torch.exp(policy_logprobs - reference_logprobs)
    
    # Group-relative rewards
    # Reward relative to baseline (average)
    baseline_reward = group_rewards.mean()
    relative_rewards = group_rewards - baseline_reward
    
    # Policy gradient term: maximize relative reward
    policy_loss = -ratio * relative_rewards
    
    # KL penalty: prevent policy from deviating too much from reference
    kl_penalty = beta * (policy_logprobs - reference_logprobs)
    
    # Total loss
    loss = (policy_loss + kl_penalty).mean()
    
    return loss


def grpo_with_clipping(policy_logprobs: torch.Tensor,
                      reference_logprobs: torch.Tensor,
                      group_rewards: torch.Tensor,
                      old_logprobs: torch.Tensor,
                      beta: float = 0.1,
                      clip_epsilon: float = 0.2) -> torch.Tensor:
    """
    GRPO with PPO-style clipping
    
    Combines GRPO with PPO clipping for stability
    
    Args:
        policy_logprobs: Current policy logprobs
        reference_logprobs: Reference model logprobs
        group_rewards: Rewards for each group
        old_logprobs: Logprobs from previous policy (for clipping)
        beta: KL penalty coefficient
        clip_epsilon: Clipping parameter
    """
    # Importance sampling ratio (current vs old policy)
    ratio = torch.exp(policy_logprobs - old_logprobs)
    
    # Group-relative advantages
    baseline_reward = group_rewards.mean()
    advantages = group_rewards - baseline_reward
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO-style clipping
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * advantages
    
    # Policy loss (pessimistic)
    policy_loss = -torch.min(unclipped, clipped)
    
    # KL penalty (relative to reference)
    kl_penalty = beta * (policy_logprobs - reference_logprobs)
    
    # Total loss
    loss = (policy_loss + kl_penalty).mean()
    
    return loss


class GRPO:
    """
    GRPO: Group Relative Policy Optimization
    
    Use case:
    - Multiple user groups with different preferences
    - Optimize policy to be better than baseline for each group
    - Example: Different age groups, different regions, different use cases
    """
    
    def __init__(self, policy_model, reference_model, beta: float = 0.1,
                 clip_epsilon: float = 0.2):
        """
        Args:
            policy_model: Current policy to optimize
            reference_model: Reference policy (frozen)
            beta: KL penalty coefficient
            clip_epsilon: PPO clipping parameter
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.beta = beta
        self.clip_epsilon = clip_epsilon
    
    def compute_group_rewards(self, responses: list, group_ids: torch.Tensor,
                             reward_model) -> torch.Tensor:
        """
        Compute rewards for each group
        
        Args:
            responses: Model responses
            group_ids: Which group each response belongs to
            reward_model: Model that scores responses
        """
        # Score responses
        scores = reward_model(responses)
        
        # Group rewards (average score per group)
        n_groups = group_ids.max() + 1
        group_rewards = torch.zeros(n_groups)
        
        for group_id in range(n_groups):
            group_mask = (group_ids == group_id)
            if group_mask.sum() > 0:
                group_rewards[group_id] = scores[group_mask].mean()
        
        return group_rewards
    
    def update(self, states: torch.Tensor, actions: torch.Tensor,
              group_ids: torch.Tensor, group_rewards: torch.Tensor,
              old_logprobs: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Update policy using GRPO
        
        Steps:
        1. Get policy and reference logprobs
        2. Compute group-relative advantages
        3. Compute GRPO loss with clipping
        4. Update policy
        """
        # Get current policy outputs
        policy_logprobs, _, _ = self.policy_model(states, actions)
        
        # Get reference policy outputs (frozen)
        with torch.no_grad():
            reference_logprobs, _, _ = self.reference_model(states, actions)
        
        # Get group rewards for each sample
        sample_group_rewards = group_rewards[group_ids]
        
        # Compute GRPO loss
        loss = grpo_with_clipping(
            policy_logprobs,
            reference_logprobs,
            sample_group_rewards,
            old_logprobs,
            self.beta,
            self.clip_epsilon
        )
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=0.5)
        optimizer.step()
        
        return loss.item()


# Usage Example
if __name__ == "__main__":
    print("GRPO (Group Relative Policy Optimization)")
    print("=" * 60)
    
    # Example: 3 groups with different preferences
    n_groups = 3
    n_samples = 30
    
    # Simulated data
    policy_logprobs = torch.randn(n_samples) * 0.1 - 2.0
    reference_logprobs = torch.randn(n_samples) * 0.1 - 2.5
    group_rewards = torch.tensor([0.8, 0.6, 0.7])  # Group 0 best, group 1 worst
    old_logprobs = policy_logprobs - 0.1  # Slightly different
    
    # Group IDs for each sample
    group_ids = torch.randint(0, n_groups, (n_samples,))
    
    # GRPO loss
    loss = grpo_with_clipping(
        policy_logprobs,
        reference_logprobs,
        group_rewards[group_ids],
        old_logprobs,
        beta=0.1,
        clip_epsilon=0.2
    )
    
    print(f"Group rewards: {group_rewards}")
    print(f"Baseline (average): {group_rewards.mean():.4f}")
    print(f"Relative rewards: {group_rewards - group_rewards.mean()}")
    print(f"\nGRPO Loss: {loss.item():.4f}")
    print()
    print("What GRPO does:")
    print("  - Optimizes policy to be better than baseline for each group")
    print("  - Group 0 (best): Small relative advantage")
    print("  - Group 1 (worst): Large relative advantage (needs more improvement)")
    print("  - Group 2 (middle): Medium relative advantage")

