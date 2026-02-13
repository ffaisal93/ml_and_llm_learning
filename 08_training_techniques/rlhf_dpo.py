"""
RLHF and DPO from Scratch
Interview question: "Explain RLHF and DPO"
"""
import numpy as np

def dpo_loss(policy_logprobs_chosen: float,
             policy_logprobs_rejected: float,
             reference_logprobs_chosen: float,
             reference_logprobs_rejected: float,
             beta: float = 0.1) -> float:
    """
    DPO Loss: Direct Preference Optimization
    
    Directly optimizes policy to prefer chosen over rejected
    No reward model needed
    
    Args:
        policy_logprobs_chosen: Log prob of chosen (from policy)
        policy_logprobs_rejected: Log prob of rejected (from policy)
        reference_logprobs_chosen: Log prob of chosen (from reference)
        reference_logprobs_rejected: Log prob of rejected (from reference)
        beta: Temperature parameter
    """
    # Log ratio
    log_ratio_chosen = policy_logprobs_chosen - reference_logprobs_chosen
    log_ratio_rejected = policy_logprobs_rejected - reference_logprobs_rejected
    
    # DPO objective
    loss = -np.log(1 / (1 + np.exp(-beta * (log_ratio_chosen - log_ratio_rejected))))
    
    return loss

def rlhf_loss(policy_logprob: float, reference_logprob: float,
              reward: float, beta: float = 0.1) -> float:
    """
    RLHF Loss (simplified PPO-style)
    
    Args:
        policy_logprob: Log prob from current policy
        reference_logprob: Log prob from reference model
        reward: Reward from reward model
        beta: KL penalty coefficient
    """
    # Ratio
    ratio = np.exp(policy_logprob - reference_logprob)
    
    # Policy gradient + KL penalty
    loss = -ratio * reward + beta * (policy_logprob - reference_logprob)
    
    return loss


# Usage
if __name__ == "__main__":
    print("RLHF and DPO")
    print("=" * 60)
    
    # DPO example
    policy_chosen = -2.0
    policy_rejected = -3.0
    ref_chosen = -2.5
    ref_rejected = -3.2
    
    dpo = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    print(f"DPO Loss: {dpo:.4f}")
    
    # RLHF example
    policy_logprob = -2.0
    ref_logprob = -2.5
    reward = 0.8
    
    rlhf = rlhf_loss(policy_logprob, ref_logprob, reward)
    print(f"RLHF Loss: {rlhf:.4f}")

