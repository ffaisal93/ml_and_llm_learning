"""
PPO (Proximal Policy Optimization) from Scratch
Detailed implementation with explanations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPO:
    """
    PPO: Proximal Policy Optimization
    
    Why PPO?
    - More stable than vanilla policy gradient
    - Prevents large policy updates
    - Used in RLHF pipeline for ChatGPT, Claude
    
    How it works:
    1. Collect trajectories with current policy
    2. Compute advantages (how good actions were)
    3. Update policy with clipped objective
    4. Clip prevents policy from changing too much
    """
    
    def __init__(self, policy_model, clip_epsilon: float = 0.2, 
                 value_coef: float = 0.5, entropy_coef: float = 0.01):
        """
        Args:
            policy_model: Policy network (outputs action probabilities)
            clip_epsilon: Clipping parameter (typically 0.1-0.3)
            value_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus (encourages exploration)
        """
        self.policy_model = policy_model
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def compute_advantages(self, rewards: torch.Tensor, 
                          values: torch.Tensor,
                          gamma: float = 0.99,
                          lambda_gae: float = 0.95) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        Advantage = Q(s,a) - V(s)
        How much better was this action than average?
        
        GAE combines TD error with eligibility traces
        More stable than simple advantage
        
        Args:
            rewards: Rewards received
            values: Value estimates V(s)
            gamma: Discount factor
            lambda_gae: GAE parameter
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Compute advantages backwards (from end to start)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + gamma * lambda_gae * gae
            advantages[t] = gae
        
        return advantages
    
    def ppo_loss(self, old_logprobs: torch.Tensor,
                 new_logprobs: torch.Tensor,
                 advantages: torch.Tensor,
                 old_values: torch.Tensor,
                 returns: torch.Tensor,
                 entropy: torch.Tensor) -> dict:
        """
        PPO Loss with clipping
        
        Mathematical Formulation:
        L^CLIP(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        
        Where:
        - r(θ) = π_θ(a|s) / π_θ_old(a|s) (importance sampling ratio)
        - A = advantage
        - ε = clip_epsilon
        
        Why clipping?
        - Prevents large policy updates
        - More stable training
        - Takes minimum (pessimistic) to avoid over-optimization
        
        Args:
            old_logprobs: Log probabilities from old policy
            new_logprobs: Log probabilities from new policy
            advantages: Advantage estimates
            old_values: Value estimates from old policy
            returns: Actual returns (discounted rewards)
            entropy: Entropy of policy (for exploration)
        """
        # Importance sampling ratio
        # r(θ) = exp(log π_θ - log π_θ_old) = π_θ / π_θ_old
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # Unclipped objective: r(θ) * A
        unclipped = ratio * advantages
        
        # Clipped objective: clip(r(θ), 1-ε, 1+ε) * A
        clipped_ratio = torch.clamp(
            ratio, 
            1 - self.clip_epsilon, 
            1 + self.clip_epsilon
        )
        clipped = clipped_ratio * advantages
        
        # Policy loss: Take minimum (pessimistic)
        # This prevents over-optimization
        policy_loss = -torch.min(unclipped, clipped).mean()
        
        # Value loss: How well does value function predict returns?
        value_loss = F.mse_loss(old_values, returns)
        
        # Entropy bonus: Encourages exploration
        # Higher entropy = more random = more exploration
        entropy_bonus = -entropy.mean()  # Negative because we maximize entropy
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coef * value_loss + 
                     self.entropy_coef * entropy_bonus)
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy.mean()
        }
    
    def update(self, states: torch.Tensor, actions: torch.Tensor,
              old_logprobs: torch.Tensor, rewards: torch.Tensor,
              old_values: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Update policy using PPO
        
        Steps:
        1. Get new policy outputs
        2. Compute advantages
        3. Compute PPO loss
        4. Backward and update
        """
        # Get new policy outputs
        new_logprobs, new_values, entropy = self.policy_model(states, actions)
        
        # Compute returns (discounted rewards)
        returns = self.compute_returns(rewards)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, old_values)
        
        # Normalize advantages (stabilizes training)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute loss
        loss_dict = self.ppo_loss(
            old_logprobs, new_logprobs, advantages,
            old_values, returns, entropy
        )
        
        # Update
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=0.5)
        optimizer.step()
        
        return loss_dict
    
    def compute_returns(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns


# Simplified Policy Model for demonstration
class SimplePolicy(nn.Module):
    """
    Simple policy network
    
    Outputs:
    - Action probabilities (for policy)
    - Value estimate (for value function)
    - Entropy (for exploration)
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Policy head (action probabilities)
        self.policy_head = nn.Linear(64, action_dim)
        
        # Value head (state value)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor = None):
        """
        Forward pass
        
        Returns:
        - logprobs: Log probabilities of actions
        - values: Value estimates V(s)
        - entropy: Entropy of policy distribution
        """
        features = self.shared(states)
        
        # Policy: action probabilities
        logits = self.policy_head(features)
        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        
        # Value: V(s)
        values = self.value_head(features).squeeze(-1)
        
        # Entropy: H(π) = -Σ π(a|s) log π(a|s)
        entropy = -(probs * logprobs).sum(dim=-1)
        
        # If actions provided, get logprob of those actions
        if actions is not None:
            action_logprobs = logprobs.gather(1, actions.unsqueeze(1)).squeeze(1)
            return action_logprobs, values, entropy
        
        return logprobs, values, entropy


# Usage Example
if __name__ == "__main__":
    print("PPO (Proximal Policy Optimization)")
    print("=" * 60)
    
    # Create policy
    state_dim = 4
    action_dim = 2
    policy = SimplePolicy(state_dim, action_dim)
    
    # Create PPO
    ppo = PPO(policy, clip_epsilon=0.2)
    
    # Dummy data (simulated trajectory)
    batch_size = 10
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    old_logprobs, old_values, _ = policy(states, actions)
    rewards = torch.randn(batch_size)  # Simulated rewards
    
    print(f"States shape: {states.shape}")
    print(f"Actions: {actions}")
    print(f"Rewards: {rewards}")
    print()
    
    # Compute advantages
    advantages = ppo.compute_advantages(rewards, old_values)
    print(f"Advantages: {advantages}")
    print()
    
    # Update
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    loss_dict = ppo.update(states, actions, old_logprobs, rewards, old_values, optimizer)
    
    print("PPO Update Results:")
    print(f"  Policy Loss: {loss_dict['policy_loss'].item():.4f}")
    print(f"  Value Loss: {loss_dict['value_loss'].item():.4f}")
    print(f"  Entropy: {loss_dict['entropy'].item():.4f}")
    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")

