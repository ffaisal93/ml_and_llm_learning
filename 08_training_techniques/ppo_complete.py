"""
PPO (Proximal Policy Optimization) - Complete Implementation
With all four models clearly separated: Policy, Critic, Reference, Reward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional
import copy

# ==================== MODEL 1: POLICY MODEL ====================

class PolicyModel(nn.Module):
    """
    Policy Model (π_θ)
    
    ROLE:
    - Generates responses/actions
    - Outputs probability distribution over actions
    - This is what we're optimizing
    
    MATHEMATICAL ROLE:
    π_θ(a|s): Probability of action a given state s
    In LLMs: π_θ(y|x): Probability of response y given prompt x
    
    USED FOR:
    1. Generation: Generate responses during training
    2. Loss computation: Compute policy gradient
    3. Importance sampling: Compute ratio r(θ) = π_θ / π_θ_old
    """
    def __init__(self, vocab_size: int, d_model: int = 768):
        super().__init__()
        # Simplified transformer-like architecture
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8),
            num_layers=6
        )
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        # Simplified - in practice would use proper attention masks
        x = self.transformer(x, x)
        logits = self.head(x)
        return logits
    
    def get_logprobs(self, input_ids: torch.Tensor, 
                     response_ids: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities of response
        
        Args:
            input_ids: Prompt tokens
            response_ids: Response tokens
        Returns:
            logprobs: Log probabilities, shape (batch,)
        """
        # Concatenate prompt and response
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        logits = self.forward(full_ids)
        
        # Get logprobs for response tokens
        response_logits = logits[:, input_ids.size(1):, :]
        logprobs = F.log_softmax(response_logits, dim=-1)
        
        # Sum over response tokens
        response_logprobs = logprobs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        return response_logprobs.sum(dim=1)  # (batch,)


# ==================== MODEL 2: CRITIC MODEL (VALUE FUNCTION) ====================

class CriticModel(nn.Module):
    """
    Critic Model / Value Function (V_φ)
    
    ROLE:
    - Estimates the value of a state
    - Predicts expected future return
    - Used to compute advantages
    
    MATHEMATICAL ROLE:
    V_φ(s) = E[∑_{t=0}^∞ γ^t r_t | s_0 = s]
    In LLMs: V_φ(x) = Expected reward for prompt x
    
    USED FOR:
    1. Advantage computation: A = Q - V
    2. Value loss: L^VF = (V_φ(s) - R)^2
    3. Baseline: Reduces variance in policy gradient
    
    ARCHITECTURE:
    - Can be separate model
    - Or shared base with policy (actor-critic)
    """
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)  # Simplified
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8),
            num_layers=6
        )
        self.value_head = nn.Linear(d_model, 1)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Returns:
            values: (batch,) - Value estimates
        """
        x = self.embedding(input_ids)
        x = self.transformer(x, x)
        # Use last token for value
        values = self.value_head(x[:, -1, :]).squeeze(-1)
        return values


# ==================== MODEL 3: REFERENCE MODEL ====================

class ReferenceModel:
    """
    Reference Model (π_ref)
    
    ROLE:
    - Frozen copy of policy before RL training
    - Used to compute KL penalty
    - Prevents policy from deviating too much
    
    MATHEMATICAL ROLE:
    π_ref(a|s): Reference policy (frozen)
    KL(π_θ || π_ref) = E[log(π_θ(a|s) / π_ref(a|s))]
    
    USED FOR:
    1. KL penalty computation: β * KL(π_θ || π_ref)
    2. Importance sampling ratio: r(θ) = π_θ / π_ref
    3. Regularization: Prevents policy collapse
    
    KEY POINT:
    - Frozen (not trained)
    - Provides stability
    - Typically the SFT model
    """
    def __init__(self, policy_model: PolicyModel):
        # Deep copy of policy model
        self.model = copy.deepcopy(policy_model)
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_logprobs(self, input_ids: torch.Tensor,
                     response_ids: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities (frozen, no gradients)
        """
        with torch.no_grad():
            return self.model.get_logprobs(input_ids, response_ids)


# ==================== MODEL 4: REWARD MODEL ====================

class RewardModel(nn.Module):
    """
    Reward Model (r_ψ)
    
    ROLE:
    - Predicts reward for a response
    - Trained on human preferences
    - Scores how good a response is
    
    MATHEMATICAL ROLE:
    r_ψ(x, y): Reward for response y to prompt x
    Trained with: L = -log σ(r_ψ(x, y_w) - r_ψ(x, y_l))
    
    USED FOR:
    1. Reward computation: Score generated responses
    2. Return computation: R = ∑ γ^t r_t
    3. Advantage computation: A = Q - V
    
    KEY POINT:
    - Trained separately before RL
    - Captures human preferences
    - Typically frozen during RL (can update)
    """
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8),
            num_layers=6
        )
        self.reward_head = nn.Linear(d_model, 1)
    
    def forward(self, input_ids: torch.Tensor,
                response_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Returns:
            rewards: (batch,) - Reward scores
        """
        # Concatenate prompt and response
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        x = self.embedding(full_ids)
        x = self.transformer(x, x)
        # Use last token for reward
        rewards = self.reward_head(x[:, -1, :]).squeeze(-1)
        return rewards


# ==================== PPO WITH ALL MODELS ====================

class PPOComplete:
    """
    Complete PPO Implementation with all four models
    
    MODELS:
    1. Policy Model π_θ: Generates responses (being trained)
    2. Critic Model V_φ: Estimates values (being trained)
    3. Reference Model π_ref: Regularization (frozen)
    4. Reward Model r_ψ: Scores responses (typically frozen)
    
    TRAINING LOOP:
    1. Generate responses with π_θ
    2. Score with r_ψ
    3. Get logprobs from π_θ and π_ref
    4. Compute values with V_φ
    5. Compute advantages
    6. Update π_θ and V_φ
    """
    def __init__(self, policy_model: PolicyModel,
                 critic_model: CriticModel,
                 reference_model: ReferenceModel,
                 reward_model: RewardModel,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 kl_coef: float = 0.1,
                 entropy_coef: float = 0.01):
        self.policy_model = policy_model
        self.critic_model = critic_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
    
    def compute_advantages(self, rewards: torch.Tensor,
                          values: torch.Tensor,
                          gamma: float = 0.99,
                          lambda_gae: float = 0.95) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        MATHEMATICAL FORMULATION:
        A(s,a) = Q(s,a) - V(s)
        
        GAE:
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        
        Where:
        δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Compute backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE
            gae = delta + gamma * lambda_gae * gae
            advantages[t] = gae
        
        return advantages
    
    def compute_returns(self, rewards: torch.Tensor,
                       gamma: float = 0.99) -> torch.Tensor:
        """
        Compute discounted returns
        
        R_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
        """
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def ppo_loss(self, policy_logprobs: torch.Tensor,
                 ref_logprobs: torch.Tensor,
                 advantages: torch.Tensor,
                 values: torch.Tensor,
                 returns: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Complete PPO Loss with all components
        
        MATHEMATICAL FORMULATION:
        L_PPO = L^CLIP + c_v * L^VF + β * KL(π_θ || π_ref)
        
        Where:
        L^CLIP = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]
        L^VF = E[(V_φ(s) - R)^2]
        KL = E[log π_θ - log π_ref]
        r(θ) = π_θ / π_ref
        """
        # Importance sampling ratio
        ratio = torch.exp(policy_logprobs - ref_logprobs)
        
        # Clipped policy loss
        unclipped = ratio * advantages
        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        )
        clipped = clipped_ratio * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # KL penalty
        kl_penalty = self.kl_coef * (policy_logprobs - ref_logprobs).mean()
        
        # Total loss
        total_loss = (policy_loss +
                     self.value_coef * value_loss +
                     kl_penalty)
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'kl_penalty': kl_penalty,
            'kl_divergence': (policy_logprobs - ref_logprobs).mean()
        }
    
    def update(self, prompts: torch.Tensor,
               responses: torch.Tensor,
               policy_optimizer: torch.optim.Optimizer,
               critic_optimizer: torch.optim.Optimizer):
        """
        Complete PPO update step
        
        STEPS:
        1. Get rewards from reward model
        2. Get logprobs from policy and reference
        3. Get values from critic
        4. Compute advantages and returns
        5. Compute loss
        6. Update policy and critic
        """
        # Step 1: Score with reward model
        rewards = self.reward_model(prompts, responses)  # (batch,)
        
        # Step 2: Get log probabilities
        policy_logprobs = self.policy_model.get_logprobs(prompts, responses)
        ref_logprobs = self.reference_model.get_logprobs(prompts, responses)
        
        # Step 3: Get values
        # For simplicity, use prompt for value (in practice, use full sequence)
        values = self.critic_model(prompts)  # (batch,)
        
        # Step 4: Compute returns and advantages
        returns = self.compute_returns(rewards)
        advantages = self.compute_advantages(rewards, values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Step 5: Compute loss
        loss_dict = self.ppo_loss(
            policy_logprobs, ref_logprobs, advantages, values, returns
        )
        
        # Step 6: Update policy
        policy_optimizer.zero_grad()
        loss_dict['policy_loss'].backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=0.5)
        policy_optimizer.step()
        
        # Step 7: Update critic
        critic_optimizer.zero_grad()
        loss_dict['value_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), max_norm=0.5)
        critic_optimizer.step()
        
        return loss_dict


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("PPO Complete Implementation")
    print("=" * 80)
    print("\nFour Models:")
    print("1. Policy Model π_θ: Generates responses (being trained)")
    print("2. Critic Model V_φ: Estimates values (being trained)")
    print("3. Reference Model π_ref: Regularization (frozen)")
    print("4. Reward Model r_ψ: Scores responses (frozen)")
    print("\n" + "=" * 80)
    
    # Create models
    vocab_size = 10000
    d_model = 768
    
    policy_model = PolicyModel(vocab_size, d_model)
    critic_model = CriticModel(d_model)
    reference_model = ReferenceModel(policy_model)
    reward_model = RewardModel(d_model)
    
    # Create PPO
    ppo = PPOComplete(
        policy_model, critic_model, reference_model, reward_model,
        clip_epsilon=0.2, value_coef=0.5, kl_coef=0.1
    )
    
    # Dummy data
    batch_size = 4
    prompt_len = 10
    response_len = 20
    
    prompts = torch.randint(0, vocab_size, (batch_size, prompt_len))
    responses = torch.randint(0, vocab_size, (batch_size, response_len))
    
    # Optimizers
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=3e-4)
    
    # Update
    loss_dict = ppo.update(prompts, responses, policy_optimizer, critic_optimizer)
    
    print("\nTraining Results:")
    print(f"  Policy Loss: {loss_dict['policy_loss'].item():.4f}")
    print(f"  Value Loss: {loss_dict['value_loss'].item():.4f}")
    print(f"  KL Penalty: {loss_dict['kl_penalty'].item():.4f}")
    print(f"  KL Divergence: {loss_dict['kl_divergence'].item():.4f}")
    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
    
    print("\n" + "=" * 80)
    print("Key Points:")
    print("=" * 80)
    print("""
    1. Policy Model: Generates responses, being optimized
    2. Critic Model: Estimates values, computes advantages
    3. Reference Model: Frozen, provides KL penalty
    4. Reward Model: Scores responses, provides learning signal
    
    All four models work together in PPO training!
    """)

