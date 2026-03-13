"""
Mixture of Experts: Complete Implementation
Simple, interview-writable code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

# ==================== BASIC MOE LAYER ====================

class Expert(nn.Module):
    """
    Single Expert Network
    
    Standard feed-forward network
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard FFN: FFN(x) = ReLU(xW1 + b1)W2 + b2
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Router(nn.Module):
    """
    Router/Gating Network
    
    Decides which experts to activate
    """
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute expert scores
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        Returns:
            Expert scores, shape (batch, seq_len, num_experts)
        """
        return self.gate(x)  # Logits, not probabilities yet


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer
    
    KEY COMPONENTS:
    1. Multiple experts (feed-forward networks)
    2. Router (gating network)
    3. Top-k routing (select k experts)
    4. Weighted combination of expert outputs
    
    EFFICIENCY:
    - Total parameters: num_experts × params_per_expert
    - Active parameters: k × params_per_expert
    - Only k experts compute per token
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int, 
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = Router(d_model, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with MoE
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        Returns:
            output: MoE output, shape (batch, seq_len, d_model)
            routing_info: Dictionary with routing statistics
        """
        batch_size, seq_len, d_model = x.shape
        
        # Step 1: Compute router scores
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Step 2: Top-k routing
        # Select k experts with highest scores
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # top_k_probs: (batch, seq_len, k)
        # top_k_indices: (batch, seq_len, k)
        
        # Step 3: Renormalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Step 4: Process through selected experts
        # Flatten for processing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (batch * seq_len, k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)  # (batch * seq_len, k)
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Process each position
        for i in range(batch_size * seq_len):
            expert_indices = top_k_indices_flat[i]  # (k,)
            expert_probs = top_k_probs_flat[i]  # (k,)
            
            # Weighted combination of expert outputs
            for j, expert_idx in enumerate(expert_indices):
                expert_output = self.experts[expert_idx](x_flat[i:i+1])
                output_flat[i] += expert_probs[j] * expert_output.squeeze(0)
        
        # Reshape back
        output = output_flat.view(batch_size, seq_len, d_model)
        
        # Compute routing statistics
        routing_info = {
            'expert_usage': self._compute_expert_usage(top_k_indices),
            'load_balance_loss': self._compute_load_balance_loss(router_probs)
        }
        
        return output, routing_info
    
    def _compute_expert_usage(self, top_k_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute how many times each expert is used
        
        Returns: (num_experts,) tensor with usage counts
        """
        usage = torch.zeros(self.num_experts, device=top_k_indices.device)
        for idx in top_k_indices.flatten():
            usage[idx] += 1
        return usage / top_k_indices.numel()  # Normalize
    
    def _compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss
        
        Encourages uniform expert usage
        L = (1/num_experts) * sum(load_i)²
        
        Where load_i is fraction of tokens routed to expert i
        """
        # Average router probabilities across batch and sequence
        avg_probs = router_probs.mean(dim=(0, 1))  # (num_experts,)
        
        # Load balancing loss: encourage uniform distribution
        # Variance of expert usage (want low variance = uniform)
        load_balance_loss = torch.var(avg_probs)
        
        return load_balance_loss


# ==================== EFFICIENT MOE (VECTORIZED) ====================

class EfficientMoE(nn.Module):
    """
    More efficient MoE implementation
    
    Uses vectorized operations instead of loops
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int,
                 top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Efficient forward pass (vectorized)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Router scores
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Flatten
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (batch * seq_len, k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)  # (batch * seq_len, k)
        
        # Process all experts (inefficient but simpler)
        # In practice, would only process selected experts
        expert_outputs = torch.stack([
            expert(x_flat) for expert in self.experts
        ], dim=1)  # (batch * seq_len, num_experts, d_model)
        
        # Select and weight
        batch_indices = torch.arange(batch_size * seq_len, device=x.device)
        selected_outputs = expert_outputs[batch_indices.unsqueeze(1), top_k_indices_flat]
        # (batch * seq_len, k, d_model)
        
        # Weighted combination
        output_flat = (selected_outputs * top_k_probs_flat.unsqueeze(-1)).sum(dim=1)
        # (batch * seq_len, d_model)
        
        output = output_flat.view(batch_size, seq_len, d_model)
        
        routing_info = {
            'load_balance_loss': torch.var(router_probs.mean(dim=(0, 1)))
        }
        
        return output, routing_info


# ==================== SWITCH ROUTING (k=1) ====================

class SwitchMoE(nn.Module):
    """
    Switch Transformer style MoE (k=1)
    
    Always activates exactly 1 expert
    Maximum sparsity
    """
    def __init__(self, d_model: int, d_ff: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k=1, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Switch routing: always k=1
        """
        return self.moe(x)


# ==================== COMPARISON ====================

def compare_moe_vs_dense(d_model: int = 768, d_ff: int = 3072,
                         num_experts: int = 8, top_k: int = 2):
    """
    Compare MoE vs Dense model
    """
    print("=" * 80)
    print("MoE vs Dense Model Comparison")
    print("=" * 80)
    
    # Dense model parameters
    dense_params = 2 * d_model * d_ff  # FFN: d_model → d_ff → d_model
    
    # MoE model parameters
    expert_params = 2 * d_model * d_ff  # Per expert
    router_params = d_model * num_experts  # Router
    moe_total_params = num_experts * expert_params + router_params
    moe_active_params = top_k * expert_params + router_params
    
    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    
    print("\n" + "-" * 80)
    print("PARAMETERS")
    print("-" * 80)
    print(f"Dense:     {dense_params:>12,} parameters")
    print(f"MoE Total: {moe_total_params:>12,} parameters ({moe_total_params/dense_params:.1f}×)")
    print(f"MoE Active: {moe_active_params:>11,} parameters ({moe_active_params/dense_params:.1f}×)")
    
    print("\n" + "-" * 80)
    print("COMPUTATION (per token)")
    print("-" * 80)
    dense_flops = 2 * d_model * d_ff
    moe_active_flops = top_k * 2 * d_model * d_ff + d_model * num_experts
    print(f"Dense:     {dense_flops:>12,} FLOPs")
    print(f"MoE Active: {moe_active_flops:>11,} FLOPs ({moe_active_flops/dense_flops:.2f}×)")
    
    print("\n" + "-" * 80)
    print("EFFICIENCY")
    print("-" * 80)
    print(f"MoE has {moe_total_params/dense_params:.1f}× more parameters")
    print(f"But only uses {moe_active_params/dense_params:.1f}× for computation")
    print(f"Efficiency: {moe_total_params/moe_active_params:.1f}× parameter efficiency")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("Mixture of Experts Implementation")
    print("=" * 80)
    
    # Comparison
    compare_moe_vs_dense(d_model=768, d_ff=3072, num_experts=8, top_k=2)
    
    print("\n\n" + "=" * 80)
    print("Example: MoE Layer")
    print("=" * 80)
    
    d_model = 768
    d_ff = 3072
    num_experts = 8
    top_k = 2
    
    moe = MixtureOfExperts(d_model, d_ff, num_experts, top_k)
    
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)
    
    output, routing_info = moe(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expert usage: {routing_info['expert_usage']}")
    print(f"Load balance loss: {routing_info['load_balance_loss']:.4f}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
    1. MoE enables models with many parameters (trillions)
    2. But only activates subset per token (efficient)
    3. Router decides which experts to use
    4. Load balancing ensures all experts are utilized
    5. Used in GPT-4, Mixtral-8x7B for efficiency
    """)

