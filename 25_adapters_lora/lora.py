"""
LoRA: Low-Rank Adaptation
Simple implementation
"""
import torch
import torch.nn as nn
import numpy as np

class LoRALinear(nn.Module):
    """
    LoRA Linear Layer
    
    Mathematical Formulation:
    W' = W + ΔW = W + BA
    
    Where:
    - W: Original weight (frozen) (out_features × in_features)
    - B: Low-rank matrix (out_features × rank)
    - A: Low-rank matrix (rank × in_features)
    - rank << min(out_features, in_features)
    
    Forward: y = xW^T + xA^TB^T * (alpha/rank)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original weight (frozen - not trained)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # Freeze original weights
        
        # LoRA matrices (trainable)
        # A: initialized with small random values
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        # B: initialized to zero (so initial ΔW = 0)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor: alpha/rank
        self.scale = alpha / rank
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input (batch_size, ..., in_features)
        Returns:
            Output (batch_size, ..., out_features)
        """
        # Original: x @ W^T
        original_out = x @ self.weight.T
        
        # LoRA: x @ A^T @ B^T * scale
        # Step 1: x @ A^T -> (..., rank)
        lora_intermediate = x @ self.lora_A.T
        # Step 2: @ B^T -> (..., out_features)
        lora_out = lora_intermediate @ self.lora_B.T
        # Step 3: Scale
        lora_out = lora_out * self.scale
        
        # Combined output
        return original_out + lora_out
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        lora_params = self.lora_A.numel() + self.lora_B.numel()
        return lora_params
    
    def get_total_params(self):
        """Get total parameters (frozen + trainable)"""
        return self.weight.numel() + self.get_trainable_params()


class Adapter(nn.Module):
    """
    Adapter Layer
    
    Architecture:
    Input → Down Projection → Activation → Up Projection → Output
    With residual connection
    """
    
    def __init__(self, d_model: int, adapter_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.adapter_size = adapter_size
        
        # Down projection: d_model → adapter_size
        self.down_proj = nn.Linear(d_model, adapter_size)
        # Activation
        self.activation = nn.ReLU()
        # Up projection: adapter_size → d_model
        self.up_proj = nn.Linear(adapter_size, d_model)
    
    def forward(self, x):
        """
        Forward pass with residual connection
        
        Args:
            x: Input (batch_size, seq_len, d_model)
        Returns:
            Output (batch_size, seq_len, d_model)
        """
        # Adapter path
        adapter_out = self.down_proj(x)  # (..., adapter_size)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.up_proj(adapter_out)  # (..., d_model)
        
        # Residual connection
        return x + adapter_out


# Usage Example
if __name__ == "__main__":
    print("LoRA and Adapters")
    print("=" * 60)
    
    # Example: LoRA for attention layer
    in_features = 768
    out_features = 768
    rank = 8
    alpha = 8
    
    lora_layer = LoRALinear(in_features, out_features, rank, alpha)
    
    print("LoRA Layer:")
    print(f"  Input features: {in_features}")
    print(f"  Output features: {out_features}")
    print(f"  Rank: {rank}")
    print(f"  Alpha: {alpha}")
    print(f"  Scale: {lora_layer.scale}")
    print()
    
    # Parameter count
    total_params = lora_layer.get_total_params()
    trainable_params = lora_layer.get_trainable_params()
    frozen_params = total_params - trainable_params
    
    print("Parameter Count:")
    print(f"  Total: {total_params:,}")
    print(f"  Frozen (W): {frozen_params:,}")
    print(f"  Trainable (A + B): {trainable_params:,}")
    print(f"  Reduction: {trainable_params/total_params*100:.2f}% of total")
    print()
    
    # Forward pass
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, in_features)
    output = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()
    
    # Adapter example
    d_model = 768
    adapter_size = 64
    adapter = Adapter(d_model, adapter_size)
    
    print("Adapter Layer:")
    print(f"  d_model: {d_model}")
    print(f"  adapter_size: {adapter_size}")
    print(f"  Parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    print()
    
    # Forward pass
    x_adapter = torch.randn(batch_size, seq_len, d_model)
    output_adapter = adapter(x_adapter)
    print(f"Input shape: {x_adapter.shape}")
    print(f"Output shape: {output_adapter.shape}")

