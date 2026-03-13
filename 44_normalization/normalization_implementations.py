"""
Normalization Techniques: Complete Implementations
Batch Normalization and Layer Normalization from scratch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# ==================== BATCH NORMALIZATION ====================

class BatchNorm1D(nn.Module):
    """
    Batch Normalization (1D)
    
    USED IN: CNNs, image classification, large batch training
    
    WHAT IT DOES:
    - Normalizes activations across the batch dimension
    - Uses batch statistics (mean, variance) during training
    - Uses running statistics during inference
    - Helps with training stability and convergence
    
    MATHEMATICAL FORMULATION:
    μ_B = (1/m) ∑_{i=1}^m x_i  (mean across batch)
    σ²_B = (1/m) ∑_{i=1}^m (x_i - μ_B)²  (variance across batch)
    x̂ = (x - μ_B) / √(σ²_B + ε)  (normalize)
    y = γ * x̂ + β  (scale and shift)
    
    WHERE:
    - m: batch size
    - x_i: i-th sample in batch
    - γ: learnable scale parameter
    - β: learnable shift parameter
    - ε: small constant for numerical stability
    
    KEY PROPERTIES:
    - Normalizes across batch dimension (first dimension)
    - Requires batch statistics (needs batch_size > 1)
    - Different behavior in training vs inference
    - Running mean/variance for inference
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))  # Scale
        self.beta = nn.Parameter(torch.zeros(num_features))  # Shift
        
        # Running statistics (for inference)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (batch, features) or (batch, seq_len, features)
        Returns:
            Normalized tensor, same shape as input
        """
        if x.dim() == 2:
            # (batch, features)
            return self._forward_2d(x)
        elif x.dim() == 3:
            # (batch, seq_len, features) - apply to each position
            batch, seq_len, features = x.shape
            x_flat = x.view(-1, features)  # (batch * seq_len, features)
            out_flat = self._forward_2d(x_flat)
            return out_flat.view(batch, seq_len, features)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D input (batch, features)"""
        if self.training:
            # Training: use batch statistics
            mean = x.mean(dim=0, keepdim=True)  # (1, features) - mean across batch
            var = x.var(dim=0, keepdim=True, unbiased=False)  # (1, features) - var across batch
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
                self.num_batches_tracked += 1
        else:
            # Inference: use running statistics
            mean = self.running_mean.unsqueeze(0)
            var = self.running_var.unsqueeze(0)
        
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        
        return out


class BatchNorm2D(nn.Module):
    """
    Batch Normalization (2D) - for CNNs
    
    Normalizes across batch and spatial dimensions
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 2D CNN input
        
        Args:
            x: Input tensor, shape (batch, channels, height, width)
        Returns:
            Normalized tensor, same shape
        """
        if self.training:
            # Compute statistics across batch and spatial dimensions
            # Mean over (batch, height, width), keep channels
            mean = x.mean(dim=(0, 2, 3), keepdim=True)  # (1, channels, 1, 1)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)  # (1, channels, 1, 1)
            
            # Update running statistics
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        
        return out


# ==================== LAYER NORMALIZATION ====================

class LayerNorm(nn.Module):
    """
    Layer Normalization
    
    USED IN: Transformers, RNNs, NLP models
    
    WHAT IT DOES:
    - Normalizes activations across the feature dimension
    - Uses per-sample statistics (independent of batch)
    - Same behavior in training and inference
    - Works with any batch size (even batch_size=1)
    
    MATHEMATICAL FORMULATION:
    μ_L = (1/d) ∑_{i=1}^d x_i  (mean across features)
    σ²_L = (1/d) ∑_{i=1}^d (x_i - μ_L)²  (variance across features)
    x̂ = (x - μ_L) / √(σ²_L + ε)  (normalize)
    y = γ * x̂ + β  (scale and shift)
    
    WHERE:
    - d: number of features
    - x_i: i-th feature
    - γ: learnable scale parameter
    - β: learnable shift parameter
    - ε: small constant for numerical stability
    
    KEY PROPERTIES:
    - Normalizes across feature dimension (last dimension)
    - Independent of batch size
    - Same in training and inference
    - No running statistics needed
    - Works with batch_size=1
    
    WHY TRANSFORMERS USE LAYERNORM:
    1. Sequence length varies → can't normalize across sequence
    2. Batch size can be small → BatchNorm unstable
    3. Need per-sample normalization → LayerNorm perfect
    4. Same behavior train/test → simpler
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(normalized_shape))  # Scale
        self.beta = nn.Parameter(torch.zeros(normalized_shape))  # Shift
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor, shape (..., features)
            Can be 2D: (batch, features)
            Or 3D: (batch, seq_len, features)
            Or any shape where last dim is features
        Returns:
            Normalized tensor, same shape as input
        """
        # Compute mean and variance across last dimension (features)
        # Keep all dimensions except last
        mean = x.mean(dim=-1, keepdim=True)  # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # (..., 1)
        
        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        out = self.gamma * x_hat + self.beta
        
        return out


# ==================== COMPARISON AND ANALYSIS ====================

def compare_normalization(x: torch.Tensor, batch_norm: BatchNorm1D, layer_norm: LayerNorm):
    """
    Compare BatchNorm and LayerNorm on same input
    """
    print("=" * 80)
    print("BatchNorm vs LayerNorm Comparison")
    print("=" * 80)
    
    batch_size, seq_len, features = x.shape
    print(f"\nInput shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features: {features}")
    
    # BatchNorm
    batch_norm.eval()
    with torch.no_grad():
        x_bn = batch_norm(x)
        bn_mean = x_bn.mean(dim=(0, 1))  # Mean across batch and sequence
        bn_std = x_bn.std(dim=(0, 1))
    
    # LayerNorm
    layer_norm.eval()
    with torch.no_grad():
        x_ln = layer_norm(x)
        ln_mean = x_ln.mean(dim=-1)  # Mean across features (per sample)
        ln_std = x_ln.std(dim=-1)
    
    print("\n" + "-" * 80)
    print("BatchNorm Statistics:")
    print("-" * 80)
    print(f"  Mean across batch+seq (should be ~0): {bn_mean.mean().item():.6f}")
    print(f"  Std across batch+seq (should be ~1): {bn_std.mean().item():.6f}")
    print(f"  Normalizes: Across batch dimension")
    print(f"  Requires: batch_size > 1")
    
    print("\n" + "-" * 80)
    print("LayerNorm Statistics:")
    print("-" * 80)
    print(f"  Mean across features (per sample): {ln_mean.mean().item():.6f}")
    print(f"  Std across features (per sample): {ln_std.mean().item():.6f}")
    print(f"  Normalizes: Across feature dimension")
    print(f"  Requires: Any batch size (even 1)")
    
    print("\n" + "-" * 80)
    print("Key Differences:")
    print("-" * 80)
    print("""
    BatchNorm:
    - Normalizes across batch (first dimension)
    - Uses batch statistics during training
    - Uses running statistics during inference
    - Requires batch_size > 1
    - Different behavior train vs test
    
    LayerNorm:
    - Normalizes across features (last dimension)
    - Uses per-sample statistics
    - Same behavior train and test
    - Works with any batch size
    - No running statistics needed
    """)


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("Normalization Techniques: BatchNorm and LayerNorm")
    print("=" * 80)
    
    # Example 1: 2D input (batch, features)
    print("\n1. 2D Input Example (batch, features)")
    print("-" * 80)
    
    batch_size = 4
    features = 128
    
    x_2d = torch.randn(batch_size, features)
    
    # BatchNorm
    bn_2d = BatchNorm1D(features)
    bn_2d.train()
    x_bn_2d = bn_2d(x_2d)
    print(f"Input shape: {x_2d.shape}")
    print(f"BatchNorm output shape: {x_bn_2d.shape}")
    print(f"BatchNorm mean (across batch): {x_bn_2d.mean(dim=0).mean().item():.6f}")
    print(f"BatchNorm std (across batch): {x_bn_2d.std(dim=0).mean().item():.6f}")
    
    # LayerNorm
    ln_2d = LayerNorm(features)
    x_ln_2d = ln_2d(x_2d)
    print(f"LayerNorm output shape: {x_ln_2d.shape}")
    print(f"LayerNorm mean (across features, per sample): {x_ln_2d.mean(dim=-1).mean().item():.6f}")
    print(f"LayerNorm std (across features, per sample): {x_ln_2d.std(dim=-1).mean().item():.6f}")
    
    # Example 2: 3D input (batch, seq_len, features) - like transformers
    print("\n2. 3D Input Example (batch, seq_len, features) - Transformer Style")
    print("-" * 80)
    
    seq_len = 512
    x_3d = torch.randn(batch_size, seq_len, features)
    
    # BatchNorm
    bn_3d = BatchNorm1D(features)
    bn_3d.train()
    x_bn_3d = bn_3d(x_3d)
    print(f"Input shape: {x_3d.shape}")
    print(f"BatchNorm output shape: {x_bn_3d.shape}")
    print(f"BatchNorm normalizes across: batch dimension")
    
    # LayerNorm (used in transformers)
    ln_3d = LayerNorm(features)
    x_ln_3d = ln_3d(x_3d)
    print(f"LayerNorm output shape: {x_ln_3d.shape}")
    print(f"LayerNorm normalizes across: feature dimension (per position)")
    
    # Comparison
    compare_normalization(x_3d, bn_3d, ln_3d)
    
    # Example 3: Why LayerNorm works with batch_size=1
    print("\n3. Batch Size = 1 Example")
    print("-" * 80)
    
    x_single = torch.randn(1, seq_len, features)
    
    # BatchNorm with batch_size=1 (problematic)
    bn_single = BatchNorm1D(features)
    bn_single.eval()  # Must use eval mode (running stats)
    x_bn_single = bn_single(x_single)
    print(f"BatchNorm with batch_size=1:")
    print(f"  Uses running statistics (not batch statistics)")
    print(f"  May not normalize correctly if running stats not updated")
    
    # LayerNorm with batch_size=1 (works perfectly)
    ln_single = LayerNorm(features)
    x_ln_single = ln_single(x_single)
    print(f"LayerNorm with batch_size=1:")
    print(f"  Works perfectly (normalizes across features)")
    print(f"  Mean: {x_ln_single.mean(dim=-1).mean().item():.6f}")
    print(f"  Std: {x_ln_single.std(dim=-1).mean().item():.6f}")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
    1. BatchNorm: Normalizes across batch → needs batch_size > 1
    2. LayerNorm: Normalizes across features → works with any batch size
    3. Transformers use LayerNorm because:
       - Sequence length varies
       - Batch size can be small
       - Need per-sample normalization
    4. LayerNorm is simpler (no running statistics)
    5. LayerNorm has same behavior in train and test
    """)

