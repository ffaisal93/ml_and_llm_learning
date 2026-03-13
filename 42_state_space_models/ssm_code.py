"""
State Space Models: Complete Implementation
Simple implementations of SSM, S4, and Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# ==================== BASIC STATE SPACE MODEL ====================

class StateSpaceModel(nn.Module):
    """
    Basic State Space Model
    
    Continuous-time SSM:
    dh/dt = A h + B u
    y = C h + D u
    
    Discrete-time SSM:
    h[k+1] = A_d h[k] + B_d u[k]
    y[k] = C_d h[k] + D_d u[k]
    
    COMPLEXITY: O(n) - linear in sequence length
    """
    def __init__(self, state_dim: int = 64, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # State matrices
        # A: State transition matrix (state_dim × state_dim)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        
        # B: Input matrix (state_dim × input_dim)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        
        # C: Output matrix (output_dim × state_dim)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.1)
        
        # D: Feedthrough (output_dim × input_dim)
        self.D = nn.Parameter(torch.randn(output_dim, input_dim) * 0.1)
        
        # Initial state
        self.h0 = nn.Parameter(torch.zeros(state_dim))
    
    def discretize(self, delta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous-time SSM to discrete-time
        
        Zero-order hold method:
        A_d = exp(Δ A)
        B_d = (A_d - I) A^(-1) B
        
        Args:
            delta: Time step size
        Returns:
            A_d, B_d: Discretized matrices
        """
        # A_d = exp(Δ A)
        A_d = torch.matrix_exp(delta * self.A)
        
        # B_d = (A_d - I) A^(-1) B
        # For numerical stability, use matrix inverse or solve
        I = torch.eye(self.state_dim, device=self.A.device)
        A_inv = torch.linalg.inv(self.A)
        B_d = (A_d - I) @ A_inv @ self.B
        
        return A_d, B_d
    
    def forward(self, u: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
        """
        Forward pass through SSM
        
        Args:
            u: Input sequence, shape (batch, seq_len, input_dim)
            delta: Time step size
        Returns:
            y: Output sequence, shape (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = u.shape
        
        # Discretize
        A_d, B_d = self.discretize(delta)
        
        # Initialize state
        h = self.h0.unsqueeze(0).expand(batch_size, -1)  # (batch, state_dim)
        
        outputs = []
        
        # Recurrence: process sequence step by step
        for k in range(seq_len):
            # State update: h[k+1] = A_d h[k] + B_d u[k]
            u_k = u[:, k, :]  # (batch, input_dim)
            h = torch.matmul(h, A_d.t()) + torch.matmul(u_k, B_d.t())
            # h shape: (batch, state_dim)
            
            # Output: y[k] = C h[k] + D u[k]
            y_k = torch.matmul(h, self.C.t()) + torch.matmul(u_k, self.D.t())
            # y_k shape: (batch, output_dim)
            
            outputs.append(y_k)
        
        # Stack outputs
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, output_dim)
        
        return y


# ==================== S4 LAYER ====================

class S4Layer(nn.Module):
    """
    S4 (Structured State Space for Sequence Modeling) Layer
    
    Uses structured state matrices for efficiency
    Can be computed via convolution (FFT) or recurrence
    """
    def __init__(self, d_model: int, state_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Input projection
        self.input_proj = nn.Linear(d_model, d_model)
        
        # S4 parameters (simplified - would use structured matrices in practice)
        self.ssm = StateSpaceModel(state_dim=state_dim, input_dim=d_model, output_dim=d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        S4 forward pass
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        # Residual connection
        residual = x
        
        # Input projection
        x = self.input_proj(x)
        
        # S4 processing
        x = self.ssm(x)  # (batch, seq_len, d_model)
        
        # Output projection
        x = self.output_proj(x)
        x = self.dropout(x)
        
        # Residual + normalization
        x = self.norm(x + residual)
        
        return x


# ==================== MAMBA (SELECTIVE SSM) ====================

class MambaBlock(nn.Module):
    """
    Mamba Block: Selective State Space Model
    
    KEY INNOVATION: Makes B and C input-dependent
    
    Standard SSM: Fixed A, B, C
    Mamba: Fixed A, but B[k] and C[k] depend on input u[k]
    
    This enables selective information processing
    """
    def __init__(self, d_model: int, state_dim: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)  # *2 for gate
        
        # Selective parameters (input-dependent)
        self.B_proj = nn.Linear(self.d_inner, state_dim)  # B depends on input
        self.C_proj = nn.Linear(self.d_inner, state_dim)  # C depends on input
        self.delta_proj = nn.Linear(self.d_inner, state_dim)  # Δ depends on input
        
        # Fixed A matrix (for efficiency)
        # In practice, would use structured form (e.g., diagonal)
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mamba forward pass
        
        Args:
            x: Input, shape (batch, seq_len, d_model)
        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Input projection + gate
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x, gate = x_proj.chunk(2, dim=-1)  # Split into x and gate
        x = F.silu(x)  # SiLU activation
        gate = F.silu(gate)
        
        # Selective parameters (input-dependent)
        B = self.B_proj(x)  # (batch, seq_len, state_dim) - depends on input!
        C = self.C_proj(x)  # (batch, seq_len, state_dim) - depends on input!
        delta = F.softplus(self.delta_proj(x))  # (batch, seq_len, state_dim) - step size
        
        # Discretize with input-dependent delta
        # A_d = exp(Δ A) for each position
        A_d = torch.exp(delta.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))
        # Shape: (batch, seq_len, state_dim, state_dim)
        
        # B_d = (A_d - I) A^(-1) B (simplified)
        I = torch.eye(self.state_dim, device=x.device)
        A_inv = torch.linalg.inv(self.A)
        B_d = (A_d - I.unsqueeze(0).unsqueeze(0)) @ A_inv.unsqueeze(0).unsqueeze(0) @ B.unsqueeze(-1)
        B_d = B_d.squeeze(-1)  # (batch, seq_len, state_dim)
        
        # State recurrence (sequential)
        h = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []
        
        for k in range(seq_len):
            # State update: h[k+1] = A_d[k] h[k] + B_d[k] u[k]
            A_d_k = A_d[:, k, :, :]  # (batch, state_dim, state_dim)
            B_d_k = B_d[:, k, :]  # (batch, state_dim)
            u_k = x[:, k, :]  # (batch, d_inner)
            
            # Project u_k to state_dim for state update
            # In practice, would have separate projection
            h = torch.bmm(A_d_k, h.unsqueeze(-1)).squeeze(-1) + B_d_k
            
            # Output: y[k] = C[k] h[k]
            C_k = C[:, k, :]  # (batch, state_dim)
            y_k = (C_k * h).sum(dim=-1, keepdim=True)  # Simplified
            outputs.append(y_k)
        
        y = torch.stack(outputs, dim=1)  # (batch, seq_len, 1)
        
        # Expand back to d_inner (simplified - would have proper projection)
        y = y.expand(-1, -1, self.d_inner)
        
        # Apply gate
        y = y * gate
        
        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)
        
        # Residual + normalization
        y = self.norm(y + residual)
        
        return y


# ==================== COMPARISON ====================

def compare_ssm_vs_transformer(seq_len: int = 10000, d_model: int = 768):
    """
    Compare SSM vs Transformer complexity
    """
    print("=" * 80)
    print("SSM vs Transformer Complexity Comparison")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  seq_len: {seq_len:,}")
    print(f"  d_model: {d_model}")
    
    # Transformer attention
    transformer_time = seq_len * seq_len * d_model  # O(n²d)
    transformer_space = seq_len * seq_len  # O(n²)
    
    # SSM recurrence
    ssm_time = seq_len * d_model  # O(nd)
    ssm_space = seq_len * d_model  # O(nd)
    
    print("\n" + "-" * 80)
    print("TIME COMPLEXITY (operations)")
    print("-" * 80)
    print(f"Transformer: {transformer_time:>15,} operations (O(n²d))")
    print(f"SSM:         {ssm_time:>15,} operations (O(nd))")
    print(f"Speedup:     {transformer_time/ssm_time:.1f}×")
    
    print("\n" + "-" * 80)
    print("SPACE COMPLEXITY (memory)")
    print("-" * 80)
    print(f"Transformer: {transformer_space:>15,} values (O(n²))")
    print(f"SSM:         {ssm_space:>15,} values (O(nd))")
    print(f"Reduction:   {transformer_space/ssm_space:.1f}×")
    
    print("\n" + "-" * 80)
    print("KEY INSIGHT")
    print("-" * 80)
    print(f"For seq_len={seq_len:,}:")
    print(f"  - Transformer: Quadratic scaling ({seq_len:,}² = {seq_len*seq_len:,})")
    print(f"  - SSM: Linear scaling ({seq_len:,})")
    print(f"  - SSM is {transformer_time/ssm_time:.0f}× faster for this sequence length")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("State Space Models Implementation")
    print("=" * 80)
    
    # Comparison
    compare_ssm_vs_transformer(seq_len=10000, d_model=768)
    
    print("\n\n" + "=" * 80)
    print("Example: Basic SSM")
    print("=" * 80)
    
    state_dim = 64
    input_dim = 1
    output_dim = 1
    
    ssm = StateSpaceModel(state_dim, input_dim, output_dim)
    
    batch_size = 2
    seq_len = 100
    u = torch.randn(batch_size, seq_len, input_dim)
    
    y = ssm(u)
    
    print(f"\nInput shape: {u.shape}")
    print(f"Output shape: {y.shape}")
    print(f"SSM parameters: {sum(p.numel() for p in ssm.parameters()):,}")
    
    print("\n" + "=" * 80)
    print("Example: Mamba Block")
    print("=" * 80)
    
    d_model = 768
    mamba = MambaBlock(d_model, state_dim=16)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = mamba(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mamba parameters: {sum(p.numel() for p in mamba.parameters()):,}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
    1. SSMs use linear recurrence: O(n) complexity
    2. Transformers use attention: O(n²) complexity
    3. SSMs better for very long sequences (100K+)
    4. Mamba makes parameters input-dependent (selective)
    5. State-of-the-art for long sequence modeling
    """)

