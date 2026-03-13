"""
Diffusion Models: Complete Implementation
Simple, interview-writable code for continuous diffusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

# ==================== VARIANCE SCHEDULE ====================

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, 
                         beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear variance schedule
    
    Args:
        timesteps: Number of diffusion steps
        beta_start: Starting noise level
        beta_end: Ending noise level
    Returns:
        Beta schedule: (timesteps,)
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine variance schedule (often works better)
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent β_t from being too small
    Returns:
        Beta schedule: (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


# ==================== FORWARD DIFFUSION ====================

def q_sample(x_start: torch.Tensor, t: torch.Tensor, sqrt_alphas_cumprod: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor, noise: Optional[torch.Tensor] = None):
    """
    Forward diffusion: Add noise to data
    
    q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
    
    Args:
        x_start: Clean data, shape (batch_size, ...)
        t: Timesteps, shape (batch_size,)
        sqrt_alphas_cumprod: √(ᾱ_t) for each t
        sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t) for each t
        noise: Optional noise (for reproducibility)
    Returns:
        Noisy data x_t, shape (batch_size, ...)
    """
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # Get values for each sample in batch
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, *([1] * (x_start.ndim - 1)))
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(
        -1, *([1] * (x_start.ndim - 1))
    )
    
    # Sample: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# ==================== NOISE PREDICTION MODEL ====================

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding
    
    Encodes timestep t into a vector for conditioning
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Timesteps, shape (batch_size,)
        Returns:
            Time embeddings, shape (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SimpleDiffusionModel(nn.Module):
    """
    Simple diffusion model for continuous data
    
    Predicts noise ε given noisy data x_t and timestep t
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        
        # Simple MLP
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # Predict noise (same dim as input)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise ε given noisy data x_t and timestep t
        
        Args:
            x: Noisy data, shape (batch_size, input_dim)
            t: Timesteps, shape (batch_size,)
        Returns:
            Predicted noise, shape (batch_size, input_dim)
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Project inputs
        x_emb = self.input_proj(x)
        t_emb_proj = self.time_proj(t_emb)
        
        # Combine
        h = x_emb + t_emb_proj
        h = self.layers(h)
        
        return h


# ==================== TRAINING ====================

def train_diffusion_model(model: nn.Module, dataloader, num_epochs: int = 10,
                         timesteps: int = 1000, device: str = 'cpu'):
    """
    Train diffusion model
    
    Loss: L = E[||ε - ε_θ(x_t, t)||²]
    """
    # Setup variance schedule
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    
    # Move to device
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, x_0 in enumerate(dataloader):
            x_0 = x_0.to(device)
            batch_size = x_0.size(0)
            
            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            
            # Sample noise
            noise = torch.randn_like(x_0)
            
            # Forward diffusion: create noisy data
            x_t = q_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise)
            
            # Predict noise
            noise_pred = model(x_t, t)
            
            # Loss: MSE between predicted and actual noise
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


# ==================== SAMPLING/GENERATION ====================

@torch.no_grad()
def p_sample(model: nn.Module, x: torch.Tensor, t: int, sqrt_recip_alphas: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor, betas: torch.Tensor,
             posterior_variance: torch.Tensor) -> torch.Tensor:
    """
    Single reverse diffusion step
    
    p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_t)
    
    Args:
        x: Noisy data at step t, shape (batch_size, ...)
        t: Current timestep
        sqrt_recip_alphas: 1/√(α_t)
        sqrt_one_minus_alphas_cumprod: √(1-ᾱ_t)
        betas: β_t
        posterior_variance: Variance for sampling
    Returns:
        Denoised data at step t-1, shape (batch_size, ...)
    """
    # Predict noise
    t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
    noise_pred = model(x, t_tensor)
    
    # Compute predicted mean
    # μ_θ = (1/√(α_t))(x_t - (β_t/√(1-ᾱ_t))ε_θ)
    sqrt_recip_alphas_t = sqrt_recip_alphas[t]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    beta_t = betas[t]
    
    # Reshape for broadcasting
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.reshape(-1, *([1] * (x.ndim - 1)))
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.reshape(
        -1, *([1] * (x.ndim - 1))
    )
    beta_t = beta_t.reshape(-1, *([1] * (x.ndim - 1)))
    posterior_variance_t = posterior_variance[t].reshape(-1, *([1] * (x.ndim - 1)))
    
    # Predicted mean
    pred_mean = sqrt_recip_alphas_t * (
        x - beta_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )
    
    # Sample
    if t == 0:
        return pred_mean  # No noise at last step
    else:
        noise = torch.randn_like(x)
        return pred_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample(model: nn.Module, shape: Tuple[int, ...], timesteps: int = 1000,
           device: str = 'cpu') -> torch.Tensor:
    """
    Generate samples by reversing diffusion process
    
    Start from pure noise x_T ~ N(0, I) and iteratively denoise
    
    Args:
        model: Trained diffusion model
        shape: Shape of samples to generate (batch_size, ...)
        timesteps: Number of diffusion steps
        device: Device to run on
    Returns:
        Generated samples, shape (batch_size, ...)
    """
    # Setup variance schedule
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod) / (1.0 - alphas_cumprod)
    
    # Start from pure noise
    x = torch.randn(shape, device=device)
    
    # Reverse diffusion: iterate from T to 0
    for t in reversed(range(timesteps)):
        x = p_sample(model, x, t, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod,
                     betas, posterior_variance)
    
    return x


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("Diffusion Model Example")
    print("=" * 60)
    
    # Example: 2D data
    input_dim = 2
    timesteps = 1000
    
    # Create model
    model = SimpleDiffusionModel(input_dim=input_dim)
    
    # Example training data (2D points)
    # In practice, you'd use real data
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example: Generate samples
    print("\nGenerating samples...")
    generated = sample(model, shape=(5, input_dim), timesteps=timesteps)
    print(f"Generated samples shape: {generated.shape}")
    print(f"Sample values:\n{generated}")

