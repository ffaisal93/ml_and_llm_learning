"""
Diffusion Model Training: Complete Guide
Detailed training procedures and best practices
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np

# ==================== TRAINING SETUP ====================

class DiffusionTrainer:
    """
    Complete training setup for diffusion models
    """
    def __init__(self, model: nn.Module, timesteps: int = 1000,
                 beta_start: float = 0.0001, beta_end: float = 0.02,
                 schedule_type: str = 'linear', device: str = 'cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Setup variance schedule
        if schedule_type == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        # Precompute values for efficiency
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('betas', betas.to(device))
        self.register_buffer('alphas', alphas.to(device))
        self.register_buffer('alphas_cumprod', alphas_cumprod.to(device))
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(device))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                            torch.sqrt(1.0 - alphas_cumprod).to(device))
        self.register_buffer('posterior_variance', 
                            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod).to(device))
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register buffer (for storing precomputed values)"""
        setattr(self, name, tensor)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: Add noise to data
        
        q(x_t | x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(
            -1, *([1] * (x_start.ndim - 1))
        )
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(
            -1, *([1] * (x_start.ndim - 1))
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, x_0: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute training loss
        
        Loss: L = E[||ε - ε_θ(x_t, t)||²]
        """
        batch_size = x_0.size(0)
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Forward diffusion: create noisy data
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = self.model(x_t, t)
        
        # Loss: MSE between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'x_t': x_t,
            't': t,
            'noise': noise,
            'noise_pred': noise_pred
        }
    
    def train_step(self, x_0: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Single training step
        """
        self.model.train()
        
        # Compute loss
        loss_dict = self.compute_loss(x_0)
        loss = loss_dict['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'mse': F.mse_loss(loss_dict['noise_pred'], loss_dict['noise']).item()
        }


# ==================== TRAINING LOOP ====================

def train_diffusion_model_complete(model: nn.Module, train_loader, num_epochs: int = 100,
                                  lr: float = 1e-4, timesteps: int = 1000,
                                  device: str = 'cpu', save_path: Optional[str] = None):
    """
    Complete training procedure for diffusion model
    
    Best Practices:
    1. Use learning rate scheduling
    2. Monitor loss carefully
    3. Save checkpoints regularly
    4. Use gradient clipping
    5. Monitor sample quality during training
    """
    # Setup trainer
    trainer = DiffusionTrainer(model, timesteps=timesteps, device=device)
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, x_0 in enumerate(train_loader):
            x_0 = x_0.to(device)
            
            # Training step
            metrics = trainer.train_step(x_0, optimizer)
            epoch_loss += metrics['loss']
            num_batches += 1
            
            # Logging
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {metrics['loss']:.4f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} completed:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save checkpoint
        if save_path and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{save_path}_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved: {save_path}_epoch_{epoch+1}.pt")
        
        # Generate sample for quality check
        if (epoch + 1) % 20 == 0:
            print("Generating sample for quality check...")
            # This would call your sampling function
            # sample = generate_sample(model, ...)
            # print(f"Sample quality: {evaluate_sample(sample)}")


# ==================== TRAINING TIPS ====================

"""
TRAINING BEST PRACTICES:

1. Variance Schedule:
   - Start with linear schedule (simple)
   - Try cosine schedule (often better)
   - Can learn schedule (advanced)

2. Learning Rate:
   - Start with 1e-4 to 1e-3
   - Use learning rate scheduling
   - Cosine annealing works well

3. Batch Size:
   - Larger batches help (if memory allows)
   - Typical: 32-128 for images
   - Adjust based on data size

4. Gradient Clipping:
   - Important for stability
   - Clip norm to 1.0
   - Prevents exploding gradients

5. Timesteps:
   - More timesteps = better quality but slower
   - Typical: 1000-4000
   - Can use fewer for faster training

6. Monitoring:
   - Watch loss carefully
   - Generate samples during training
   - Check for mode collapse

7. Data Augmentation:
   - Standard augmentations work
   - Can help with generalization

8. Model Architecture:
   - U-Net for images
   - Transformer for text
   - Time embedding is crucial

9. Initialization:
   - Proper initialization important
   - Xavier/He initialization
   - Time embedding: sinusoidal

10. Regularization:
    - Dropout in model
    - Weight decay
    - Early stopping if needed
"""


# ==================== ADVANCED TRAINING ====================

def train_with_classifier_free_guidance(model: nn.Module, train_loader,
                                       condition_loader, num_epochs: int = 100,
                                       guidance_dropout: float = 0.1, device: str = 'cpu'):
    """
    Train with classifier-free guidance
    
    Trains model with and without conditioning
    Allows stronger conditioning at inference
    """
    # During training, randomly drop conditions
    # This teaches model to work with and without conditions
    
    for epoch in range(num_epochs):
        for (x_0, c) in zip(train_loader, condition_loader):
            x_0 = x_0.to(device)
            c = c.to(device)
            
            # Randomly drop condition
            drop_mask = torch.rand(x_0.size(0), device=device) < guidance_dropout
            c[drop_mask] = None  # No condition for these samples
            
            # Train with/without condition
            # ... training code ...
            pass


if __name__ == "__main__":
    print("Diffusion Model Training Guide")
    print("=" * 60)
    print("See code for complete training procedures and best practices")

