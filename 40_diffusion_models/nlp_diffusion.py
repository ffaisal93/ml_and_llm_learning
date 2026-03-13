"""
Discrete Diffusion for NLP: Text Generation
Simple implementation for discrete token diffusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

# ==================== DISCRETE FORWARD PROCESS ====================

def discrete_forward_process(x: torch.Tensor, t: torch.Tensor, 
                            transition_matrix: torch.Tensor,
                            num_timesteps: int) -> torch.Tensor:
    """
    Discrete forward diffusion: Corrupt tokens
    
    Instead of adding Gaussian noise, we use transition matrix
    to corrupt discrete tokens.
    
    Args:
        x: Token indices, shape (batch_size, seq_len)
        t: Timesteps, shape (batch_size,)
        transition_matrix: Q_t matrix, shape (vocab_size, vocab_size)
        num_timesteps: Total number of timesteps
    Returns:
        Corrupted tokens, shape (batch_size, seq_len)
    """
    batch_size, seq_len = x.shape
    vocab_size = transition_matrix.size(0)
    
    # Get transition probabilities for each token
    # Q_t[x] gives probability distribution for corrupting token x
    x_one_hot = F.one_hot(x, num_classes=vocab_size).float()  # (batch, seq, vocab)
    
    # Apply transition: x_one_hot @ Q_t^T
    # Result: (batch, seq, vocab) - probability distribution for each position
    transition_probs = torch.matmul(x_one_hot, transition_matrix.t())
    
    # Sample corrupted tokens
    # Reshape for sampling
    transition_probs = transition_probs.view(-1, vocab_size)
    corrupted = torch.multinomial(transition_probs, num_samples=1)
    corrupted = corrupted.view(batch_size, seq_len)
    
    return corrupted


def create_absorbing_transition_matrix(vocab_size: int, mask_token_id: int,
                                      beta_t: float) -> torch.Tensor:
    """
    Create transition matrix with absorbing state (mask token)
    
    At each step, tokens transition to [MASK] with probability β_t
    
    Args:
        vocab_size: Vocabulary size
        mask_token_id: ID of mask token
        beta_t: Probability of transitioning to mask
    Returns:
        Transition matrix Q_t, shape (vocab_size, vocab_size)
    """
    Q = torch.eye(vocab_size)
    # Each token transitions to mask with prob β_t, stays with prob (1-β_t)
    Q[:, mask_token_id] = beta_t
    Q = Q + (1 - beta_t - 1) * torch.eye(vocab_size)  # Adjust diagonal
    Q = Q / Q.sum(dim=1, keepdim=True)  # Normalize
    return Q


# ==================== DISCRETE DIFFUSION MODEL ====================

class DiscreteDiffusionModel(nn.Module):
    """
    Discrete diffusion model for text generation
    
    Predicts original token distribution given corrupted tokens
    """
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Position embedding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Time embedding
        self.time_embedding = nn.Embedding(1000, d_model)  # Assume max 1000 timesteps
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection to vocabulary
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict token distribution given corrupted tokens and timestep
        
        Args:
            x: Corrupted token indices, shape (batch_size, seq_len)
            t: Timesteps, shape (batch_size,)
        Returns:
            Token logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        
        # Token embeddings
        x_emb = self.token_embedding(x)  # (batch, seq, d_model)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embedding(positions)
        
        # Time embeddings (broadcast to sequence length)
        t_emb = self.time_embedding(t)  # (batch, d_model)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, d_model)
        
        # Combine embeddings
        h = x_emb + pos_emb + t_emb
        
        # Transformer
        h = self.transformer(h)
        
        # Output logits
        logits = self.output_proj(h)
        
        return logits


# ==================== TRAINING ====================

def train_discrete_diffusion(model: nn.Module, dataloader, num_epochs: int = 10,
                            timesteps: int = 1000, mask_token_id: int = 0,
                            device: str = 'cpu'):
    """
    Train discrete diffusion model
    
    Loss: Cross-entropy between predicted and original tokens
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    # Linear schedule for β_t
    betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, x_0 in enumerate(dataloader):
            x_0 = x_0.to(device)  # Original tokens, shape (batch, seq_len)
            batch_size = x_0.size(0)
            
            # Sample random timesteps
            t = torch.randint(1, timesteps, (batch_size,), device=device)
            
            # Create transition matrix for each sample
            # Simplified: use average beta for batch
            beta_t = betas[t].mean().item()
            Q_t = create_absorbing_transition_matrix(
                model.vocab_size, mask_token_id, beta_t
            ).to(device)
            
            # Forward diffusion: corrupt tokens
            x_t = discrete_forward_process(x_0, t, Q_t, timesteps)
            
            # Predict original tokens
            logits = model(x_t, t)  # (batch, seq_len, vocab_size)
            
            # Loss: cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, model.vocab_size),
                x_0.view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


# ==================== SAMPLING/GENERATION ====================

@torch.no_grad()
def discrete_sample(model: nn.Module, shape: Tuple[int, int], timesteps: int = 1000,
                   mask_token_id: int = 0, vocab_size: int = 10000,
                   device: str = 'cpu') -> torch.Tensor:
    """
    Generate text by reversing discrete diffusion
    
    Start from all [MASK] tokens and iteratively denoise
    
    Args:
        model: Trained discrete diffusion model
        shape: (batch_size, seq_len)
        timesteps: Number of diffusion steps
        mask_token_id: ID of mask token
        vocab_size: Vocabulary size
        device: Device to run on
    Returns:
        Generated token sequences, shape (batch_size, seq_len)
    """
    batch_size, seq_len = shape
    
    # Start from all mask tokens
    x = torch.full((batch_size, seq_len), mask_token_id, device=device, dtype=torch.long)
    
    # Linear schedule
    betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
    
    # Reverse diffusion: iterate from T to 0
    for t in reversed(range(timesteps)):
        # Predict token distribution
        logits = model(x, torch.full((batch_size,), t, device=device, dtype=torch.long))
        
        # Sample tokens (can use different strategies)
        if t == 0:
            # Last step: take argmax
            x = torch.argmax(logits, dim=-1)
        else:
            # Intermediate steps: sample from distribution
            probs = F.softmax(logits, dim=-1)
            x = torch.multinomial(probs.view(-1, vocab_size), num_samples=1)
            x = x.view(batch_size, seq_len)
    
    return x


# ==================== TEXT INPAINTING ====================

@torch.no_grad()
def text_inpainting(model: nn.Module, text: torch.Tensor, mask_positions: torch.Tensor,
                   timesteps: int = 1000, mask_token_id: int = 0,
                   device: str = 'cpu') -> torch.Tensor:
    """
    Text inpainting: Fill in masked tokens
    
    Args:
        model: Trained discrete diffusion model
        text: Token sequence with some tokens masked, shape (batch_size, seq_len)
        mask_positions: Boolean mask indicating which positions to fill, shape (batch_size, seq_len)
        timesteps: Number of diffusion steps
        mask_token_id: ID of mask token
        device: Device to run on
    Returns:
        Text with filled-in tokens, shape (batch_size, seq_len)
    """
    batch_size = text.size(0)
    
    # Start: replace masked positions with mask token
    x = text.clone()
    x[mask_positions] = mask_token_id
    
    # Linear schedule
    betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
    
    # Reverse diffusion
    for t in reversed(range(timesteps)):
        # Predict token distribution
        logits = model(x, torch.full((batch_size,), t, device=device, dtype=torch.long))
        
        # Only update masked positions
        if t == 0:
            # Last step: take argmax for masked positions
            predicted = torch.argmax(logits, dim=-1)
            x[mask_positions] = predicted[mask_positions]
        else:
            # Intermediate steps: sample for masked positions
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, model.vocab_size), num_samples=1)
            sampled = sampled.view(batch_size, -1)
            x[mask_positions] = sampled[mask_positions]
    
    return x


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("Discrete Diffusion for NLP")
    print("=" * 60)
    
    vocab_size = 10000
    seq_len = 128
    timesteps = 1000
    
    # Create model
    model = DiscreteDiffusionModel(vocab_size=vocab_size)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example: Generate text
    print("\nGenerating text...")
    generated = discrete_sample(
        model, shape=(2, seq_len), timesteps=timesteps,
        mask_token_id=0, vocab_size=vocab_size
    )
    print(f"Generated text shape: {generated.shape}")
    
    # Example: Text inpainting
    print("\nText inpainting example...")
    text = torch.randint(1, vocab_size, (1, seq_len))
    mask_positions = torch.zeros(1, seq_len, dtype=torch.bool)
    mask_positions[0, 10:20] = True  # Mask positions 10-20
    filled = text_inpainting(model, text, mask_positions, timesteps=timesteps)
    print(f"Filled text shape: {filled.shape}")

