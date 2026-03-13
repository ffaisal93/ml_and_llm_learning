"""
Complete GPT Implementation: Simplest Code for All Components
This file contains a complete, simple implementation of GPT with all parts
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== 1. POSITIONAL ENCODING ====================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    
    Adds position information to token embeddings
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Compute div_term: 1 / (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Token embeddings, shape (batch_size, seq_len, d_model)
        Returns:
            x + positional encoding, shape (batch_size, seq_len, d_model)
        """
        # Add positional encoding to embeddings
        return x + self.pe[:, :x.size(1)]


# ==================== 2. MULTI-HEAD ATTENTION ====================

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    
    Complexity: O(n^2 * d) where n is sequence length, d is model dimension
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Scale factor for attention scores
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attending to certain positions
        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # 1. Linear projections: Q, K, V
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)  # (batch_size, seq_len, d_model)
        V = self.W_v(x)  # (batch_size, seq_len, d_model)
        
        # 2. Reshape for multi-head: split into num_heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Compute attention scores
        # Q @ K^T: (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, d_k, seq_len)
        # Result: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # 4. Apply mask if provided (for causal attention in GPT)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 5. Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        
        # 6. Apply attention to values
        # attention_weights @ V: (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, d_k)
        # Result: (batch_size, num_heads, seq_len, d_k)
        attended = torch.matmul(attention_weights, V)
        
        # 7. Concatenate heads
        # Transpose and reshape: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # 8. Output projection
        output = self.W_o(attended)
        
        return output


# ==================== 3. FEED-FORWARD NETWORK ====================

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    
    Two linear transformations with ReLU activation
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # FFN(x) = ReLU(xW1 + b1)W2 + b2
        return self.linear2(self.relu(self.linear1(x)))


# ==================== 4. TRANSFORMER BLOCK ====================

class TransformerBlock(nn.Module):
    """
    Single transformer decoder block (used in GPT)
    
    Contains: Multi-head attention + Feed-forward + Residual connections + Layer norm
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape (batch_size, seq_len, d_model)
            mask: Causal mask for GPT
        Returns:
            Output tensor, shape (batch_size, seq_len, d_model)
        """
        # 1. Self-attention with residual connection and layer norm
        # Pre-norm architecture: norm -> attention -> residual
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        # 2. Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


# ==================== 5. CAUSAL MASK ====================

def create_causal_mask(seq_len, device='cpu'):
    """
    Create causal mask for GPT (prevents attending to future tokens)
    
    Returns upper triangular matrix of -inf (masked) and 0 (allowed)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.to(device)


# ==================== 6. COMPLETE GPT MODEL ====================

class GPT(nn.Module):
    """
    Complete GPT (Generative Pre-trained Transformer) Model
    
    Architecture:
    1. Token embeddings
    2. Positional encoding
    3. N transformer blocks
    4. Layer norm
    5. Output projection (vocab_size)
    """
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_len=1024, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 1. Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # 5. Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Token indices, shape (batch_size, seq_len)
            mask: Optional attention mask
        Returns:
            Logits for next token prediction, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        
        # 1. Token embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        # Shape: (batch_size, seq_len, d_model)
        
        # 2. Add positional encoding
        x = self.pos_encoding(x)
        # Shape: (batch_size, seq_len, d_model)
        
        # 3. Create causal mask if not provided
        if mask is None:
            mask = create_causal_mask(seq_len, x.device)
        
        # 4. Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        # Shape: (batch_size, seq_len, d_model)
        
        # 5. Final layer norm
        x = self.final_norm(x)
        
        # 6. Project to vocabulary size
        logits = self.output_projection(x)
        # Shape: (batch_size, seq_len, vocab_size)
        
        return logits


# ==================== 7. TRAINING FUNCTION ====================

def train_gpt(model, dataloader, optimizer, device='cpu', num_epochs=1):
    """
    Training loop for GPT
    
    GPT is trained with next token prediction (language modeling)
    """
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            # Input: tokens [0, 1, 2, ..., n-1]
            # Target: tokens [1, 2, 3, ..., n] (shifted by 1)
            logits = model(input_ids)
            # logits shape: (batch_size, seq_len, vocab_size)
            
            # Reshape for cross-entropy loss
            # Flatten: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            
            # Compute loss (cross-entropy)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for training stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4f}")


# ==================== 8. DECODING FUNCTION ====================

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, 
                  top_k=50, top_p=0.9, device='cpu'):
    """
    Generate text using GPT (autoregressive decoding)
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer to convert text to/from tokens
        prompt: Starting text
        max_length: Maximum length of generated sequence
        temperature: Controls randomness (higher = more random)
        top_k: Sample from top-k tokens
        top_p: Nucleus sampling threshold
        device: Device to run on
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids]).to(device)
    
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(generated)
            # Get logits for last token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(
                    next_token_logits, top_k
                )[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # Stop if end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    return generated_text


# ==================== 9. EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example: Create a small GPT model
    vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    
    # Create model
    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Example forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Example: Training step
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1)
    )
    print(f"Example loss: {loss.item():.4f}")

