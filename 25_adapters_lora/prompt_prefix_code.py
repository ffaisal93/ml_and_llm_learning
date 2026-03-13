"""
Prompt Tuning and Prefix Tuning: Complete Implementations
Simple, interview-writable code
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================== PROMPT TUNING ====================

class PromptTuning(nn.Module):
    """
    Prompt Tuning: Learn continuous prompt embeddings
    
    Only trains prompt embeddings, keeps model frozen
    Parameter-efficient: Only p × d_model parameters
    """
    def __init__(self, base_model, prompt_length=20, prompt_init="random"):
        super().__init__()
        self.base_model = base_model
        self.prompt_length = prompt_length
        self.d_model = base_model.config.n_embd  # Model dimension
        
        # Freeze entire model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Initialize prompt embeddings
        if prompt_init == "random":
            # Random initialization
            self.prompt_embeddings = nn.Parameter(
                torch.randn(prompt_length, self.d_model) * 0.02
            )
        elif prompt_init == "vocab":
            # Initialize from vocabulary
            vocab_embeddings = base_model.transformer.wte.weight
            random_indices = torch.randint(0, vocab_embeddings.size(0), 
                                         (prompt_length,))
            self.prompt_embeddings = nn.Parameter(
                vocab_embeddings[random_indices].clone()
            )
        else:
            raise ValueError(f"Unknown prompt_init: {prompt_init}")
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward with prompt tuning
        
        Args:
            input_ids: Token indices, shape (batch_size, seq_len)
        Returns:
            Logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size = input_ids.size(0)
        
        # Get input embeddings
        input_embeddings = self.base_model.transformer.wte(input_ids)
        # Shape: (batch_size, seq_len, d_model)
        
        # Expand prompt for batch
        prompt_embeddings = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        # Shape: (batch_size, prompt_length, d_model)
        
        # Concatenate: [prompt; input]
        combined_embeddings = torch.cat(
            [prompt_embeddings, input_embeddings], dim=1
        )
        # Shape: (batch_size, prompt_length + seq_len, d_model)
        
        # Adjust attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.prompt_length,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Forward through frozen model
        outputs = self.base_model.transformer(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask
        )
        
        # Get logits
        logits = self.base_model.lm_head(outputs.last_hidden_state)
        
        return logits
    
    def get_num_params(self):
        """Get number of trainable parameters"""
        return self.prompt_embeddings.numel()


# ==================== PREFIX TUNING ====================

class PrefixTuning(nn.Module):
    """
    Prefix Tuning: Add trainable prefixes at each layer
    
    Adds prefix key-value pairs at every transformer layer
    More parameters than prompt tuning but more expressive
    """
    def __init__(self, base_model, prefix_length=20, reparam=True):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.num_layers = base_model.config.n_layer
        self.d_model = base_model.config.n_embd
        self.num_heads = base_model.config.n_head
        self.d_k = self.d_model // self.num_heads
        
        # Freeze entire model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Prefix embeddings (reparameterized or direct)
        if reparam:
            # Reparameterization: learn in smaller space, project up
            # More stable training
            self.prefix_emb = nn.Parameter(
                torch.randn(prefix_length, self.d_model // 2) * 0.02
            )
            self.prefix_proj = nn.Linear(self.d_model // 2, self.d_model)
        else:
            # Direct parameterization
            self.prefix_emb = nn.Parameter(
                torch.randn(prefix_length, self.d_model) * 0.02
            )
            self.prefix_proj = None
        
        # Project to key and value for each layer
        self.prefix_k_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.num_layers)
        ])
        self.prefix_v_proj = nn.ModuleList([
            nn.Linear(self.d_model, self.d_model)
            for _ in range(self.num_layers)
        ])
    
    def get_prefix_kv(self, layer_idx, batch_size):
        """
        Get prefix key and value for a layer
        
        Args:
            layer_idx: Which transformer layer
            batch_size: Batch size
        Returns:
            prefix_k, prefix_v: Shape (batch_size, prefix_length, d_model)
        """
        # Get prefix embeddings
        if self.prefix_proj is not None:
            prefix = self.prefix_proj(self.prefix_emb)
        else:
            prefix = self.prefix_emb
        # Shape: (prefix_length, d_model)
        
        # Project to key and value for this layer
        prefix_k = self.prefix_k_proj[layer_idx](prefix)
        prefix_v = self.prefix_v_proj[layer_idx](prefix)
        # Shape: (prefix_length, d_model)
        
        # Expand for batch
        prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1)
        prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prefix_k, prefix_v
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward with prefix tuning
        
        Modifies attention at each layer
        """
        batch_size, seq_len = input_ids.size()
        
        # Standard embedding
        hidden_states = self.base_model.transformer.wte(input_ids)
        # Shape: (batch_size, seq_len, d_model)
        
        # Process through each layer with prefix
        for layer_idx, layer in enumerate(self.base_model.transformer.h):
            # Get prefix for this layer
            prefix_k, prefix_v = self.get_prefix_kv(layer_idx, batch_size)
            # Shape: (batch_size, prefix_length, d_model)
            
            # Standard attention computation
            # This is simplified - in practice, you'd modify the attention function
            # to include prefix in key/value
            
            # For demonstration, we'll add prefix to hidden states
            # In real implementation, modify attention mechanism directly
            prefix_hidden = (prefix_k + prefix_v) / 2  # Simplified
            combined_hidden = torch.cat([prefix_hidden, hidden_states], dim=1)
            
            # Apply layer (simplified - real implementation modifies attention)
            # In practice, you need to modify the attention function to use
            # prefix_k and prefix_v in the key-value cache
            
            # For now, just pass through (this is conceptual)
            hidden_states = layer(combined_hidden)[:, self.prefix_length:, :]
        
        # Final layer norm
        hidden_states = self.base_model.transformer.ln_f(hidden_states)
        
        # Language model head
        logits = self.base_model.lm_head(hidden_states)
        
        return logits
    
    def get_num_params(self):
        """Get number of trainable parameters"""
        total = 0
        total += self.prefix_emb.numel()
        if self.prefix_proj is not None:
            total += sum(p.numel() for p in self.prefix_proj.parameters())
        total += sum(p.numel() for p in self.prefix_k_proj.parameters())
        total += sum(p.numel() for p in self.prefix_v_proj.parameters())
        return total


# ==================== COMPARISON ====================

def compare_methods(model, vocab_size=50257, d_model=768, num_layers=12, 
                    prompt_length=20):
    """
    Compare parameter counts for different methods
    """
    # Full model parameters
    # Approximate: vocab_size * d_model + layers * (attention + ffn)
    full_params = (
        vocab_size * d_model +  # Embeddings
        num_layers * (
            4 * d_model * d_model +  # Attention Q, K, V, O
            2 * d_model * (4 * d_model)  # FFN (expand 4x)
        ) +
        vocab_size * d_model  # Output projection
    )
    
    # Prompt tuning
    prompt_params = prompt_length * d_model
    
    # Prefix tuning
    prefix_params = (
        prompt_length * (d_model // 2) +  # Reparameterized prefix
        (d_model // 2) * d_model +  # Projection
        num_layers * 2 * d_model * d_model  # K and V projections per layer
    )
    
    print("Parameter Comparison:")
    print(f"Full fine-tuning: {full_params:,} parameters")
    print(f"Prompt tuning: {prompt_params:,} parameters ({prompt_params/full_params*100:.4f}%)")
    print(f"Prefix tuning: {prefix_params:,} parameters ({prefix_params/full_params*100:.4f}%)")
    print(f"\nEfficiency:")
    print(f"Prompt tuning: {full_params/prompt_params:.1f}x fewer parameters")
    print(f"Prefix tuning: {full_params/prefix_params:.1f}x fewer parameters")


# ==================== TRAINING EXAMPLES ====================

def train_prompt_tuning(prompt_model, dataloader, num_epochs=5, lr=0.3):
    """
    Train prompt tuning
    
    Only prompt embeddings are updated
    """
    optimizer = torch.optim.Adam(
        [prompt_model.prompt_embeddings],
        lr=lr
    )
    
    prompt_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward
            logits = prompt_model(input_ids)
            
            # Loss (next token prediction)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward (only updates prompt embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        print(f"Trainable params: {prompt_model.get_num_params():,}")


def train_prefix_tuning(prefix_model, dataloader, num_epochs=5, lr=2e-5):
    """
    Train prefix tuning
    
    Only prefix parameters are updated
    """
    optimizer = torch.optim.Adam(
        prefix_model.parameters(),
        lr=lr
    )
    
    prefix_model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids']
            labels = batch['labels']
            
            # Forward
            logits = prefix_model(input_ids)
            
            # Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        print(f"Trainable params: {prefix_model.get_num_params():,}")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    # Example: Compare parameter counts
    print("=" * 60)
    print("Parameter Efficiency Comparison")
    print("=" * 60)
    compare_methods(
        model=None,  # Not needed for calculation
        vocab_size=50257,
        d_model=768,
        num_layers=12,
        prompt_length=20
    )
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("1. Prompt tuning: Only trains input embeddings")
    print("2. Prefix tuning: Trains key-value at each layer")
    print("3. Both keep base model frozen")
    print("4. Much more parameter-efficient than full fine-tuning")
    print("5. Can achieve similar performance with 0.01-0.1% parameters")

