# Topic 25: Adapters & LoRA (Parameter-Efficient Fine-tuning)

## What You'll Learn

This topic teaches you parameter-efficient fine-tuning:
- Adapters
- LoRA (Low-Rank Adaptation)
- How they work
- When to use them
- Simple implementations

## Why We Need This

### Interview Importance
- **Hot topic**: LoRA is widely used
- **Efficiency**: Shows understanding of efficient training
- **Practical knowledge**: Used in production

### Real-World Application
- **Fine-tuning**: Fine-tune large models efficiently
- **Cost savings**: Much cheaper than full fine-tuning
- **Multiple tasks**: Train multiple adapters for different tasks

## Industry Use Cases

### 1. **Adapters**
**Use Case**: Task-specific fine-tuning
- Add small adapter layers
- Freeze main model
- Train only adapters

### 2. **LoRA**
**Use Case**: Most popular PEFT method
- Low-rank decomposition
- Train only small matrices
- Can combine multiple LoRAs

## Theory

### Adapters

**Concept:**
- Add small adapter layers between transformer layers
- Freeze original model weights
- Train only adapter parameters

**Architecture:**
```
Original: X → Transformer → Y
With Adapter: X → Transformer → Adapter → Y
```

**Parameters:**
- `adapter_size`: Hidden dimension of adapter (e.g., 64, 128)
- `adapter_layers`: Which layers to add adapters (all or specific)

### LoRA (Low-Rank Adaptation)

**Concept:**
- Instead of updating all weights W, update low-rank matrices
- W' = W + ΔW, where ΔW = BA (low-rank)
- Train only B and A matrices

**Mathematical Formulation:**
```
Original: h = Wx
LoRA: h = Wx + ΔWx = Wx + BAx

Where:
- W: Original weight matrix (d × d)
- B: Low-rank matrix (d × r), r << d
- A: Low-rank matrix (r × d)
- r: Rank (typically 1-16)
```

**Why it works:**
- Low-rank assumption: Weight updates have low intrinsic rank
- Much fewer parameters: r × d instead of d × d
- Example: d=4096, r=8 → 8×4096×2 = 65K params vs 16M params

**Parameters:**
- `rank` (r): Rank of decomposition (1-16, typically 8)
- `alpha`: Scaling factor (usually = rank)
- `target_modules`: Which layers to apply LoRA (attention, MLP, etc.)
- `dropout`: Dropout in LoRA layers

## Industry-Standard Boilerplate Code

### Adapter Implementation

```python
"""
Adapter: Small trainable layers
"""
import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    Adapter layer
    
    Architecture:
    - Down projection: d → adapter_size
    - Activation: ReLU
    - Up projection: adapter_size → d
    - Residual connection
    """
    
    def __init__(self, d_model: int, adapter_size: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, adapter_size)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_size, d_model)
    
    def forward(self, x):
        # Adapter: down → activation → up
        adapter_out = self.up_proj(self.activation(self.down_proj(x)))
        # Residual connection
        return x + adapter_out
```

### LoRA Implementation

```python
"""
LoRA: Low-Rank Adaptation
"""
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA layer
    
    W' = W + BA
    Where B (d × r) and A (r × d) are trainable
    """
    
    def __init__(self, d_model: int, rank: int = 8, alpha: int = 8):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(rank, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(d_model, rank))
        
        # Scaling factor
        self.scale = alpha / rank
    
    def forward(self, x, W):
        """
        Forward pass
        
        Args:
            x: Input (batch, seq_len, d_model)
            W: Original weight matrix (frozen)
        """
        # Original: Wx
        original_out = x @ W.T
        
        # LoRA: BAx
        lora_out = (x @ self.A.T) @ self.B.T
        lora_out = lora_out * self.scale
        
        # Combined: Wx + BAx
        return original_out + lora_out

class LoRALinear(nn.Module):
    """
    LoRA Linear layer (complete implementation)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.weight.requires_grad = False  # Freeze
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scale = alpha / rank
    
    def forward(self, x):
        # Original: xW^T
        out = x @ self.weight.T
        
        # LoRA: xA^TB^T * scale
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        lora_out = lora_out * self.scale
        
        return out + lora_out
```

## Parameter Comparison

### Full Fine-tuning
- Parameters: All model parameters (7B model = 7B params)
- Memory: High (gradients + optimizer states)
- Time: Slow

### Adapters
- Parameters: ~0.1-1% of model (7B model = 7-70M params)
- Memory: Medium
- Time: Medium

### LoRA
- Parameters: ~0.01-0.1% of model (7B model = 0.7-7M params)
- Memory: Low
- Time: Fast

## When to Use

### Use Adapters When:
- Need task-specific layers
- Want modular design
- Multiple adapters for different tasks

### Use LoRA When:
- Want maximum efficiency
- Need to combine multiple LoRAs
- Limited compute resources

## Exercises

1. Implement adapter layer
2. Implement LoRA layer
3. Compare parameter counts
4. Fine-tune with LoRA

## Next Steps

- **Topic 26**: Tree-based methods
- Review parameter-efficient methods

