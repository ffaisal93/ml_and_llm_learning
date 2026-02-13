# Topic 14: Advanced Positional Embeddings (RoPE, ALiBi)

## What You'll Learn

This topic teaches you advanced positional embedding methods:
- RoPE (Rotary Position Embedding)
- ALiBi (Attention with Linear Biases)
- Why they're better than sinusoidal
- Simple implementations

## Why We Need This

### Interview Importance
- **Hot topic**: RoPE used in LLaMA, GPT-NeoX
- **Recent advancement**: Shows you know latest techniques
- **Understanding**: Better than basic positional encoding

### Real-World Application
- **LLaMA**: Uses RoPE
- **Long context**: RoPE generalizes to longer sequences
- **Extrapolation**: Better than sinusoidal for longer sequences

## Industry Use Cases

### 1. **RoPE (Rotary Position Embedding)**
**Use Case**: LLaMA, GPT-NeoX, PaLM
- Better extrapolation to longer sequences
- Relative position encoding
- Used in modern LLMs

### 2. **ALiBi (Attention with Linear Biases)**
**Use Case**: Alternative to positional embeddings
- No positional embeddings needed
- Add bias to attention scores
- Works well for long sequences

## Industry-Standard Boilerplate Code

### RoPE Implementation

```python
"""
RoPE: Rotary Position Embedding
Interview question: "Implement RoPE"
"""
import numpy as np

def apply_rope(q: np.ndarray, k: np.ndarray, position: int, 
               dim: int, base: float = 10000.0) -> tuple:
    """
    Apply RoPE to query and key
    
    RoPE rotates Q and K by position-dependent angles
    This encodes relative position information
    
    Args:
        q: Query vector (d_model,)
        k: Key vector (d_model,)
        position: Current position
        dim: Dimension of model
        base: Base for frequency calculation
    """
    # Create rotation angles
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
    angles = position * inv_freq
    
    # Split into pairs for rotation
    q_rot = np.zeros_like(q)
    k_rot = np.zeros_like(k)
    
    # Apply rotation to pairs
    for i in range(0, dim, 2):
        if i + 1 < dim:
            cos_angle = np.cos(angles[i // 2])
            sin_angle = np.sin(angles[i // 2])
            
            # Rotate pair
            q_rot[i] = q[i] * cos_angle - q[i+1] * sin_angle
            q_rot[i+1] = q[i] * sin_angle + q[i+1] * cos_angle
            
            k_rot[i] = k[i] * cos_angle - k[i+1] * sin_angle
            k_rot[i+1] = k[i] * sin_angle + k[i+1] * cos_angle
        else:
            # Odd dimension, no rotation
            q_rot[i] = q[i]
            k_rot[i] = k[i]
    
    return q_rot, k_rot
```

### ALiBi Implementation

```python
"""
ALiBi: Attention with Linear Biases
No positional embeddings, just add bias to attention scores
"""
import numpy as np

def alibi_attention_bias(seq_len: int, num_heads: int) -> np.ndarray:
    """
    Generate ALiBi bias matrix
    
    Adds linear bias based on distance between positions
    Closer positions get less negative bias
    
    Args:
        seq_len: Sequence length
        num_heads: Number of attention heads
    Returns:
        Bias matrix (num_heads, seq_len, seq_len)
    """
    # Different slopes for different heads
    slopes = np.array([2 ** (-(2 ** -(np.log2(8) - 3)) * i) 
                       for i in range(1, num_heads + 1)])
    
    bias = np.zeros((num_heads, seq_len, seq_len))
    
    for head in range(num_heads):
        for i in range(seq_len):
            for j in range(seq_len):
                # Linear bias based on distance
                distance = j - i
                bias[head, i, j] = -slopes[head] * abs(distance)
    
    return bias
```

## Theory

### Why RoPE is Better

**Sinusoidal Problems:**
- Fixed maximum length
- Doesn't extrapolate well
- Absolute positions only

**RoPE Advantages:**
- Relative position encoding
- Better extrapolation
- Used in modern LLMs (LLaMA)

### How RoPE Works
- Rotates Q and K by position-dependent angles
- Relative positions encoded in rotation
- Generalizes to longer sequences

## Exercises

1. Implement RoPE
2. Compare with sinusoidal
3. Test extrapolation
4. Implement ALiBi

## Next Steps

- **Topic 15**: Tokenization methods
- **Topic 16**: Training behaviors

