# Topic 14: Advanced Positional Embeddings (RoPE, ALiBi)

> 🔥 **For interviews, read these first:**
> - **`POSITIONAL_DEEP_DIVE.md`** — frontier-lab interview deep dive: sinusoidal/learned/T5-bias/RoPE/ALiBi/NoPE, full RoPE derivation showing relative-position from rotated dot products, NTK scaling, YaRN, length extrapolation.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

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

## Core Intuition

Transformers need positional information because self-attention alone does not know token order.

Basic sinusoidal encodings work, but modern LLMs often use alternatives because positional handling affects:
- long-context behavior
- extrapolation beyond training length
- implementation simplicity
- inductive bias about relative distance

### RoPE

RoPE injects position by rotating query and key vectors.

The important intuition is:
- position affects how queries and keys line up
- relative offsets emerge naturally from the rotation structure

That is why RoPE is often described as giving relative position behavior inside the attention computation itself.

### ALiBi

ALiBi does not add a positional vector to token embeddings.

Instead, it changes attention scores directly using a distance-dependent bias.

Intuition:
- farther positions receive a stronger penalty
- attention naturally prefers closer positions unless evidence is strong

## Technical Details Interviewers Often Want

### Why Modern LLMs Often Prefer RoPE

RoPE is attractive because it:
- integrates with attention cleanly
- captures relative position effects
- often extrapolates better than simple absolute schemes

### Why ALiBi Is Interesting

ALiBi is interesting because it is simple:
- no learned positional embedding table
- no explicit sinusoidal addition
- just score biasing based on relative distance

This simplicity is part of why it is a common interview discussion point.

### Positional Method Trade-Off

The real interview answer is not "RoPE is better."

It is:
- different methods encode order differently
- some favor relative structure
- some extrapolate better
- some are simpler to implement or reason about

## Common Failure Modes

- describing RoPE as if it were just another additive embedding
- ignoring that positional methods affect extrapolation behavior
- assuming better long-context extrapolation on paper always means better end-task behavior
- forgetting that context scaling tricks can interact with positional encoding choices

## Edge Cases and Follow-Up Questions

1. Why does self-attention need position information at all?
2. Why is RoPE often described as relative rather than purely absolute?
3. Why can positional encoding choice matter for long-context extrapolation?
4. How is ALiBi different from adding learned position embeddings?
5. Why is "works for longer context" not the same as "works better on all long-context tasks"?

## What to Practice Saying Out Loud

1. Why RoPE is useful in modern LLMs
2. How ALiBi changes attention without explicit positional embeddings
3. Why positional encoding choice is really an inductive-bias choice

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
