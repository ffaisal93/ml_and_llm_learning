# Topic 5: Attention Mechanisms

## What You'll Learn

This topic teaches you different attention mechanisms:
- Self-attention
- Cross-attention
- Scaled dot-product attention
- Sparse attention
- Longformer/BigBird attention
- What problems each solves

## Why We Need This

### Interview Importance
- **Common question**: "Explain different attention mechanisms"
- **Problem-solving**: Know which attention to use when
- **Understanding**: Deep understanding of attention

### Real-World Application
- **Long context**: Sparse attention for long sequences
- **Efficiency**: Different attentions have different costs
- **Specialized tasks**: Different tasks need different attention

## Industry Use Cases

### 1. **Self-Attention**
**Use Case**: BERT, GPT
- Language understanding
- Text generation
- Standard transformer attention

### 2. **Sparse Attention**
**Use Case**: Longformer, BigBird
- Long documents
- Efficient long-context processing
- Reduces quadratic complexity

### 3. **Cross-Attention**
**Use Case**: Encoder-decoder models
- Translation
- Question answering
- Cross-modal tasks

## Industry-Standard Boilerplate Code

### Self-Attention (Standard)

```python
"""
Self-Attention
Standard attention used in transformers
"""
import numpy as np

def self_attention(Q, K, V, d_k, mask=None):
    """Standard self-attention"""
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    attention_weights = softmax(scores)
    return attention_weights @ V
```

### Causal Attention (GPT-style)

```python
"""
Causal Attention
Masks future positions (for autoregressive generation)
"""
def causal_attention(Q, K, V, d_k):
    """Causal attention with lower triangular mask"""
    seq_len = Q.shape[0]
    # Create lower triangular mask
    mask = np.tril(np.ones((seq_len, seq_len)))
    return self_attention(Q, K, V, d_k, mask=mask)
```

### Sparse Attention (Longformer-style)

```python
"""
Sparse Attention
Only attends to local + global positions
Reduces O(n²) to O(n)
"""
def sparse_attention(Q, K, V, d_k, window_size=512, global_indices=None):
    """
    Sparse attention: local window + global tokens
    
    Args:
        window_size: Local attention window
        global_indices: Positions that attend to all (e.g., [CLS] token)
    """
    seq_len = Q.shape[0]
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Create sparse mask
    mask = np.zeros((seq_len, seq_len))
    
    # Local attention (sliding window)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        mask[i, start:end] = 1
    
    # Global attention
    if global_indices:
        for idx in global_indices:
            mask[idx, :] = 1  # Attend to all
            mask[:, idx] = 1  # All attend to this
    
    # Apply mask
    scores = np.where(mask == 1, scores, -1e9)
    attention_weights = softmax(scores)
    return attention_weights @ V
```

### Cross-Attention

```python
"""
Cross-Attention
Query from one sequence, Key/Value from another
Used in encoder-decoder architectures
"""
def cross_attention(Q_encoder, K_decoder, V_decoder, d_k):
    """
    Cross-attention: Q from encoder, K/V from decoder
    
    Args:
        Q_encoder: Queries from encoder (encoder_len, d_k)
        K_decoder: Keys from decoder (decoder_len, d_k)
        V_decoder: Values from decoder (decoder_len, d_v)
    """
    scores = Q_encoder @ K_decoder.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    return attention_weights @ V_decoder
```

## What Problems They Solve

### Self-Attention
- **Problem**: Need to relate all positions
- **Solution**: Every position attends to every position
- **Cost**: O(n²)

### Causal Attention
- **Problem**: Autoregressive generation (can't see future)
- **Solution**: Mask future positions
- **Use**: GPT, language models

### Sparse Attention
- **Problem**: O(n²) too expensive for long sequences
- **Solution**: Only attend to local + few global positions
- **Use**: Longformer, BigBird, long documents

### Cross-Attention
- **Problem**: Need to relate two sequences
- **Solution**: Query from one, Key/Value from other
- **Use**: Translation, encoder-decoder

## Theory

### Attention Complexity

| Type | Complexity | Use Case |
|------|-----------|----------|
| Self-attention | O(n²) | Standard transformers |
| Sparse attention | O(n) | Long sequences |
| Local attention | O(n×w) | Fixed window size w |

### When to Use Which

- **Self-attention**: Standard, short sequences
- **Causal attention**: Autoregressive generation
- **Sparse attention**: Long sequences (>2048 tokens)
- **Cross-attention**: Encoder-decoder tasks

## Exercises

1. Implement causal mask
2. Implement sparse attention
3. Compare attention patterns
4. Measure computational cost

## Next Steps

- **Topic 6**: LLM inference techniques
- **Topic 7**: LLM problem solving

