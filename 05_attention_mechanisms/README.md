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

**What This Code Does:**

**Step 1: Get sequence length**
```python
seq_len = Q.shape[0]  # Number of tokens
```

**Step 2: Create lower triangular mask**
```python
mask = np.tril(np.ones((seq_len, seq_len)))
```

**What happens:**
- `np.ones((seq_len, seq_len))` creates matrix of all 1s
- `np.tril()` keeps only lower triangular part (sets upper to 0)
- Result: Lower triangular matrix where:
  - `mask[i, j] = 1` if j ≤ i (can attend to past/current)
  - `mask[i, j] = 0` if j > i (cannot attend to future)

**Example for seq_len=4:**
```
[[1, 0, 0, 0],   ← Position 0: can only see itself
 [1, 1, 0, 0],   ← Position 1: can see 0, 1
 [1, 1, 1, 0],   ← Position 2: can see 0, 1, 2
 [1, 1, 1, 1]]   ← Position 3: can see all (0, 1, 2, 3)
```

**Step 3: Apply mask in attention**
```python
return self_attention(Q, K, V, d_k, mask=mask)
```

**Inside self_attention:**
- Computes attention scores: `scores = Q @ K.T / √d_k`
- Applies mask: `scores[mask == 0] = -∞` (future positions)
- After softmax: Future positions get 0 attention weight
- Result: Each position only attends to past and current tokens

**Why Lower Triangular?**
- Lower triangular = can attend to positions ≤ current (past + current)
- Upper triangular = wrong (would allow future, block past)
- This enforces causal constraint for autoregressive generation

**See `causal_attention_detailed.md` for complete explanation!**

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

**Detailed Analysis:** See `attention_complexity.md` for complete complexity breakdown!

| Type | Time Complexity | Space Complexity | Use Case |
|------|----------------|------------------|----------|
| Self-attention | O(n²d) | O(n²) | Standard transformers |
| Multi-head | O(n²d) | O(n²) | GPT, parallelizable |
| Linear | O(nd²) | O(nd) | Very long sequences (n >> d) |
| Sparse (Longformer) | O(n√n d) | O(n√n) | Long sequences |
| Sparse (BigBird) | O(n log n d) | O(n log n) | Very long sequences |
| Flash Attention | O(n²d) | O(n) | Memory-constrained training |

**Key Insight:** Standard attention is O(n²d) because it computes pairwise relationships between all n tokens, with each computation involving d-dimensional vectors. The n² term comes from the attention matrix (n×n), and the d term comes from the vector dimension.

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

## Causal Attention: Detailed Explanation

**New Comprehensive Guide:**

- **`causal_attention_detailed.md`**: Complete theoretical explanation
  - Why we need causal attention (autoregressive constraint)
  - How causal attention works (lower triangular mask)
  - Step-by-step code explanation
  - Visual examples
  - Why lower triangular (not upper)
  - Comparison with/without mask
  - Common mistakes and pitfalls
  - Advanced topics

- **`causal_attention_code.py`**: Complete implementation with visualization
  - Step-by-step visualization
  - Comparison with/without mask
  - Explanation of lower triangular
  - Interactive examples

**Key Concepts:**
- Lower triangular mask: `np.tril(np.ones((seq_len, seq_len)))`
- Sets future positions to -∞ in attention scores
- After softmax: Future positions get 0 attention weight
- Enforces: Each position can only see past and current tokens
- Critical for autoregressive models like GPT

## Advanced Attention Mechanisms

**New Comprehensive Content:**

- **`advanced_attention_mechanisms.md`**: Complete theoretical guide
  - Multi-Head Attention (MHA) - baseline
  - Multi-Query Attention (MQA) - shares K, V across all heads
  - Group Query Attention (GQA) - shares K, V within groups
  - Paged Attention - memory-efficient cache management
  - Detailed comparisons and trade-offs
  - Real-world usage and examples

- **`advanced_attention_code.py`**: Complete implementations
  - MultiQueryAttention class
  - GroupQueryAttention class
  - PagedKVCache class (conceptual)
  - Comparison utilities
  - Memory analysis

**Key Concepts:**

**Multi-Query Attention (MQA):**
- Shares K, V across all heads
- KV Cache: seq_len × (d_k + d_v) (not per head!)
- Reduction: num_heads× (e.g., 32× for 32 heads)

**Group Query Attention (GQA):**
- Shares K, V within groups of heads
- KV Cache: num_groups × seq_len × (d_k + d_v)
- Reduction: (num_heads / num_groups)× (e.g., 4× for 32 heads, 8 groups)
- Recommended for production (best balance)

**Paged Attention:**
- Manages KV cache in non-contiguous pages
- Eliminates memory fragmentation
- 95%+ memory utilization (vs ~70% standard)
- Core of vLLM's efficiency

**Note on "Multi-Head Latent Attention":**
- Not a standard term in literature
- Related concepts exist (latent variables, low-rank attention)
- Mostly research topics, not widely deployed
- Production systems typically use GQA, MQA, or standard MHA

## Next Steps

- **Topic 6**: LLM inference techniques
- **Topic 7**: LLM problem solving

