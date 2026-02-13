# Topic 9: Sampling Techniques

## What You'll Learn

This topic teaches you text generation sampling:
- Greedy decoding
- Top-k sampling
- Top-p (nucleus) sampling
- Temperature sampling
- Beam search
- Implementations from scratch

## Why We Need This

### Interview Importance
- **Common question**: "Implement top-p sampling"
- **Understanding**: Know how LLMs generate text
- **Application**: Choose right sampling for task

### Real-World Application
- **Text generation**: All LLMs use sampling
- **Quality control**: Sampling affects output quality
- **Creativity vs determinism**: Trade-off

## Industry Use Cases

### 1. **Greedy Decoding**
**Use Case**: Deterministic tasks
- Code generation
- Translation
- When you want same output

### 2. **Top-p Sampling**
**Use Case**: Most common
- ChatGPT, Claude
- Balanced creativity/quality
- Default for many models

### 3. **Temperature Sampling**
**Use Case**: Control creativity
- Low temp = more deterministic
- High temp = more creative
- Adjustable per use case

## Industry-Standard Boilerplate Code

### Greedy Decoding

```python
"""
Greedy Decoding
Always pick most likely token
Deterministic but can be repetitive
"""
import numpy as np

def greedy_decode(logits: np.ndarray) -> int:
    """
    Greedy: Pick token with highest probability
    
    Args:
        logits: (vocab_size,) unnormalized scores
    Returns:
        token_id: Most likely token
    """
    return np.argmax(logits)
```

### Top-k Sampling

```python
"""
Top-k Sampling
Sample from top k most likely tokens
"""
import numpy as np

def top_k_sampling(logits: np.ndarray, k: int = 50) -> int:
    """
    Top-k: Sample from top k tokens
    
    Args:
        logits: (vocab_size,) unnormalized scores
        k: Number of top tokens to consider
    Returns:
        token_id: Sampled token
    """
    # Get top k indices
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Softmax over top k
    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample
    sampled_idx = np.random.choice(len(top_k_indices), p=probs)
    return top_k_indices[sampled_idx]
```

### Top-p (Nucleus) Sampling

```python
"""
Top-p (Nucleus) Sampling
Sample from smallest set of tokens with cumulative probability >= p
Most popular method
"""
import numpy as np

def top_p_sampling(logits: np.ndarray, p: float = 0.9) -> int:
    """
    Top-p: Sample from tokens whose cumulative probability >= p
    
    Args:
        logits: (vocab_size,) unnormalized scores
        p: Nucleus probability threshold (0.0 to 1.0)
    Returns:
        token_id: Sampled token
    """
    # Sort logits descending
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    # Softmax
    exp_logits = np.exp(sorted_logits - np.max(sorted_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Cumulative probability
    cum_probs = np.cumsum(probs)
    
    # Find smallest set with cum_prob >= p
    nucleus_size = np.searchsorted(cum_probs, p) + 1
    nucleus_size = min(nucleus_size, len(probs))
    
    # Sample from nucleus
    nucleus_probs = probs[:nucleus_size]
    nucleus_probs = nucleus_probs / np.sum(nucleus_probs)  # Renormalize
    
    sampled_idx = np.random.choice(nucleus_size, p=nucleus_probs)
    return sorted_indices[sampled_idx]
```

### Temperature Sampling

```python
"""
Temperature Sampling
Control randomness by scaling logits
"""
import numpy as np

def temperature_sampling(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Temperature: Scale logits before softmax
    
    Args:
        logits: (vocab_size,) unnormalized scores
        temperature: 
            - < 1.0: More deterministic (sharp distribution)
            - = 1.0: Normal
            - > 1.0: More random (flat distribution)
    Returns:
        token_id: Sampled token
    """
    # Scale by temperature
    scaled_logits = logits / temperature
    
    # Softmax
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Sample
    return np.random.choice(len(logits), p=probs)
```

### Combined: Top-p + Temperature

```python
"""
Combined Sampling: Top-p + Temperature
Most common in practice (ChatGPT, Claude)
"""
def sample_token(logits: np.ndarray, 
                 temperature: float = 1.0,
                 top_p: float = 0.9) -> int:
    """
    Combined: Apply temperature, then top-p
    
    This is what most production LLMs use
    """
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Then top-p
    return top_p_sampling(scaled_logits, top_p)
```

## Theory

### Sampling Comparison

| Method | Determinism | Creativity | Use Case |
|--------|------------|------------|----------|
| Greedy | Very High | Very Low | Code, translation |
| Top-k | Medium | Medium | General purpose |
| Top-p | Medium | Medium | **Most common** |
| Temperature | Adjustable | Adjustable | Control creativity |

### When to Use Which

- **Greedy**: Need deterministic output
- **Top-k**: Simple, works well
- **Top-p**: **Default choice**, adaptive
- **Temperature**: Fine-tune creativity

## Exercises

1. Implement all sampling methods
2. Compare outputs
3. Tune temperature
4. Combine methods

## Next Steps

- **Topic 10**: Optimizers
- **Topic 11**: Regularization

