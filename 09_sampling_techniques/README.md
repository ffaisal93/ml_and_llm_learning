# Topic 9: Sampling Techniques

> 🔥 **For interviews, read these first:**
> - **`SAMPLING_DEEP_DIVE.md`** — frontier-lab interview deep dive: greedy/beam/temperature/top-k/top-p/min-p/typical/Mirostat/penalties, why beam search fails for LLMs, speculative decoding, best-of-N for test-time scaling.
> - **`INTERVIEW_GRILL.md`** — 45 active-recall questions.

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

## Core Intuition

Sampling is the step where model probabilities become actual generated tokens.

That means decoding controls the model's behavior a lot more than many people first realize.

Even with the same model:
- greedy decoding may look repetitive
- high temperature may look creative but unstable
- top-p may feel more natural than top-k

So decoding is not just a post-processing detail. It is part of system behavior.

### Greedy Decoding

Greedy decoding always takes the highest-probability token.

That makes it:
- deterministic
- simple
- often too repetitive or myopic

### Top-k Sampling

Top-k keeps only the `k` most likely options and samples from them.

This gives some diversity while preventing very low-probability tokens from being chosen.

### Top-p Sampling

Top-p keeps the smallest set of tokens whose cumulative probability mass exceeds `p`.

This is adaptive:
- if the distribution is sharp, the candidate set stays small
- if the distribution is broad, the candidate set can grow

That is why top-p often feels more natural than fixed top-k.

### Temperature

Temperature reshapes the distribution before sampling.

- low temperature sharpens the distribution
- high temperature flattens it

That means temperature is not choosing tokens by itself. It changes the probability landscape first.

## Technical Details Interviewers Often Want

### Why Greedy Can Be Weak

Greedy decoding is locally optimal, not globally optimal for quality or diversity.

It can:
- lock into repetitive loops
- over-commit early
- miss good but slightly lower-probability branches

### Top-k vs Top-p

This is a classic follow-up.

- **Top-k**: fixed candidate count
- **Top-p**: variable candidate count based on probability mass

Top-p adapts to uncertainty better, which is why it is common in LLM products.

### Temperature Edge Cases

If temperature is very low:
- output approaches greedy decoding

If temperature is very high:
- the distribution becomes too flat
- low-quality tokens become more likely

### Beam Search

Beam search is different from random sampling.

It tries to keep multiple high-probability partial sequences alive, which is useful in structured tasks like translation but not always ideal for open-ended chat generation.

## Common Failure Modes

- high temperature causing nonsense generations
- greedy decoding causing repetition
- forgetting to renormalize probabilities after top-k or top-p filtering
- claiming one sampling method is always best
- using beam search for tasks where diversity matters more than likelihood

## Edge Cases and Follow-Up Questions

1. Why is top-p often preferred over top-k?
2. Why can greedy decoding be repetitive?
3. What happens when temperature approaches zero?
4. Why does beam search often look better for translation than for creative chat?
5. Why must probabilities be renormalized after filtering?

## What to Practice Saying Out Loud

1. Why decoding strategy changes behavior even with the same model
2. The difference between temperature and top-p
3. Why "more randomness" is not the same as "better creativity"

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
