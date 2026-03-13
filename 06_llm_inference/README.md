# Topic 6: LLM Inference Techniques

## What You'll Learn

This topic teaches you LLM inference optimization:
- KV caching
- Quantization (INT8, INT4)
- Speculative decoding
- Continuous batching
- Memory optimization

## Why We Need This

### Interview Importance
- **Common question**: "How does KV caching work?"
- **Optimization**: Critical for production
- **Understanding**: Know how inference is optimized

### Real-World Application
- **Production serving**: Need fast inference
- **Cost reduction**: Optimization saves money
- **User experience**: Faster = better UX

## Industry Use Cases

### 1. **KV Caching**
**Use Case**: All LLM inference
- Avoid recomputing attention
- 10-100x speedup
- Essential for generation

### 2. **Quantization**
**Use Case**: Production serving
- Reduce memory 2-8x
- Faster inference
- Trade accuracy for speed

### 3. **Speculative Decoding**
**Use Case**: High-throughput serving
- Generate multiple tokens
- Verify with main model
- Faster generation

## Industry-Standard Boilerplate Code

### KV Cache Implementation

```python
"""
KV Cache from Scratch
Interview question: "How does KV caching work?"
"""
import numpy as np
from typing import Dict, Optional

class KVCache:
    """
    KV Cache stores Key and Value matrices to avoid recomputation
    Critical optimization for autoregressive generation
    """
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache: Dict[int, Dict[str, np.ndarray]] = {}
    
    def initialize(self, layer_idx: int):
        """Initialize cache for a layer"""
        self.cache[layer_idx] = {
            'keys': [],
            'values': []
        }
    
    def update(self, layer_idx: int, keys: np.ndarray, values: np.ndarray):
        """
        Update cache with new keys and values
        
        Args:
            layer_idx: Which transformer layer
            keys: New key vectors (num_heads, head_dim)
            values: New value vectors (num_heads, head_dim)
        """
        if layer_idx not in self.cache:
            self.initialize(layer_idx)
        
        self.cache[layer_idx]['keys'].append(keys)
        self.cache[layer_idx]['values'].append(values)
    
    def get(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Get cached keys and values for a layer"""
        if layer_idx not in self.cache:
            return {'keys': None, 'values': None}
        
        # Concatenate all cached keys/values
        keys = np.concatenate(self.cache[layer_idx]['keys'], axis=0)
        values = np.concatenate(self.cache[layer_idx]['values'], axis=0)
        
        return {'keys': keys, 'values': values}
    
    def clear(self):
        """Clear cache"""
        self.cache = {}


def attention_with_kv_cache(Q: np.ndarray, K_cache: np.ndarray, 
                           V_cache: np.ndarray, K_new: np.ndarray,
                           V_new: np.ndarray, d_k: int) -> np.ndarray:
    """
    Attention with KV cache
    Only compute attention for new token, reuse cached K/V
    
    Args:
        Q: Query for new token (1, d_k)
        K_cache: Cached keys (seq_len-1, d_k)
        V_cache: Cached values (seq_len-1, d_v)
        K_new: New key (1, d_k)
        V_new: New value (1, d_v)
        d_k: Key dimension
    """
    # Concatenate cached + new
    K = np.concatenate([K_cache, K_new], axis=0)
    V = np.concatenate([V_cache, V_new], axis=0)
    
    # Compute attention (only Q is new)
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = attention_weights @ V
    
    return output
```

### Quantization (Simple)

```python
"""
Quantization from Scratch
Reduce model precision to save memory and speed
"""
import numpy as np

def quantize_to_int8(weights: np.ndarray) -> tuple:
    """
    Quantize FP32 weights to INT8
    
    Returns:
        (quantized_weights, scale, zero_point)
    """
    # Find range
    w_min = np.min(weights)
    w_max = np.max(weights)
    
    # Calculate scale and zero point
    scale = (w_max - w_min) / 255.0
    zero_point = -w_min / scale
    
    # Quantize
    quantized = np.round(weights / scale + zero_point).astype(np.int8)
    quantized = np.clip(quantized, -128, 127)
    
    return quantized, scale, zero_point

def dequantize_from_int8(quantized: np.ndarray, scale: float, 
                         zero_point: float) -> np.ndarray:
    """Dequantize INT8 back to FP32"""
    return (quantized.astype(np.float32) - zero_point) * scale
```

## Theory

### KV Caching
- **Problem**: Recompute attention for all previous tokens each step
- **Solution**: Cache K and V, only compute for new token
- **Speedup**: 10-100x for generation
- **Memory**: Trade memory for speed

### Quantization
- **Problem**: Models too large, inference slow
- **Solution**: Reduce precision (FP32 → INT8 → INT4)
- **Trade-off**: Accuracy vs speed/memory
- **Use**: Production when speed/memory critical

## Exercises

1. Implement KV cache
2. Measure speedup from caching
3. Implement quantization
4. Compare quantized vs full precision

## KV Cache Detailed Explanation

**New Comprehensive Guides:**

- **`kv_cache_detailed.md`**: Complete detailed explanation
  - The problem with standard inference (redundancy)
  - How KV cache solves it (step-by-step)
  - Code-level comparison (standard vs KV cache)
  - Computational complexity analysis
  - Memory considerations
  - Practical implementation details

- **`kv_cache_comparison.py`**: Side-by-side code comparison
  - Standard inference implementation (shows the problem)
  - KV cache implementation (shows the solution)
  - Step-by-step comparison showing exactly what changes
  - The key code difference highlighted

**Key Improvements:**
- **Standard**: O(n³d) total, recomputes all K, V every step
- **KV Cache**: O(n²d) total, only computes K, V for new token
- **Speedup**: ~n× for sequences of length n

**Code Changes:**
- Standard: `input_ids = entire_sequence`, recomputes all
- KV Cache: `input_ids = [new_token]`, reuses cache
- Key operation: `concatenate([K_cache, K_new])` - reuses cached values!

**The Key Code:**
```python
# Standard (without cache):
K = compute_K([token_0, ..., token_i])  # Recomputes all

# KV Cache (with cache):
K_new = compute_K([token_i])  # Only new token
K = concatenate([K_cache, K_new])  # Reuses cache!
```

## Next Steps

- **Topic 7**: LLM problem solving
- **Topic 8**: Training techniques

