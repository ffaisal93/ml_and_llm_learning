"""
KV Cache from Scratch
Interview question: "How does KV caching work?"
"""
import numpy as np
from typing import Dict, List, Optional

class KVCache:
    """
    KV Cache stores Key and Value matrices to avoid recomputation
    Critical optimization for autoregressive generation
    
    How it works:
    1. First token: Compute full Q, K, V, cache K and V
    2. Next tokens: Only compute Q for new token, reuse cached K, V
    3. Result: Much faster generation
    """
    
    def __init__(self, num_layers: int, num_heads: int, head_dim: int):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Cache structure: {layer_idx: {'keys': [...], 'values': [...]}}
        self.cache: Dict[int, Dict[str, List[np.ndarray]]] = {}
    
    def initialize_layer(self, layer_idx: int):
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
            keys: New key vectors (num_heads, head_dim) or (1, num_heads, head_dim)
            values: New value vectors (num_heads, head_dim) or (1, num_heads, head_dim)
        """
        if layer_idx not in self.cache:
            self.initialize_layer(layer_idx)
        
        # Handle different input shapes
        if keys.ndim == 2:
            keys = keys[np.newaxis, :, :]  # Add batch dimension
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
        
        self.cache[layer_idx]['keys'].append(keys)
        self.cache[layer_idx]['values'].append(values)
    
    def get(self, layer_idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Get cached keys and values for a layer"""
        if layer_idx not in self.cache or len(self.cache[layer_idx]['keys']) == 0:
            return None
        
        # Concatenate all cached keys/values along sequence dimension
        keys = np.concatenate(self.cache[layer_idx]['keys'], axis=0)
        values = np.concatenate(self.cache[layer_idx]['values'], axis=0)
        
        return {'keys': keys, 'values': values}
    
    def clear(self):
        """Clear cache (start new sequence)"""
        self.cache = {}
    
    def get_cache_size(self) -> int:
        """Get total cache size in elements"""
        total = 0
        for layer_idx in self.cache:
            for key in ['keys', 'values']:
                for item in self.cache[layer_idx][key]:
                    total += item.size
        return total


def attention_with_kv_cache(Q: np.ndarray, K_cache: Optional[np.ndarray],
                            V_cache: Optional[np.ndarray], K_new: np.ndarray,
                            V_new: np.ndarray, d_k: int) -> np.ndarray:
    """
    Attention with KV cache
    Only compute attention for new token, reuse cached K/V
    
    This is the key optimization: instead of recomputing attention
    for all previous tokens, we reuse cached K and V.
    
    Args:
        Q: Query for new token (num_heads, head_dim)
        K_cache: Cached keys (seq_len-1, num_heads, head_dim) or None
        V_cache: Cached values (seq_len-1, num_heads, head_dim) or None
        K_new: New key (num_heads, head_dim)
        V_new: New value (num_heads, head_dim)
        d_k: Key dimension
    """
    # If no cache, this is first token
    if K_cache is None:
        K = K_new[np.newaxis, :, :]  # (1, num_heads, head_dim)
        V = V_new[np.newaxis, :, :]
    else:
        # Concatenate cached + new
        K = np.concatenate([K_cache, K_new[np.newaxis, :, :]], axis=0)
        V = np.concatenate([V_cache, V_new[np.newaxis, :, :]], axis=0)
    
    # Compute attention scores
    # Q: (num_heads, head_dim), K: (seq_len, num_heads, head_dim)
    # We need to compute Q @ K^T for each head
    seq_len = K.shape[0]
    num_heads = Q.shape[0]
    
    scores = np.zeros((num_heads, seq_len))
    for head in range(num_heads):
        scores[head] = Q[head] @ K[:, head, :].T / np.sqrt(d_k)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Apply to values
    output = np.zeros_like(Q)
    for head in range(num_heads):
        output[head] = attention_weights[head] @ V[:, head, :]
    
    return output


# Usage Example
if __name__ == "__main__":
    print("KV Cache Example")
    print("=" * 60)
    
    # Initialize cache
    num_layers = 12
    num_heads = 8
    head_dim = 64
    
    cache = KVCache(num_layers, num_heads, head_dim)
    
    # Simulate generation: 3 tokens
    for token_idx in range(3):
        layer_idx = 0  # First layer
        
        # Generate random K, V for this token
        K_new = np.random.randn(num_heads, head_dim)
        V_new = np.random.randn(num_heads, head_dim)
        
        # Get cached K, V
        cached = cache.get(layer_idx)
        
        if cached is None:
            print(f"Token {token_idx}: First token, no cache")
        else:
            print(f"Token {token_idx}: Using cache with {len(cache.cache[layer_idx]['keys'])} previous tokens")
        
        # Compute attention (simplified - would use Q in real implementation)
        Q = np.random.randn(num_heads, head_dim)
        output = attention_with_kv_cache(
            Q, 
            cached['keys'] if cached else None,
            cached['values'] if cached else None,
            K_new, V_new, head_dim
        )
        
        # Update cache
        cache.update(layer_idx, K_new, V_new)
    
    print(f"\nCache size: {cache.get_cache_size()} elements")
    print(f"Memory saved: Instead of recomputing, we reuse cached K/V")

