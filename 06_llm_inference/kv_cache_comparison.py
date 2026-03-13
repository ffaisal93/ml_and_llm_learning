"""
KV Cache: Side-by-Side Comparison
Shows exactly what changes between standard and KV cache inference
"""
import numpy as np
from typing import Optional, List

# ==================== STANDARD INFERENCE (WITHOUT KV CACHE) ====================

def standard_generation_step(model, input_ids: List[int], step: int):
    """
    STANDARD INFERENCE: Recomputes everything at each step
    
    PROBLEM:
    - Processes ENTIRE sequence from scratch
    - Recomputes K, V for ALL previous tokens
    - Wasteful: same computation repeated many times
    
    At step i:
    - Input: [token_0, token_1, ..., token_i]  ← Entire sequence
    - Computes: K_0, V_0, K_1, V_1, ..., K_i, V_i  ← Recomputes all!
    - Attention: Uses all K, V
    
    Complexity: O(i²d) for step i (where i is sequence length)
    """
    # Convert to tensor (entire sequence)
    input_tensor = np.array([input_ids])  # Shape: (1, i+1)
    
    # Forward pass processes ENTIRE sequence
    # This recomputes K and V for ALL tokens, including previous ones
    embeddings = model.embed(input_tensor)  # (1, i+1, d_model)
    
    # At each layer, compute Q, K, V for ENTIRE sequence
    for layer in model.layers:
        # THIS IS THE PROBLEM: Recomputes K, V for all tokens
        Q = embeddings @ layer.W_q  # (1, i+1, d_k)
        K = embeddings @ layer.W_k  # (1, i+1, d_k) ← Recomputes K_0, K_1, ..., K_{i-1}!
        V = embeddings @ layer.W_v  # (1, i+1, d_v) ← Recomputes V_0, V_1, ..., V_{i-1}!
        
        # Attention computation
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        attention = softmax(scores) @ V
        
        embeddings = attention
    
    # Get logits for last position
    logits = model.lm_head(embeddings[0, -1, :])
    
    return logits


def standard_generate(model, prompt: List[int], max_length: int):
    """
    Standard generation: Recomputes everything at each step
    
    Example generating "The cat sat":
    
    Step 1: Generate "The"
        Input: [<start>]
        Computes: K_0, V_0
        Output: "The"
    
    Step 2: Generate "cat"
        Input: [<start>, "The"]
        Computes: K_0, V_0, K_1, V_1  ← REPROCESSES <start>! Recomputes K_0, V_0!
        Output: "cat"
    
    Step 3: Generate "sat"
        Input: [<start>, "The", "cat"]
        Computes: K_0, V_0, K_1, V_1, K_2, V_2  ← REPROCESSES all! Recomputes everything!
        Output: "sat"
    
    Total computation: O(n³d) for n tokens
    """
    generated = prompt.copy()
    
    for step in range(max_length):
        # At each step, process ENTIRE sequence
        # This is the problem: recomputes everything
        logits = standard_generation_step(model, generated, step)
        
        # Sample next token
        next_token = sample(logits)
        generated.append(next_token)
    
    return generated


# ==================== KV CACHE INFERENCE (WITH KV CACHE) ====================

def kv_cache_generation_step(model, input_ids: List[int], past_key_values: Optional[dict], step: int):
    """
    KV CACHE INFERENCE: Only computes for new token, reuses cache
    
    SOLUTION:
    - Processes ONLY the new token
    - Reuses cached K, V for previous tokens
    - Efficient: each K, V computed only once
    
    At step i:
    - Input: [token_i]  ← Only new token!
    - Computes: K_i, V_i  ← Only for new token!
    - Retrieves: K_0, ..., K_{i-1}, V_0, ..., V_{i-1} from cache  ← Reuses cached!
    - Attention: Uses cached + new
    
    Complexity: O(id) for step i (linear in sequence length)
    """
    # Convert to tensor (ONLY new token)
    input_tensor = np.array([[input_ids[-1]]])  # Shape: (1, 1) ← Only new token!
    
    # Forward pass processes ONLY new token
    embeddings = model.embed(input_tensor)  # (1, 1, d_model)
    
    # At each layer, compute Q, K, V for ONLY new token
    new_past_key_values = {}
    
    for layer_idx, layer in enumerate(model.layers):
        # THIS IS THE KEY: Only computes K, V for new token
        Q = embeddings @ layer.W_q  # (1, 1, d_k) ← Only new token!
        K_new = embeddings @ layer.W_k  # (1, 1, d_k) ← Only computes for new token!
        V_new = embeddings @ layer.W_v  # (1, 1, d_v) ← Only computes for new token!
        
        # Retrieve cached K, V for previous tokens
        if past_key_values and layer_idx in past_key_values:
            K_past = past_key_values[layer_idx]['keys']  # (1, i, d_k) ← Cached!
            V_past = past_key_values[layer_idx]['values']  # (1, i, d_v) ← Cached!
            
            # THIS IS THE KEY OPTIMIZATION: Concatenate cached + new
            # Instead of recomputing, we reuse cached values
            K = np.concatenate([K_past, K_new], axis=1)  # (1, i+1, d_k)
            V = np.concatenate([V_past, V_new], axis=1)  # (1, i+1, d_v)
        else:
            # First token: no cache yet
            K = K_new
            V = V_new
        
        # Attention computation (uses cached + new)
        scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
        attention = softmax(scores) @ V
        
        # Store in cache for next step
        new_past_key_values[layer_idx] = {
            'keys': K,
            'values': V
        }
        
        embeddings = attention
    
    # Get logits for last position
    logits = model.lm_head(embeddings[0, -1, :])
    
    return logits, new_past_key_values


def kv_cache_generate(model, prompt: List[int], max_length: int):
    """
    KV Cache generation: Only computes for new token, reuses cache
    
    Example generating "The cat sat":
    
    Step 1: Generate "The"
        Input: [<start>]
        Computes: K_0, V_0
        Cache: {0: (K_0, V_0)}
        Output: "The"
    
    Step 2: Generate "cat"
        Input: ["The"]  ← Only new token!
        Computes: K_1, V_1  ← Only for new token!
        Retrieves: K_0, V_0 from cache  ← Reuses cached!
        Cache: {0: (K_0, V_0, K_1, V_1)}  ← Updated
        Output: "cat"
    
    Step 3: Generate "sat"
        Input: ["cat"]  ← Only new token!
        Computes: K_2, V_2  ← Only for new token!
        Retrieves: K_0, V_0, K_1, V_1 from cache  ← Reuses all cached!
        Cache: {0: (K_0, V_0, K_1, V_1, K_2, V_2)}  ← Updated
        Output: "sat"
    
    Total computation: O(n²d) for n tokens
    """
    generated = prompt.copy()
    past_key_values = None
    
    # Process prompt if any
    if len(prompt) > 1:
        # Process all but last token of prompt
        for i in range(len(prompt) - 1):
            logits, past_key_values = kv_cache_generation_step(
                model, prompt[:i+1], past_key_values, i
            )
        generated = [prompt[-1]]
    
    # Generate new tokens
    for step in range(max_length):
        # At each step, process ONLY new token
        # This is the optimization: only computes for new token
        logits, past_key_values = kv_cache_generation_step(
            model, generated, past_key_values, step
        )
        
        # Sample next token
        next_token = sample(logits)
        generated.append(next_token)
    
    return generated


# ==================== SIDE-BY-SIDE COMPARISON ====================

def compare_step_by_step():
    """
    Shows exactly what happens at each step in both methods
    """
    print("=" * 80)
    print("STANDARD INFERENCE (WITHOUT KV CACHE)")
    print("=" * 80)
    print("\nStep 1: Generate token 1")
    print("  Input: [token_0]")
    print("  Computes: K_0, V_0")
    print("  Output: token_1")
    
    print("\nStep 2: Generate token 2")
    print("  Input: [token_0, token_1]  ← Entire sequence")
    print("  Computes: K_0, V_0, K_1, V_1  ← REPROCESSES token_0! Recomputes K_0, V_0!")
    print("  Output: token_2")
    
    print("\nStep 3: Generate token 3")
    print("  Input: [token_0, token_1, token_2]  ← Entire sequence")
    print("  Computes: K_0, V_0, K_1, V_1, K_2, V_2  ← REPROCESSES all! Recomputes everything!")
    print("  Output: token_3")
    
    print("\n" + "=" * 80)
    print("KV CACHE INFERENCE (WITH KV CACHE)")
    print("=" * 80)
    print("\nStep 1: Generate token 1")
    print("  Input: [token_0]")
    print("  Computes: K_0, V_0")
    print("  Cache: {K_0, V_0}")
    print("  Output: token_1")
    
    print("\nStep 2: Generate token 2")
    print("  Input: [token_1]  ← Only new token!")
    print("  Computes: K_1, V_1  ← Only for new token!")
    print("  Retrieves: K_0, V_0 from cache  ← Reuses cached!")
    print("  Cache: {K_0, V_0, K_1, V_1}  ← Updated")
    print("  Output: token_2")
    
    print("\nStep 3: Generate token 3")
    print("  Input: [token_2]  ← Only new token!")
    print("  Computes: K_2, V_2  ← Only for new token!")
    print("  Retrieves: K_0, V_0, K_1, V_1 from cache  ← Reuses all cached!")
    print("  Cache: {K_0, V_0, K_1, V_1, K_2, V_2}  ← Updated")
    print("  Output: token_3")
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)
    print("\n1. INPUT SIZE:")
    print("   Standard: Entire sequence [token_0, ..., token_i]")
    print("   KV Cache: Only new token [token_i]")
    
    print("\n2. K, V COMPUTATION:")
    print("   Standard: Computes K, V for ALL tokens (recomputes previous)")
    print("   KV Cache: Computes K, V only for NEW token (reuses cached)")
    
    print("\n3. THE KEY CODE:")
    print("   Standard: K = compute_K([token_0, ..., token_i])  # Recomputes all")
    print("   KV Cache: K = concatenate([K_cache, K_new])  # Reuses cache")
    
    print("\n4. COMPLEXITY:")
    print("   Standard: O(n³d) total, O(i²d) per step")
    print("   KV Cache: O(n²d) total, O(id) per step")
    print("   Speedup: ~n× for sequences of length n")
    
    print("\n5. MEMORY:")
    print("   Standard: O(1) - no cache")
    print("   KV Cache: O(nd) - stores K, V for all tokens")
    print("   Trade-off: Memory for computation speed")


# ==================== THE KEY CODE DIFFERENCE ====================

def show_key_code_difference():
    """
    Shows the exact code that makes KV cache work
    """
    print("=" * 80)
    print("THE KEY CODE DIFFERENCE")
    print("=" * 80)
    
    print("\nSTANDARD INFERENCE:")
    print("-" * 80)
    print("""
    # At each step, process entire sequence
    input_ids = [token_0, token_1, ..., token_i]  # Entire sequence
    
    # Recompute K, V for ALL tokens
    K = compute_K(input_ids)  # Recomputes K_0, K_1, ..., K_i
    V = compute_V(input_ids)  # Recomputes V_0, V_1, ..., V_i
    
    # Attention uses all K, V
    attention = compute_attention(Q, K, V)
    """)
    
    print("\nKV CACHE INFERENCE:")
    print("-" * 80)
    print("""
    # At each step, process ONLY new token
    input_ids = [token_i]  # Only new token!
    
    # Only compute K, V for new token
    K_new = compute_K(input_ids)  # Only computes K_i
    V_new = compute_V(input_ids)  # Only computes V_i
    
    # THIS IS THE KEY: Concatenate cached + new
    K = concatenate([K_cache, K_new])  # Reuses cached K_0, ..., K_{i-1}
    V = concatenate([V_cache, V_new])  # Reuses cached V_0, ..., V_{i-1}
    
    # Attention uses cached + new
    attention = compute_attention(Q, K, V)
    
    # Update cache for next step
    K_cache = K  # Store for next step
    V_cache = V
    """)
    
    print("\nTHE CONCATENATION IS THE KEY:")
    print("  K = concatenate([K_cache, K_new])")
    print("  This single line reuses all cached K, V values!")
    print("  Without it, we'd have to recompute everything.")
    print("  With it, we only compute for the new token.")


if __name__ == "__main__":
    compare_step_by_step()
    print("\n")
    show_key_code_difference()

