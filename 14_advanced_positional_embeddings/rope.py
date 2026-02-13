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
    Returns:
        (q_rotated, k_rotated)
    """
    # Create rotation angles based on position
    # Different frequencies for different dimensions
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
            
            # Rotate pair (2D rotation)
            q_rot[i] = q[i] * cos_angle - q[i+1] * sin_angle
            q_rot[i+1] = q[i] * sin_angle + q[i+1] * cos_angle
            
            k_rot[i] = k[i] * cos_angle - k[i+1] * sin_angle
            k_rot[i+1] = k[i] * sin_angle + k[i+1] * cos_angle
        else:
            # Odd dimension, no rotation
            q_rot[i] = q[i]
            k_rot[i] = k[i]
    
    return q_rot, k_rot


def rope_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   positions: np.ndarray, d_k: int) -> np.ndarray:
    """
    Attention with RoPE
    
    Args:
        Q: Queries (seq_len, d_model)
        K: Keys (seq_len, d_model)
        V: Values (seq_len, d_model)
        positions: Position indices (seq_len,)
        d_k: Key dimension
    """
    seq_len, d_model = Q.shape
    output = np.zeros_like(Q)
    
    for i in range(seq_len):
        q_i = Q[i]
        # Apply RoPE to Q and K at position i
        q_rot, _ = apply_rope(q_i, q_i, positions[i], d_model)
        
        # For each key position
        scores = np.zeros(seq_len)
        for j in range(seq_len):
            k_j = K[j]
            _, k_rot = apply_rope(k_j, k_j, positions[j], d_model)
            
            # Compute attention score with rotated Q and K
            scores[j] = q_rot @ k_rot / np.sqrt(d_k)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        attention_weights = exp_scores / np.sum(exp_scores)
        
        # Apply to values
        output[i] = attention_weights @ V
    
    return output


# Usage Example
if __name__ == "__main__":
    print("RoPE Example")
    print("=" * 60)
    
    d_model = 64
    seq_len = 10
    
    # Random Q, K, V
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_model)
    K = np.random.randn(seq_len, d_model)
    V = np.random.randn(seq_len, d_model)
    positions = np.arange(seq_len)
    
    # Apply RoPE attention
    output = rope_attention(Q, K, V, positions, d_model)
    
    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"RoPE applied to positions 0-{seq_len-1}")

