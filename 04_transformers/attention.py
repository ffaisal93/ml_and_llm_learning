"""
Self-Attention from Scratch
Interview question: "Implement attention mechanism"

Mathematical Formulation:
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Steps:
1. Compute attention scores: scores = Q @ K^T
2. Scale by √d_k: scores = scores / √d_k (prevents large values)
3. Apply mask (if provided): scores = scores + mask (mask = -∞ for masked positions)
4. Softmax: attention_weights = softmax(scores)
5. Apply to values: output = attention_weights @ V

Why √d_k? 
- Without scaling, dot products grow large (variance = d_k)
- Large values → extreme softmax → vanishing gradients
- Scaling keeps variance = 1
"""
import numpy as np
from typing import Optional, Tuple

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                   d_k: int, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-Attention: Attention(Q, K, V) = softmax(QK^T / √d_k) × V
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
        d_k: Dimension of keys (for scaling)
        mask: Optional attention mask
    
    Returns:
        (output, attention_weights)
    """
    # Compute attention scores: Q @ K^T
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to -inf)
    # why?
    # because we want to mask out the future positions, so that the model cannot attend to future positions
    # we do this by setting the scores of the future positions to -1e9, so that the softmax function will output 0 for the future positions
    # this is done to prevent the model from attending to future positions
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax over last dimension
    # Subtract max for numerical stability
    # explain the following code line by line:
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Apply to values
    output = attention_weights @ V
    
    return output, attention_weights


def multi_head_attention(x: np.ndarray, d_model: int, num_heads: int,
                        W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                        W_o: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Multi-Head Attention
    
    Args:
        x: Input (seq_len, d_model)
        d_model: Model dimension
        num_heads: Number of attention heads
        W_q, W_k, W_v, W_o: Weight matrices
        mask: Optional attention mask
    """
    seq_len, _ = x.shape
    d_k = d_model // num_heads
    
    # Project to Q, K, V
    Q = x @ W_q  # (seq_len, d_model)
    K = x @ W_k
    V = x @ W_v
    
    # Reshape for multi-head: (num_heads, seq_len, d_k)
    # explain transpose function:
    # transpose function is used to transpose the array, so that the shape of the array is changed
    # for example, if the array is (2,3,4), then the transpose function will change the shape to (4,3,2)
    # this is done to make the array easier to understand and work with
    # in this case, we are reshaping the array to (num_heads, seq_len, d_k) and then transposing it to (seq_len, num_heads, d_k)
    # this is done to make the array easier to understand and work with
    Q = Q.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(seq_len, num_heads, d_k).transpose(1, 0, 2)
    
    # Apply attention to each head
    attention_outputs = []
    for head in range(num_heads):
        output, _ = self_attention(
            Q[head], K[head], V[head], 
            d_k, mask
        )
        attention_outputs.append(output)
    
    # Concatenate heads: (seq_len, d_model)
    concat = np.concatenate(attention_outputs, axis=-1)
    
    # Final projection
    output = concat @ W_o
    
    return output


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    
    ## explain the following with example values:
    # seq_len = 10
    # d_model = 64
    # position = np.arange(seq_len).reshape(-1, 1) = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
    # div_term = np.exp(np.arange(0, d_model, 2) * 
    #                  -(np.log(10000.0) / d_model)) = [10000^(0/64), 10000^(2/64), 10000^(4/64), 10000^(6/64), 10000^(8/64), 10000^(10/64), 10000^(12/64), 10000^(14/64), 10000^(16/64), 10000^(18/64)]
    # pe[:, 0::2] = np.sin(position * div_term) = [[sin(0*10000^(0/64)), sin(1*10000^(0/64)), sin(2*10000^(0/64)), sin(3*10000^(0/64)), sin(4*10000^(0/64)), sin(5*10000^(0/64)), sin(6*10000^(0/64)), sin(7*10000^(0/64)), sin(8*10000^(0/64)), sin(9*10000^(0/64))]]
    # pe[:, 1::2] = np.cos(position * div_term) = [[cos(0*10000^(0/64)), cos(1*10000^(0/64)), cos(2*10000^(0/64)), cos(3*10000^(0/64)), cos(4*10000^(0/64)), cos(5*10000^(0/64)), cos(6*10000^(0/64)), cos(7*10000^(0/64)), cos(8*10000^(0/64)), cos(9*10000^(0/64))]]  
    # pe = [[sin(0*10000^(0/64)), cos(0*10000^(0/64)), sin(1*10000^(0/64)), cos(1*10000^(0/64)), sin(2*10000^(0/64)), cos(2*10000^(0/64)), sin(3*10000^(0/64)), cos(3*10000^(0/64)), sin(4*10000^(0/64)), cos(4*10000^(0/64)), sin(5*10000^(0/64)), cos(5*10000^(0/64)), sin(6*10000^(0/64)), cos(6*10000^(0/64)), sin(7*10000^(0/64)), cos(7*10000^(0/64)), sin(8*10000^(0/64)), cos(8*10000^(0/64)), sin(9*10000^(0/64)), cos(9*10000^(0/64))]]
    # return pe
    
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * 
                     -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


# Usage Example
if __name__ == "__main__":
    # Example: Simple self-attention
    seq_len = 5
    d_k = 64
    
    # Random Q, K, V
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    # Apply attention
    output, attention_weights = self_attention(Q, K, V, d_k)
    
    print(f"Input shape: Q={Q.shape}, K={K.shape}, V={V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"\nAttention weights (first row):\n{attention_weights[0]}")
    
    # Example: Multi-head attention
    d_model = 128
    num_heads = 8
    x = np.random.randn(seq_len, d_model)
    
    # Initialize weights
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    W_o = np.random.randn(d_model, d_model) * 0.1
    
    output = multi_head_attention(x, d_model, num_heads, W_q, W_k, W_v, W_o)
    print(f"\nMulti-head attention output shape: {output.shape}")
    
    # Example: Positional encoding
    pe = positional_encoding(seq_len=10, d_model=64)
    print(f"\nPositional encoding shape: {pe.shape}")
    
    # Example: Causal attention
    print("\n" + "=" * 60)
    print("Causal Attention Example")
    print("=" * 60)
    
    def causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, d_k: int):
        """
        Causal Attention: Masks future positions (for autoregressive generation)
        
        WHAT THIS DOES:
        1. Creates lower triangular mask: np.tril(np.ones((seq_len, seq_len)))
           - Position i can attend to positions j where j ≤ i (past and current)
           - Position i cannot attend to positions j where j > i (future)
        
        2. Applies mask to attention scores
           - Future positions get -∞ (which becomes 0 after softmax)
           - Past/current positions keep their computed scores
        
        3. Result: Each position only attends to itself and previous positions
        
        WHY LOWER TRIANGULAR?
        - Lower triangular: 1s on and below diagonal (can attend to past/current)
        - Upper triangular: Would be wrong (allows future, blocks past)
        """
        seq_len = Q.shape[0]
        # Create lower triangular mask
        mask = np.tril(np.ones((seq_len, seq_len)))
        return self_attention(Q, K, V, d_k, mask=mask)
    
    # Apply causal attention
    causal_output, causal_weights = causal_attention(Q, K, V, d_k)
    print(f"Causal attention output shape: {causal_output.shape}")
    print(f"Causal attention weights shape: {causal_weights.shape}")
    print(f"\nCausal attention weights (first row - position 0):")
    print(causal_weights[0].round(3))
    print("\nNote: Position 0 can only attend to itself (future positions = 0.0)")
    print("This enforces autoregressive property for GPT-style models!")

