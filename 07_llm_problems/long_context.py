"""
Long Context Solutions
Problem: Standard attention O(n²) too expensive for long sequences
Solutions: Chunking, sliding window, sparse attention
"""
import numpy as np

def softmax(x, axis=-1):
    """Softmax function"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def standard_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, d_k: int) -> np.ndarray:
    """
    Standard attention: O(n²) complexity
    Problem: Too expensive for long sequences
    """
    scores = Q @ K.T / np.sqrt(d_k)
    attention_weights = softmax(scores)
    return attention_weights @ V

def chunked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                     chunk_size: int, d_k: int) -> np.ndarray:
    """
    Chunked Attention: Process in chunks to reduce memory
    
    Problem: Full attention O(n²) memory
    Solution: Process sequence in chunks
    Memory: O(chunk_size × n) instead of O(n²)
    """
    seq_len = Q.shape[0]
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        chunk_end = min(i + chunk_size, seq_len)
        Q_chunk = Q[i:chunk_end]
        
        # Attend to all K, V (can also limit K/V to window)
        scores = Q_chunk @ K.T / np.sqrt(d_k)
        attention_weights = softmax(scores)
        output_chunk = attention_weights @ V
        
        outputs.append(output_chunk)
    
    return np.concatenate(outputs, axis=0)

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                            window_size: int, d_k: int) -> np.ndarray:
    """
    Sliding Window Attention: Each position only attends to local window
    
    Problem: Full attention O(n²) too expensive
    Solution: Local attention + some global positions
    Complexity: O(n × window_size) instead of O(n²)
    """
    seq_len = Q.shape[0]
    outputs = []
    
    for i in range(seq_len):
        # Define window around current position
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        
        Q_i = Q[i:i+1]  # Current query
        K_window = K[start:end]  # Keys in window
        V_window = V[start:end]  # Values in window
        
        # Local attention
        scores = Q_i @ K_window.T / np.sqrt(d_k)
        attention_weights = softmax(scores)
        output = attention_weights @ V_window
        
        outputs.append(output)
    
    return np.concatenate(outputs, axis=0)

def sparse_attention_with_global(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 window_size: int, global_indices: list,
                                 d_k: int) -> np.ndarray:
    """
    Sparse Attention: Local window + global positions
    
    Problem: Need both local and global context
    Solution: Attend to local window + specific global positions
    
    Args:
        global_indices: Positions that all tokens attend to (e.g., [0] for [CLS])
    """
    seq_len = Q.shape[0]
    outputs = []
    
    for i in range(seq_len):
        # Local window
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        local_indices = list(range(start, end))
        
        # Combine local + global
        attend_indices = sorted(set(local_indices + global_indices))
        
        Q_i = Q[i:i+1]
        K_selected = K[attend_indices]
        V_selected = V[attend_indices]
        
        # Attention
        scores = Q_i @ K_selected.T / np.sqrt(d_k)
        attention_weights = softmax(scores)
        output = attention_weights @ V_selected
        
        outputs.append(output)
    
    return np.concatenate(outputs, axis=0)


# Usage Example
if __name__ == "__main__":
    print("Long Context Solutions")
    print("=" * 60)
    
    # Simulate long sequence
    seq_len = 1000  # Long sequence
    d_k = 64
    
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    print(f"Sequence length: {seq_len}")
    print(f"Standard attention complexity: O({seq_len}²) = {seq_len**2:,} operations")
    
    # Chunked attention
    chunk_size = 128
    print(f"\nChunked Attention (chunk_size={chunk_size}):")
    print(f"Complexity: O({seq_len} × {chunk_size}) = {seq_len * chunk_size:,} operations")
    output_chunked = chunked_attention(Q, K, V, chunk_size, d_k)
    print(f"Output shape: {output_chunked.shape}")
    
    # Sliding window
    window_size = 256
    print(f"\nSliding Window (window_size={window_size}):")
    print(f"Complexity: O({seq_len} × {window_size}) = {seq_len * window_size:,} operations")
    output_window = sliding_window_attention(Q, K, V, window_size, d_k)
    print(f"Output shape: {output_window.shape}")
    
    # Sparse with global
    print(f"\nSparse with Global (window={window_size}, global=[0]):")
    output_sparse = sparse_attention_with_global(Q, K, V, window_size, [0], d_k)
    print(f"Output shape: {output_sparse.shape}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Standard: {seq_len**2:,} operations")
    print(f"Chunked: {seq_len * chunk_size:,} operations ({seq_len**2 / (seq_len * chunk_size):.1f}x faster)")
    print(f"Window: {seq_len * window_size:,} operations ({seq_len**2 / (seq_len * window_size):.1f}x faster)")

