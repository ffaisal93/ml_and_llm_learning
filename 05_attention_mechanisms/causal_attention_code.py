"""
Causal Attention: Complete Implementation with Detailed Explanations
Shows exactly what happens step-by-step
"""
import numpy as np
from typing import Optional

# ==================== HELPER: STANDARD SELF-ATTENTION ====================

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                  d_k: int, mask: Optional[np.ndarray] = None) -> tuple:
    """
    Standard self-attention
    
    Args:
        Q: Query matrix, shape (seq_len, d_k)
        K: Key matrix, shape (seq_len, d_k)
        V: Value matrix, shape (seq_len, d_v)
        d_k: Key dimension (for scaling)
        mask: Optional attention mask, shape (seq_len, seq_len)
    Returns:
        (output, attention_weights)
    """
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # Shape: (seq_len, seq_len)
    
    # Apply mask if provided
    if mask is not None:
        # THIS IS THE KEY: Set masked positions to -∞
        # Where mask == 0 (future positions), set to -∞
        # Where mask == 1 (past/current), keep original scores
        scores = np.where(mask == 0, -1e9, scores)
    
    # Softmax (with numerical stability)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    # Future positions: exp(-∞) ≈ 0, so attention weight = 0
    
    # Apply to values
    output = attention_weights @ V
    
    return output, attention_weights


# ==================== CAUSAL ATTENTION ====================

def causal_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, d_k: int) -> tuple:
    """
    Causal Attention: Masks future positions (for autoregressive generation)
    
    WHAT THIS FUNCTION DOES:
    
    1. Creates a lower triangular mask
       - Position i can attend to positions j where j ≤ i (past and current)
       - Position i cannot attend to positions j where j > i (future)
    
    2. Applies mask to attention scores
       - Future positions get -∞ (which becomes 0 after softmax)
       - Past/current positions keep their computed scores
    
    3. Result: Each position only attends to itself and previous positions
    
    WHY LOWER TRIANGULAR?
    - Lower triangular: 1s on and below diagonal (can attend to past/current)
    - Upper triangular: Would be wrong (allows future, blocks past)
    
    Args:
        Q: Query matrix, shape (seq_len, d_k)
        K: Key matrix, shape (seq_len, d_k)
        V: Value matrix, shape (seq_len, d_v)
        d_k: Key dimension
    Returns:
        (output, attention_weights)
    """
    seq_len = Q.shape[0]
    
    # STEP 1: Create lower triangular mask
    # np.ones creates matrix of all 1s
    ones_matrix = np.ones((seq_len, seq_len))
    # np.tril keeps only lower triangular part (sets upper to 0)
    mask = np.tril(ones_matrix)
    
    # STEP 2: Apply mask in attention
    # This ensures each position can only attend to past and current
    return self_attention(Q, K, V, d_k, mask=mask)


# ==================== STEP-BY-STEP VISUALIZATION ====================

def visualize_causal_attention(seq_len: int = 4):
    """
    Visualize what happens in causal attention step-by-step
    """
    print("=" * 80)
    print("CAUSAL ATTENTION: Step-by-Step Visualization")
    print("=" * 80)
    
    print(f"\nSequence length: {seq_len}")
    print("Tokens: [token_0, token_1, token_2, token_3]")
    
    # Step 1: Create mask
    print("\n" + "-" * 80)
    print("STEP 1: Create Lower Triangular Mask")
    print("-" * 80)
    
    ones_matrix = np.ones((seq_len, seq_len))
    print("\n1a. Create matrix of all 1s:")
    print(ones_matrix)
    
    mask = np.tril(ones_matrix)
    print("\n1b. Apply np.tril() (lower triangular):")
    print(mask)
    print("\nInterpretation:")
    print("  - Row i = position i")
    print("  - Column j = position j")
    print("  - 1 = can attend, 0 = cannot attend")
    print("  - Lower triangular = can attend to past/current, not future")
    
    # Step 2: Show what each position can see
    print("\n" + "-" * 80)
    print("STEP 2: What Each Position Can Attend To")
    print("-" * 80)
    
    for i in range(seq_len):
        can_attend = np.where(mask[i] == 1)[0].tolist()
        print(f"\nPosition {i} (token_{i}):")
        print(f"  Can attend to positions: {can_attend}")
        print(f"  Cannot attend to positions: {[j for j in range(seq_len) if j not in can_attend]}")
        print(f"  Interpretation: Can see tokens at positions {can_attend}")
    
    # Step 3: Show attention scores
    print("\n" + "-" * 80)
    print("STEP 3: Attention Scores (Before and After Masking)")
    print("-" * 80)
    
    # Simulate attention scores
    np.random.seed(42)
    Q = np.random.randn(seq_len, 64)
    K = np.random.randn(seq_len, 64)
    scores = Q @ K.T / np.sqrt(64)
    
    print("\n3a. Attention scores (before masking):")
    print("    (These are computed from Q @ K.T)")
    print(scores.round(2))
    print("\n    Problem: All positions can see all other positions!")
    
    # Apply mask
    masked_scores = np.where(mask == 0, -1e9, scores)
    
    print("\n3b. Attention scores (after masking):")
    print("    (Future positions set to -∞)")
    print(masked_scores.round(2))
    print("\n    Note: -1e9 represents -∞ (very large negative number)")
    print("    Future positions now have -∞ scores")
    
    # Step 4: Show attention weights after softmax
    print("\n" + "-" * 80)
    print("STEP 4: Attention Weights (After Softmax)")
    print("-" * 80)
    
    # Softmax
    exp_scores = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    print("\nAttention weights (after softmax):")
    print(attention_weights.round(3))
    print("\nInterpretation:")
    print("  - Each row sums to 1.0 (probability distribution)")
    print("  - Future positions have 0.0 attention weight")
    print("  - Past/current positions have non-zero weights")
    
    # Step 5: Summary
    print("\n" + "-" * 80)
    print("SUMMARY: What Causal Attention Achieves")
    print("-" * 80)
    print("""
    ✅ Position 0: Can only see itself (no past tokens)
    ✅ Position 1: Can see positions 0, 1 (past and current)
    ✅ Position 2: Can see positions 0, 1, 2 (past and current)
    ✅ Position 3: Can see positions 0, 1, 2, 3 (past and current)
    
    ❌ No position can see future tokens
    ❌ This enforces autoregressive property
    ❌ Makes training and inference consistent
    """)


# ==================== COMPARISON: WITH vs WITHOUT MASK ====================

def compare_with_without_mask():
    """
    Compare attention with and without causal mask
    """
    print("=" * 80)
    print("COMPARISON: With vs Without Causal Mask")
    print("=" * 80)
    
    seq_len = 4
    d_k = 64
    
    # Random Q, K, V
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    # Without mask
    print("\n1. WITHOUT CAUSAL MASK (Standard Self-Attention):")
    print("-" * 80)
    output_no_mask, weights_no_mask = self_attention(Q, K, V, d_k, mask=None)
    print("Attention weights shape:", weights_no_mask.shape)
    print("Attention weights (first row - what position 0 attends to):")
    print(weights_no_mask[0].round(3))
    print("\n❌ Problem: Position 0 can attend to ALL positions (including future)")
    print("   This breaks autoregressive property!")
    
    # With mask
    print("\n2. WITH CAUSAL MASK (Causal Attention):")
    print("-" * 80)
    output_mask, weights_mask = causal_attention(Q, K, V, d_k)
    print("Attention weights shape:", weights_mask.shape)
    print("Attention weights (first row - what position 0 attends to):")
    print(weights_mask[0].round(3))
    print("\n✅ Solution: Position 0 can only attend to itself")
    print("   Future positions have 0.0 attention weight")
    print("   This enforces autoregressive property!")
    
    # Show difference
    print("\n3. DIFFERENCE:")
    print("-" * 80)
    print("Without mask - Position 0 attention weights:")
    print(weights_no_mask[0].round(3))
    print("\nWith mask - Position 0 attention weights:")
    print(weights_mask[0].round(3))
    print("\nKey difference: Future positions (indices 1, 2, 3) are 0.0 with mask")


# ==================== WHY LOWER TRIANGULAR ====================

def explain_lower_triangular():
    """
    Explain why we use lower triangular (not upper)
    """
    print("=" * 80)
    print("WHY LOWER TRIANGULAR?")
    print("=" * 80)
    
    seq_len = 4
    
    print("\n1. LOWER TRIANGULAR (CORRECT for causal attention):")
    print("-" * 80)
    lower_mask = np.tril(np.ones((seq_len, seq_len)))
    print(lower_mask)
    print("\nInterpretation:")
    print("  - Position 0: Can see [0] (only itself)")
    print("  - Position 1: Can see [0, 1] (past and current)")
    print("  - Position 2: Can see [0, 1, 2] (past and current)")
    print("  - Position 3: Can see [0, 1, 2, 3] (past and current)")
    print("\n✅ This is what we want for autoregressive generation!")
    
    print("\n2. UPPER TRIANGULAR (WRONG for causal attention):")
    print("-" * 80)
    upper_mask = np.triu(np.ones((seq_len, seq_len)))
    print(upper_mask)
    print("\nInterpretation:")
    print("  - Position 0: Can see [0, 1, 2, 3] (including future!)")
    print("  - Position 1: Can see [1, 2, 3] (cannot see past!)")
    print("  - Position 2: Can see [2, 3] (cannot see past!)")
    print("  - Position 3: Can see [3] (only itself)")
    print("\n❌ This is the OPPOSITE of what we want!")
    print("   Position 0 can see future, position 1 cannot see past")
    
    print("\n3. CONCLUSION:")
    print("-" * 80)
    print("✅ Use LOWER triangular for causal attention")
    print("❌ Do NOT use upper triangular")


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    print("Causal Attention: Complete Explanation")
    print("=" * 80)
    
    # Visualize step-by-step
    visualize_causal_attention(seq_len=4)
    
    print("\n\n")
    
    # Compare with/without mask
    compare_with_without_mask()
    
    print("\n\n")
    
    # Explain lower triangular
    explain_lower_triangular()
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print("""
    1. Causal attention uses LOWER triangular mask
    2. Lower triangular = can attend to past/current, not future
    3. Mask sets future positions to -∞ in attention scores
    4. After softmax, future positions get 0 attention weight
    5. This enforces autoregressive property for GPT-style models
    6. Makes training and inference consistent
    """)

