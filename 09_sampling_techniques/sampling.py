"""
Sampling Techniques from Scratch
Interview question: "Implement top-p sampling"
"""
import numpy as np

def softmax(x):
    """Softmax function"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def greedy_decode(logits: np.ndarray) -> int:
    """
    Greedy: Always pick most likely token
    Deterministic but can be repetitive
    """
    return np.argmax(logits)

def top_k_sampling(logits: np.ndarray, k: int = 50) -> int:
    """
    Top-k: Sample from top k most likely tokens
    
    Args:
        logits: (vocab_size,) unnormalized scores
        k: Number of top tokens to consider
    """
    # Get top k indices
    top_k_indices = np.argsort(logits)[-k:]
    top_k_logits = logits[top_k_indices]
    
    # Softmax over top k
    probs = softmax(top_k_logits)
    
    # Sample
    sampled_idx = np.random.choice(len(top_k_indices), p=probs)
    return top_k_indices[sampled_idx]

def top_p_sampling(logits: np.ndarray, p: float = 0.9) -> int:
    """
    Top-p (Nucleus): Sample from smallest set with cum_prob >= p
    
    Most popular method (used by ChatGPT, Claude)
    
    Args:
        logits: (vocab_size,) unnormalized scores
        p: Nucleus probability threshold (0.0 to 1.0)
    """
    # Sort logits descending
    sorted_indices = np.argsort(logits)[::-1]
    sorted_logits = logits[sorted_indices]
    
    # Softmax
    probs = softmax(sorted_logits)
    
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

def temperature_sampling(logits: np.ndarray, temperature: float = 1.0) -> int:
    """
    Temperature: Scale logits before softmax
    
    Args:
        logits: (vocab_size,) unnormalized scores
        temperature: 
            < 1.0: More deterministic (sharp)
            = 1.0: Normal
            > 1.0: More random (flat)
    """
    # Scale by temperature
    scaled_logits = logits / temperature
    
    # Softmax and sample
    probs = softmax(scaled_logits)
    return np.random.choice(len(logits), p=probs)

def combined_sampling(logits: np.ndarray,
                      temperature: float = 1.0,
                      top_p: float = 0.9) -> int:
    """
    Combined: Temperature + Top-p
    Most common in production (ChatGPT, Claude)
    """
    # Apply temperature first
    scaled_logits = logits / temperature
    
    # Then top-p
    return top_p_sampling(scaled_logits, top_p)


# Usage Example
if __name__ == "__main__":
    print("Sampling Techniques")
    print("=" * 60)
    
    # Simulate logits (vocab_size = 1000)
    vocab_size = 1000
    np.random.seed(42)
    logits = np.random.randn(vocab_size)
    
    # Make some tokens more likely
    logits[100:110] += 5  # Make tokens 100-109 more likely
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Most likely token (greedy): {np.argmax(logits)}")
    
    # Test different sampling methods
    print("\nSampling Results (10 samples each):")
    
    print("\n1. Greedy (deterministic):")
    for _ in range(5):
        print(f"  {greedy_decode(logits)}", end=" ")
    print()
    
    print("\n2. Top-k (k=10):")
    for _ in range(5):
        print(f"  {top_k_sampling(logits, k=10)}", end=" ")
    print()
    
    print("\n3. Top-p (p=0.9):")
    for _ in range(5):
        print(f"  {top_p_sampling(logits, p=0.9)}", end=" ")
    print()
    
    print("\n4. Temperature (T=0.5, more deterministic):")
    for _ in range(5):
        print(f"  {temperature_sampling(logits, temperature=0.5)}", end=" ")
    print()
    
    print("\n5. Temperature (T=2.0, more random):")
    for _ in range(5):
        print(f"  {temperature_sampling(logits, temperature=2.0)}", end=" ")
    print()
    
    print("\n6. Combined (T=0.8, top_p=0.9):")
    for _ in range(5):
        print(f"  {combined_sampling(logits, temperature=0.8, top_p=0.9)}", end=" ")
    print()

