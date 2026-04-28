"""
Perplexity: Complete Implementation
Simple, interview-writable code for computing perplexity
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

# ==================== BASIC PERPLEXITY ====================

def compute_perplexity(log_probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute perplexity from log probabilities
    
    PP = exp(-(1/N) * Σ log P(w_i | context))
    
    Args:
        log_probs: Log probabilities of true tokens, shape (batch_size, seq_len)
        mask: Optional mask to exclude padding tokens, shape (batch_size, seq_len)
    Returns:
        Perplexity (scalar)
    """
    if mask is not None:
        # Only compute on non-padded tokens
        log_probs = log_probs * mask
        num_tokens = mask.sum().item()
    else:
        num_tokens = log_probs.numel()
        # explain log_probs.numel() with same example
        # Example:
        # log_probs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        # log_probs.numel() = 9
        # log_probs.numel() is used to get the number of elements in the tensor
    
    # Average negative log-likelihood
    avg_nll = -log_probs.sum().item() / num_tokens
    #sum.item() is used to get the sum of the tensor as a scalar
    # Example:
    # log_probs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    # log_probs.sum().item() = 4.5
    # log_probs.sum().item() is used to get the sum of the tensor as a scalar

    # Perplexity = exp(avg_nll)
    perplexity = np.exp(avg_nll)
    
    return perplexity


def perplexity_from_logits(logits: torch.Tensor, targets: torch.Tensor,
                           mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute perplexity from model logits
    
    Args:
        logits: Model output logits, shape (batch_size, seq_len, vocab_size)
        targets: True token indices, shape (batch_size, seq_len)
        mask: Optional mask to exclude padding, shape (batch_size, seq_len)
    Returns:
        Perplexity (scalar)
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get log probability of true tokens
    # Gather: log_probs[batch, seq, target[batch, seq]]
    batch_size, seq_len = targets.shape
    indices = targets.unsqueeze(-1)  # (batch, seq, 1)
    true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
    # Shape: (batch_size, seq_len)
    
    # Compute perplexity
    return compute_perplexity(true_token_log_probs, mask)


# ==================== PERPLEXITY FOR LANGUAGE MODELS ====================

def language_model_perplexity(model, dataloader, device: str = 'cpu') -> float:
    """
    Compute perplexity for a language model on a dataset
    
    Args:
        model: Language model (returns logits)
        dataloader: DataLoader with (input_ids, labels) batches
        device: Device to run on
    Returns:
        Average perplexity across dataset
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids)
            logits = outputs.logits  # (batch, seq_len, vocab_size)
            
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            if attention_mask is not None:
                shift_mask = attention_mask[:, 1:].contiguous()
            else:
                shift_mask = None
            
            # Compute perplexity for this batch
            batch_pp = perplexity_from_logits(shift_logits, shift_labels, shift_mask)
            
            # Accumulate (weighted by number of tokens)
            if shift_mask is not None:
                batch_tokens = shift_mask.sum().item()
            else:
                batch_tokens = shift_labels.numel()
            
            total_nll += np.log(batch_pp) * batch_tokens  # log(PP) = NLL
            total_tokens += batch_tokens
    
    # Average perplexity
    avg_nll = total_nll / total_tokens
    avg_perplexity = np.exp(avg_nll)
    
    return avg_perplexity


# ==================== PER-TOKEN PERPLEXITY ====================

def per_token_perplexity(logits: torch.Tensor, targets: torch.Tensor,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute perplexity for each token position
    
    Args:
        logits: Model logits, shape (batch_size, seq_len, vocab_size)
        targets: True tokens, shape (batch_size, seq_len)
        mask: Optional mask, shape (batch_size, seq_len)
    Returns:
        Per-token perplexity, shape (batch_size, seq_len)
    """
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get true token log probabilities
    batch_size, seq_len = targets.shape
    indices = targets.unsqueeze(-1)
    true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
    
    # Per-token perplexity = exp(-log_prob)
    per_token_pp = torch.exp(-true_token_log_probs)
    
    # Mask out padding
    if mask is not None:
        per_token_pp = per_token_pp * mask
    
    return per_token_pp


# ==================== CHARACTER-LEVEL PERPLEXITY ====================

def character_level_perplexity(char_logits: torch.Tensor, char_targets: torch.Tensor,
                              mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute character-level perplexity
    
    Args:
        char_logits: Character logits, shape (batch, seq_len, num_chars)
        char_targets: True character indices, shape (batch, seq_len)
        mask: Optional mask, shape (batch, seq_len)
    Returns:
        Character-level perplexity
    """
    return perplexity_from_logits(char_logits, char_targets, mask)


# ==================== BITS PER TOKEN ====================

def bits_per_token(logits: torch.Tensor, targets: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> float:
    """
    Compute bits per token (BPT)
    
    BPT = (1/N) * Σ log₂(1/P(w_i | context))
    BPT = log₂(PP)
    
    Args:
        logits: Model logits, shape (batch, seq_len, vocab_size)
        targets: True tokens, shape (batch, seq_len)
        mask: Optional mask, shape (batch, seq_len)
    Returns:
        Bits per token (scalar)
    """
    # Get log probabilities (base e)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get true token log probabilities
    batch_size, seq_len = targets.shape
    indices = targets.unsqueeze(-1)
    true_token_log_probs = log_probs.gather(dim=-1, index=indices).squeeze(-1)
    
    # Convert to base 2
    true_token_log_probs_base2 = true_token_log_probs / np.log(2)
    
    # Bits per token = -log₂(P) = -log_e(P) / log_e(2)
    if mask is not None:
        bpt = -(true_token_log_probs_base2 * mask).sum().item() / mask.sum().item()
    else:
        bpt = -true_token_log_probs_base2.mean().item()
    
    return bpt


# ==================== NORMALIZED PERPLEXITY ====================

def normalized_perplexity(perplexity: float, vocab_size: int) -> float:
    """
    Normalize perplexity by vocabulary size
    
    Normalized PP = PP / vocab_size
    
    Helps compare models with different vocabulary sizes
    
    Args:
        perplexity: Raw perplexity
        vocab_size: Vocabulary size
    Returns:
        Normalized perplexity (0-1, lower is better)
    """
    return perplexity / vocab_size


# ==================== PERPLEXITY COMPARISON ====================

def compare_perplexities(perplexities: Dict[str, float]) -> None:
    """
    Compare perplexities from different models
    
    Args:
        perplexities: Dictionary mapping model names to perplexities
    """
    print("Perplexity Comparison:")
    print("=" * 60)
    
    # Sort by perplexity (lower is better)
    sorted_models = sorted(perplexities.items(), key=lambda x: x[1])
    
    for model_name, pp in sorted_models:
        print(f"{model_name:30s}: {pp:.2f}")
    
    print("\nLower perplexity = better model")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("Perplexity Implementation")
    print("=" * 60)
    
    # Example 1: Simple perplexity computation
    print("\n1. Simple Perplexity:")
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    
    # Random logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    pp = perplexity_from_logits(logits, targets)
    print(f"Perplexity: {pp:.2f}")
    
    # Example 2: Bits per token
    print("\n2. Bits per Token:")
    bpt = bits_per_token(logits, targets)
    print(f"Bits per token: {bpt:.2f}")
    print(f"Perplexity (from BPT): {2**bpt:.2f}")
    print(f"Verification: {pp:.2f} ≈ {2**bpt:.2f}")
    
    # Example 3: Per-token perplexity
    print("\n3. Per-Token Perplexity:")
    per_token_pp = per_token_perplexity(logits, targets)
    print(f"Shape: {per_token_pp.shape}")
    print(f"Mean: {per_token_pp.mean():.2f}")
    print(f"Std: {per_token_pp.std():.2f}")
    
    # Example 4: Normalized perplexity
    print("\n4. Normalized Perplexity:")
    norm_pp = normalized_perplexity(pp, vocab_size)
    print(f"Raw perplexity: {pp:.2f}")
    print(f"Normalized: {norm_pp:.4f}")
    print(f"Interpretation: {norm_pp*100:.2f}% of vocabulary size")
    
    # Example 5: Comparison
    print("\n5. Model Comparison:")
    model_perplexities = {
        'Random': vocab_size,
        'Unigram': vocab_size * 0.8,
        'Bigram': vocab_size * 0.3,
        'GPT-2 Small': 30.0,
        'GPT-2 Large': 18.0,
        'GPT-3': 12.0
    }
    compare_perplexities(model_perplexities)
    
    print("\n" + "=" * 60)
    print("Key Insights:")
    print("=" * 60)
    print("1. Perplexity = exp(average negative log-likelihood)")
    print("2. Lower perplexity = better model")
    print("3. Typical values: 10-50 for good language models")
    print("4. Random baseline: vocabulary_size")
    print("5. Bits per token = log₂(perplexity)")

