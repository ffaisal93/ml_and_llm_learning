"""
Language Modeling Training Losses: Complete Implementations
MLM, CLM, NSP with detailed explanations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

# ==================== MLM (MASKED LANGUAGE MODELING) ====================

def mlm_loss(logits: torch.Tensor,
             input_ids: torch.Tensor,
             masked_positions: torch.Tensor,
             labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
    """
    Masked Language Modeling (MLM) Loss
    
    USED IN: BERT, RoBERTa, ALBERT
    
    WHAT IT DOES:
    - Randomly masks tokens in input
    - Predicts masked tokens from bidirectional context
    - Model can see both left and right context
    
    MATHEMATICAL FORMULATION:
    L_MLM = -∑_{i in masked} log P(x_i | x_context)
    
    Where:
    - x_i: Masked token at position i
    - x_context: All other tokens (bidirectional)
    
    PROCESS:
    1. Mask random tokens (typically 15% of tokens)
    2. Replace with [MASK] token (80%), random token (10%), or keep original (10%)
    3. Model predicts original token for each masked position
    4. Compute cross-entropy loss only on masked positions
    
    Args:
        logits: Model output, shape (batch, seq_len, vocab_size)
        input_ids: Original input token IDs, shape (batch, seq_len)
        masked_positions: Boolean mask indicating which positions are masked, shape (batch, seq_len)
        labels: Optional ground truth labels (if None, uses input_ids)
    
    Returns:
        loss: MLM loss (scalar)
        info: Dictionary with statistics
    """
    if labels is None:
        labels = input_ids
    
    # Get logits only for masked positions
    # masked_positions: (batch, seq_len) - True where masked
    # logits: (batch, seq_len, vocab_size)
    # We need to select logits at masked positions
    
    # Flatten for easier indexing
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
    labels_flat = labels.view(-1)  # (batch * seq_len)
    masked_flat = masked_positions.view(-1)  # (batch * seq_len)
    
    # Select only masked positions
    masked_logits = logits_flat[masked_flat]  # (num_masked, vocab_size)
    masked_labels = labels_flat[masked_flat]  # (num_masked,)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(masked_logits, masked_labels, reduction='mean')
    
    # Compute accuracy
    with torch.no_grad():
        predictions = masked_logits.argmax(dim=-1)
        accuracy = (predictions == masked_labels).float().mean()
        num_masked = masked_flat.sum().item()
    
    info = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'num_masked': num_masked,
        'masking_rate': num_masked / (batch_size * seq_len)
    }
    
    return loss, info


def create_mlm_mask(input_ids: torch.Tensor,
                    mask_token_id: int,
                    vocab_size: int,
                    mask_prob: float = 0.15,
                    random_replace_prob: float = 0.1,
                    keep_original_prob: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create MLM mask and apply masking strategy
    
    MASKING STRATEGY (BERT-style):
    - 15% of tokens are selected for masking
    - Of those 15%:
      - 80% replaced with [MASK] token
      - 10% replaced with random token
      - 10% kept as original (but still predicted)
    
    This strategy helps the model:
    - Learn from [MASK] tokens (main signal)
    - Handle real tokens during fine-tuning (10% original)
    - Be robust to noise (10% random)
    
    Args:
        input_ids: Input token IDs, shape (batch, seq_len)
        mask_token_id: ID of [MASK] token
        vocab_size: Vocabulary size (for random replacement)
        mask_prob: Probability of masking a token (default 0.15)
        random_replace_prob: Probability of random replacement (default 0.1)
        keep_original_prob: Probability of keeping original (default 0.1)
    
    Returns:
        masked_input_ids: Input with masking applied, shape (batch, seq_len)
        masked_positions: Boolean mask of masked positions, shape (batch, seq_len)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # Create random mask (15% of tokens)
    random_probs = torch.rand(batch_size, seq_len, device=device)
    masked_positions = random_probs < mask_prob
    
    # Don't mask special tokens (simplified - in practice would exclude [CLS], [SEP], etc.)
    # For now, we'll mask all positions that pass the probability threshold
    
    # Create masked input
    masked_input_ids = input_ids.clone()
    
    # Determine replacement strategy for each masked position
    strategy_probs = torch.rand(batch_size, seq_len, device=device)
    
    # 80%: Replace with [MASK] token
    mask_replace = masked_positions & (strategy_probs < 0.8)
    masked_input_ids[mask_replace] = mask_token_id
    
    # 10%: Replace with random token
    random_replace = masked_positions & (strategy_probs >= 0.8) & (strategy_probs < 0.9)
    random_tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    masked_input_ids[random_replace] = random_tokens[random_replace]
    
    # 10%: Keep original (but still predict it)
    # No change needed - already original
    
    return masked_input_ids, masked_positions


# ==================== CLM (CAUSAL LANGUAGE MODELING) ====================

def clm_loss(logits: torch.Tensor,
             input_ids: torch.Tensor,
             shift_labels: bool = True) -> Tuple[torch.Tensor, dict]:
    """
    Causal Language Modeling (CLM) Loss
    
    USED IN: GPT, LLaMA, all autoregressive models
    
    WHAT IT DOES:
    - Predicts next token given previous tokens
    - Autoregressive: only sees left context
    - Standard language modeling objective
    
    MATHEMATICAL FORMULATION:
    L_CLM = -∑_{t=1}^T log P(x_t | x_{<t})
    
    Where:
    - x_t: Token at position t
    - x_{<t}: All tokens before position t
    
    PROCESS:
    1. Input: [x_1, x_2, ..., x_T]
    2. Predict: [x_2, x_3, ..., x_{T+1}]
    3. Compute cross-entropy loss
    4. Each position predicts the next token
    
    KEY DIFFERENCE FROM MLM:
    - MLM: Bidirectional, predicts masked tokens
    - CLM: Unidirectional, predicts next token sequentially
    
    Args:
        logits: Model output, shape (batch, seq_len, vocab_size)
        input_ids: Input token IDs, shape (batch, seq_len)
        shift_labels: If True, shift labels for next-token prediction (default True)
    
    Returns:
        loss: CLM loss (scalar)
        info: Dictionary with statistics
    """
    if shift_labels:
        # Shift input_ids to create labels (predict next token)
        # Input:  [x_1, x_2, x_3, ..., x_T]
        # Labels: [x_2, x_3, x_4, ..., x_{T+1}]
        # We predict token at position t+1 given tokens up to position t
        labels = input_ids[:, 1:].contiguous()  # (batch, seq_len-1)
        logits = logits[:, :-1, :].contiguous()  # (batch, seq_len-1, vocab_size)
    else:
        labels = input_ids
        # Assume logits already aligned
    
    # Flatten for cross-entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)  # (batch * seq_len, vocab_size)
    labels_flat = labels.view(-1)  # (batch * seq_len)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
    
    # Compute perplexity and accuracy
    with torch.no_grad():
        predictions = logits_flat.argmax(dim=-1)
        accuracy = (predictions == labels_flat).float().mean()
        
        # Perplexity: exp(cross_entropy)
        perplexity = torch.exp(loss)
    
    info = {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item(),
        'num_tokens': batch_size * seq_len
    }
    
    return loss, info


# ==================== NSP (NEXT SENTENCE PREDICTION) ====================

def nsp_loss(logits: torch.Tensor,
             labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Next Sentence Prediction (NSP) Loss
    
    USED IN: BERT (original), less common now
    
    WHAT IT DOES:
    - Binary classification: Does sentence B follow sentence A?
    - Helps model understand sentence relationships
    - Trained on sentence pairs
    
    MATHEMATICAL FORMULATION:
    L_NSP = -log P(is_next | sentence_A, sentence_B)
    
    Where:
    - is_next: Binary label (1 if B follows A, 0 if random)
    
    PROCESS:
    1. Input: [CLS] sentence_A [SEP] sentence_B [SEP]
    2. Use [CLS] token representation
    3. Binary classification: is_next or not
    4. 50% positive (B follows A), 50% negative (B is random)
    
    NOTE:
    - Less commonly used now (RoBERTa removed it)
    - Replaced by other objectives or longer sequences
    - Still useful for understanding sentence pairs
    
    Args:
        logits: Binary classification logits, shape (batch, 2) or (batch, 1)
        labels: Binary labels (0 or 1), shape (batch,)
    
    Returns:
        loss: NSP loss (scalar)
        info: Dictionary with statistics
    """
    # Handle both (batch, 2) and (batch, 1) logits
    if logits.shape[-1] == 1:
        # Binary classification with single logit
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), labels.float(), reduction='mean'
        )
        probs = torch.sigmoid(logits.squeeze(-1))
        predictions = (probs > 0.5).long()
    else:
        # Binary classification with 2 logits
        loss = F.cross_entropy(logits, labels, reduction='mean')
        predictions = logits.argmax(dim=-1)
    
    # Compute accuracy
    with torch.no_grad():
        accuracy = (predictions == labels).float().mean()
        num_positive = labels.sum().item()
        num_negative = (labels == 0).sum().item()
    
    info = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'num_positive': num_positive,
        'num_negative': num_negative
    }
    
    return loss, info


def create_nsp_pairs(sentences: list, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sentence pairs for NSP training
    
    Creates positive pairs (B follows A) and negative pairs (B is random)
    
    Args:
        sentences: List of sentences (each as list of token IDs)
        vocab_size: Vocabulary size (for creating random sentences)
    
    Returns:
        input_ids: Concatenated sentence pairs, shape (batch, seq_len)
        segment_ids: Segment IDs (0 for sentence A, 1 for sentence B), shape (batch, seq_len)
        labels: NSP labels (1 if next, 0 if random), shape (batch,)
    """
    batch_size = len(sentences) // 2
    device = sentences[0].device if isinstance(sentences[0], torch.Tensor) else 'cpu'
    
    input_ids_list = []
    segment_ids_list = []
    labels_list = []
    
    cls_id = 1  # [CLS] token
    sep_id = 2  # [SEP] token
    
    for i in range(batch_size):
        # Positive pair: sentence i and sentence i+1
        sentence_a = sentences[i * 2]
        sentence_b = sentences[i * 2 + 1]
        
        # Create input: [CLS] A [SEP] B [SEP]
        input_ids = [cls_id] + sentence_a + [sep_id] + sentence_b + [sep_id]
        segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1)
        
        input_ids_list.append(torch.tensor(input_ids, device=device))
        segment_ids_list.append(torch.tensor(segment_ids, device=device))
        labels_list.append(1)  # Positive
        
        # Negative pair: sentence i and random sentence
        sentence_a = sentences[i * 2]
        random_sentence = torch.randint(0, vocab_size, (len(sentence_b),), device=device).tolist()
        
        input_ids = [cls_id] + sentence_a + [sep_id] + random_sentence + [sep_id]
        segment_ids = [0] * (len(sentence_a) + 2) + [1] * (len(random_sentence) + 1)
        
        input_ids_list.append(torch.tensor(input_ids, device=device))
        segment_ids_list.append(torch.tensor(segment_ids, device=device))
        labels_list.append(0)  # Negative
    
    # Pad to same length
    max_len = max(len(ids) for ids in input_ids_list)
    
    padded_input_ids = []
    padded_segment_ids = []
    
    for ids, seg_ids in zip(input_ids_list, segment_ids_list):
        pad_len = max_len - len(ids)
        padded_input_ids.append(F.pad(ids, (0, pad_len), value=0))  # 0 = [PAD]
        padded_segment_ids.append(F.pad(seg_ids, (0, pad_len), value=0))
    
    input_ids = torch.stack(padded_input_ids)
    segment_ids = torch.stack(padded_segment_ids)
    labels = torch.tensor(labels_list, device=device)
    
    return input_ids, segment_ids, labels


# ==================== COMBINED LOSSES ====================

def bert_loss(mlm_logits: torch.Tensor,
              nsp_logits: torch.Tensor,
              input_ids: torch.Tensor,
              masked_positions: torch.Tensor,
              nsp_labels: torch.Tensor,
              mlm_weight: float = 1.0,
              nsp_weight: float = 1.0) -> Tuple[torch.Tensor, dict]:
    """
    Combined BERT loss: MLM + NSP
    
    Original BERT uses both objectives:
    L_BERT = L_MLM + L_NSP
    
    Args:
        mlm_logits: MLM logits, shape (batch, seq_len, vocab_size)
        nsp_logits: NSP logits, shape (batch, 2) or (batch, 1)
        input_ids: Input token IDs, shape (batch, seq_len)
        masked_positions: Mask positions, shape (batch, seq_len)
        nsp_labels: NSP labels, shape (batch,)
        mlm_weight: Weight for MLM loss (default 1.0)
        nsp_weight: Weight for NSP loss (default 1.0)
    
    Returns:
        total_loss: Combined loss
        info: Dictionary with all statistics
    """
    # Compute MLM loss
    mlm_loss_val, mlm_info = mlm_loss(mlm_logits, input_ids, masked_positions)
    
    # Compute NSP loss
    nsp_loss_val, nsp_info = nsp_loss(nsp_logits, nsp_labels)
    
    # Combined loss
    total_loss = mlm_weight * mlm_loss_val + nsp_weight * nsp_loss_val
    
    info = {
        'total_loss': total_loss.item(),
        'mlm_loss': mlm_info,
        'nsp_loss': nsp_info
    }
    
    return total_loss, info


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    print("Language Modeling Training Losses")
    print("=" * 80)
    
    batch_size = 4
    seq_len = 128
    vocab_size = 10000
    
    # ========== MLM Example ==========
    print("\n1. MLM (Masked Language Modeling) Loss")
    print("-" * 80)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    mask_token_id = 103  # [MASK] token ID
    
    # Create mask
    masked_input_ids, masked_positions = create_mlm_mask(
        input_ids, mask_token_id, vocab_size, mask_prob=0.15
    )
    
    # Dummy model output
    mlm_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Compute loss
    mlm_loss_val, mlm_info = mlm_loss(mlm_logits, input_ids, masked_positions)
    
    print(f"MLM Loss: {mlm_loss_val.item():.4f}")
    print(f"MLM Accuracy: {mlm_info['accuracy']:.4f}")
    print(f"Number of masked tokens: {mlm_info['num_masked']}")
    print(f"Masking rate: {mlm_info['masking_rate']:.4f}")
    
    # ========== CLM Example ==========
    print("\n2. CLM (Causal Language Modeling) Loss")
    print("-" * 80)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Dummy model output (predicts next token)
    clm_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Compute loss
    clm_loss_val, clm_info = clm_loss(clm_logits, input_ids)
    
    print(f"CLM Loss: {clm_loss_val.item():.4f}")
    print(f"Perplexity: {clm_info['perplexity']:.2f}")
    print(f"Accuracy: {clm_info['accuracy']:.4f}")
    
    # ========== NSP Example ==========
    print("\n3. NSP (Next Sentence Prediction) Loss")
    print("-" * 80)
    
    # Create dummy NSP logits (binary classification)
    nsp_logits = torch.randn(batch_size, 2)  # (batch, 2) for binary classification
    nsp_labels = torch.randint(0, 2, (batch_size,))  # Binary labels
    
    # Compute loss
    nsp_loss_val, nsp_info = nsp_loss(nsp_logits, nsp_labels)
    
    print(f"NSP Loss: {nsp_loss_val.item():.4f}")
    print(f"NSP Accuracy: {nsp_info['accuracy']:.4f}")
    print(f"Positive examples: {nsp_info['num_positive']}")
    print(f"Negative examples: {nsp_info['num_negative']}")
    
    # ========== Combined BERT Loss ==========
    print("\n4. Combined BERT Loss (MLM + NSP)")
    print("-" * 80)
    
    # Create dummy data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    masked_input_ids, masked_positions = create_mlm_mask(
        input_ids, mask_token_id, vocab_size
    )
    nsp_labels = torch.randint(0, 2, (batch_size,))
    
    mlm_logits = torch.randn(batch_size, seq_len, vocab_size)
    nsp_logits = torch.randn(batch_size, 2)
    
    # Compute combined loss
    total_loss, bert_info = bert_loss(
        mlm_logits, nsp_logits, input_ids, masked_positions, nsp_labels
    )
    
    print(f"Total BERT Loss: {total_loss.item():.4f}")
    print(f"  MLM Loss: {bert_info['mlm_loss']['loss']:.4f}")
    print(f"  NSP Loss: {bert_info['nsp_loss']['loss']:.4f}")
    
    print("\n" + "=" * 80)
    print("Key Differences:")
    print("=" * 80)
    print("""
    MLM (BERT):
    - Bidirectional context
    - Predicts masked tokens
    - Better for understanding tasks
    
    CLM (GPT):
    - Unidirectional context
    - Predicts next token
    - Better for generation tasks
    
    NSP (BERT):
    - Binary classification
    - Sentence pair understanding
    - Less commonly used now
    """)

