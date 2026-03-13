"""
NLP Evaluation Metrics: BLEU, ROUGE, and Task-Specific Metrics
Complete implementations with detailed explanations
"""
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import math

# ==================== BLEU SCORE ====================

def ngram_precision(candidate: List[str], reference: List[str], n: int) -> float:
    """
    Compute n-gram precision
    
    Precision = (number of matching n-grams) / (total n-grams in candidate)
    
    Example:
    candidate: ["the", "cat", "sat"]
    reference: ["the", "cat", "sat", "on", "the", "mat"]
    bigrams candidate: [("the", "cat"), ("cat", "sat")]
    bigrams reference: [("the", "cat"), ("cat", "sat"), ("sat", "on"), ...]
    matching: 2, total: 2 → precision = 1.0
    """
    # Create n-grams
    candidate_ngrams = Counter()
    for i in range(len(candidate) - n + 1):
        ngram = tuple(candidate[i:i+n])
        candidate_ngrams[ngram] += 1
    
    reference_ngrams = Counter()
    for i in range(len(reference) - n + 1):
        ngram = tuple(reference[i:i+n])
        reference_ngrams[ngram] += 1
    
    # Count matches (clip to reference count)
    matches = 0
    total = sum(candidate_ngrams.values())
    
    for ngram, count in candidate_ngrams.items():
        matches += min(count, reference_ngrams.get(ngram, 0))
    
    return matches / total if total > 0 else 0.0

def brevity_penalty(candidate: List[str], reference: List[str]) -> float:
    """
    Brevity Penalty (BP)
    
    Penalizes short translations
    
    BP = 1 if candidate_length > reference_length
    BP = exp(1 - reference_length / candidate_length) otherwise
    """
    candidate_len = len(candidate)
    reference_len = len(reference)
    
    if candidate_len > reference_len:
        return 1.0
    else:
        return math.exp(1 - reference_len / candidate_len) if candidate_len > 0 else 0.0

def bleu_score(candidate: List[str], reference: List[str], 
               max_n: int = 4, weights: List[float] = None) -> float:
    """
    BLEU Score: Bilingual Evaluation Understudy
    
    Measures quality of machine translation (or text generation)
    
    Formula:
    BLEU = BP * exp(Σ w_n * log(p_n))
    
    Where:
    - BP: Brevity penalty
    - p_n: n-gram precision for n=1,2,3,4
    - w_n: Weights (usually [0.25, 0.25, 0.25, 0.25])
    
    Range: 0 to 1 (higher is better)
    
    Interpretation:
    - 1.0: Perfect match
    - 0.5-0.7: Good translation
    - <0.3: Poor translation
    """
    if weights is None:
        weights = [1.0 / max_n] * max_n
    
    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        prec = ngram_precision(candidate, reference, n)
        precisions.append(prec)
    
    # Compute geometric mean
    log_precisions = [math.log(p) if p > 0 else -float('inf') for p in precisions]
    geometric_mean = sum(w * log_p for w, log_p in zip(weights, log_precisions))
    
    # Apply brevity penalty
    bp = brevity_penalty(candidate, reference)
    
    # BLEU score
    bleu = bp * math.exp(geometric_mean) if geometric_mean > -float('inf') else 0.0
    
    return bleu

def bleu_example():
    """
    BLEU score example
    """
    print("BLEU Score Example")
    print("=" * 60)
    
    reference = ["the", "cat", "sat", "on", "the", "mat"]
    
    candidates = [
        ["the", "cat", "sat", "on", "the", "mat"],  # Perfect
        ["the", "cat", "sat", "on", "the", "mat", "today"],  # Extra word
        ["the", "cat", "sat"],  # Too short
        ["a", "dog", "ran", "on", "the", "mat"],  # Different words
    ]
    
    print(f"Reference: {' '.join(reference)}")
    print()
    
    for i, candidate in enumerate(candidates, 1):
        bleu = bleu_score(candidate, reference)
        print(f"Candidate {i}: {' '.join(candidate)}")
        print(f"  BLEU: {bleu:.4f}")
        print()

# ==================== ROUGE SCORE ====================

def rouge_n(candidate: List[str], reference: List[str], n: int) -> Tuple[float, float, float]:
    """
    ROUGE-N: Recall-Oriented Understudy for Gisting Evaluation
    
    Measures overlap of n-grams between candidate and reference
    
    ROUGE-N = (number of overlapping n-grams) / (number of n-grams in reference)
    
    Returns: (precision, recall, f1)
    """
    # Create n-grams
    candidate_ngrams = Counter()
    for i in range(len(candidate) - n + 1):
        ngram = tuple(candidate[i:i+n])
        candidate_ngrams[ngram] += 1
    
    reference_ngrams = Counter()
    for i in range(len(reference) - n + 1):
        ngram = tuple(reference[i:i+n])
        reference_ngrams[ngram] += 1
    
    # Count overlaps
    overlaps = 0
    for ngram in candidate_ngrams:
        overlaps += min(candidate_ngrams[ngram], reference_ngrams.get(ngram, 0))
    
    # Precision, Recall, F1
    candidate_count = sum(candidate_ngrams.values())
    reference_count = sum(reference_ngrams.values())
    
    precision = overlaps / candidate_count if candidate_count > 0 else 0.0
    recall = overlaps / reference_count if reference_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def rouge_l(candidate: List[str], reference: List[str]) -> Tuple[float, float, float]:
    """
    ROUGE-L: Longest Common Subsequence (LCS)
    
    Measures similarity based on longest common subsequence
    
    ROUGE-L = LCS(candidate, reference) / length(reference)
    
    LCS: Longest sequence of words that appear in both in same order
    (but not necessarily contiguous)
    
    Example:
    candidate: ["the", "cat", "sat", "on", "mat"]
    reference: ["the", "cat", "sat", "on", "the", "mat"]
    LCS: ["the", "cat", "sat", "on", "mat"] (length 5)
    """
    def lcs_length(seq1, seq2):
        """Compute LCS length using dynamic programming"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs = lcs_length(candidate, reference)
    candidate_len = len(candidate)
    reference_len = len(reference)
    
    precision = lcs / candidate_len if candidate_len > 0 else 0.0
    recall = lcs / reference_len if reference_len > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def rouge_example():
    """
    ROUGE score example
    """
    print("\nROUGE Score Example")
    print("=" * 60)
    
    reference = ["the", "cat", "sat", "on", "the", "mat"]
    candidate = ["the", "cat", "sat", "on", "mat"]
    
    print(f"Reference: {' '.join(reference)}")
    print(f"Candidate: {' '.join(candidate)}")
    print()
    
    # ROUGE-1 (unigram)
    p1, r1, f1_1 = rouge_n(candidate, reference, n=1)
    print(f"ROUGE-1: P={p1:.4f}, R={r1:.4f}, F1={f1_1:.4f}")
    
    # ROUGE-2 (bigram)
    p2, r2, f1_2 = rouge_n(candidate, reference, n=2)
    print(f"ROUGE-2: P={p2:.4f}, R={r2:.4f}, F1={f1_2:.4f}")
    
    # ROUGE-L (LCS)
    pl, rl, f1_l = rouge_l(candidate, reference)
    print(f"ROUGE-L: P={pl:.4f}, R={rl:.4f}, F1={f1_l:.4f}")
    print()
    print("Interpretation:")
    print("  - ROUGE-1: Word overlap")
    print("  - ROUGE-2: Bigram overlap")
    print("  - ROUGE-L: Longest common subsequence (order matters)")

# ==================== TASK-SPECIFIC METRICS ====================

def exact_match(prediction: str, reference: str) -> float:
    """
    Exact Match (EM): For question answering
    
    Returns 1 if prediction exactly matches reference, 0 otherwise
    """
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

def f1_score_qa(prediction: str, reference: str) -> float:
    """
    F1 Score for QA: Token-level F1
    
    Computes F1 based on token overlap (not exact match)
    Useful when multiple correct answers exist
    """
    pred_tokens = set(prediction.strip().lower().split())
    ref_tokens = set(reference.strip().lower().split())
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return 1.0
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    intersection = pred_tokens & ref_tokens
    
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(ref_tokens)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def code_bleu(prediction: List[str], reference: List[str]) -> float:
    """
    CodeBLEU: BLEU adapted for code generation
    
    Considers:
    - N-gram match (like BLEU)
    - Syntax match (AST similarity)
    - Semantic match (data flow)
    
    Simplified version: Uses n-gram BLEU
    """
    return bleu_score(prediction, reference, max_n=4)

def perplexity(log_probs: List[float]) -> float:
    """
    Perplexity: For language modeling
    
    Measures how well a probability model predicts a sample.
    It's the exponentiated average negative log-likelihood.
    
    Mathematical Formulation:
    PP = exp(-(1/N) * Σ log P(w_i | context))
    
    Intuition:
    - Perplexity = k means model is as uncertain as uniform choice among k options
    - Lower perplexity = better model (more confident predictions)
    - Typical values: 10-50 for good language models
    
    Connection to Entropy:
    - PP = 2^H (where H is cross-entropy in bits)
    - Perplexity measures uncertainty in "effective vocabulary size"
    
    Args:
        log_probs: List of log probabilities of true tokens
    Returns:
        Perplexity (scalar, lower is better)
    
    Example:
        >>> log_probs = [-2.0, -1.5, -2.5, -1.8]  # Example log probabilities
        >>> pp = perplexity(log_probs)
        >>> print(f"Perplexity: {pp:.2f}")  # Lower is better
    
    See also:
        - 03_evaluation_metrics/perplexity_detailed.md for complete theory
        - 03_evaluation_metrics/perplexity_code.py for full implementation
    """
    if len(log_probs) == 0:
        return float('inf')
    
    avg_log_prob = np.mean(log_probs)
    return math.exp(-avg_log_prob)

# ==================== USAGE ====================

if __name__ == "__main__":
    print("NLP Evaluation Metrics")
    print("=" * 60)
    
    # BLEU
    bleu_example()
    
    # ROUGE
    rouge_example()
    
    # Task-specific
    print("\nTask-Specific Metrics")
    print("=" * 60)
    
    # QA metrics
    pred_qa = "the cat sat on the mat"
    ref_qa = "the cat sat on the mat"
    print(f"QA - Exact Match: {exact_match(pred_qa, ref_qa):.4f}")
    print(f"QA - F1 Score: {f1_score_qa(pred_qa, ref_qa):.4f}")
    
    # Code generation
    pred_code = ["def", "add", "(", "a", ",", "b", ")", ":", "return", "a", "+", "b"]
    ref_code = ["def", "add", "(", "x", ",", "y", ")", ":", "return", "x", "+", "y"]
    print(f"CodeBLEU: {code_bleu(pred_code, ref_code):.4f}")
    
    # Perplexity
    log_probs = [-2.0, -1.5, -2.5, -1.8]  # Example log probabilities
    print(f"Perplexity: {perplexity(log_probs):.4f}")

