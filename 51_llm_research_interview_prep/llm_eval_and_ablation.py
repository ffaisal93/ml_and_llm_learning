"""
LLM evaluation and ablation boilerplate.
Small metrics and helpers that are useful in research interviews.
"""
from typing import Dict, Iterable, List, Sequence

import numpy as np


def negative_log_likelihood(token_probabilities: Sequence[float], eps: float = 1e-12) -> float:
    """Average negative log-likelihood over target token probabilities."""
    probs = np.clip(np.array(token_probabilities, dtype=float), eps, 1.0)
    return float(-np.mean(np.log(probs)))


def perplexity_from_nll(avg_nll: float) -> float:
    """Perplexity is exp(average negative log-likelihood)."""
    return float(np.exp(avg_nll))


def exact_match(prediction: str, reference: str) -> float:
    """Exact string match after simple trimming and lowercasing."""
    return float(prediction.strip().lower() == reference.strip().lower())


def token_f1(prediction_tokens: Sequence[str], reference_tokens: Sequence[str]) -> float:
    """
    Token-level F1 for QA-style answers.
    """
    pred_counts = {}
    ref_counts = {}

    for token in prediction_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in reference_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / max(len(prediction_tokens), 1)
    recall = overlap / max(len(reference_tokens), 1)
    return float(2.0 * precision * recall / (precision + recall))


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Standard pass@k estimator used in code generation evaluation.

    n: number of samples
    c: number of correct samples
    k: budget of attempts
    """
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    product = 1.0
    for i in range(k):
        product *= (n - c - i) / (n - i)
    return float(1.0 - product)


def retrieval_recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    """Recall@k for retrieval."""
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return float(len(top_k & relevant) / len(relevant))


def mean_reciprocal_rank(rank_lists: Sequence[Sequence[str]], relevant_ids: Sequence[Sequence[str]]) -> float:
    """
    MRR across queries.
    """
    rr_scores = []
    for ranked, relevant in zip(rank_lists, relevant_ids):
        relevant_set = set(relevant)
        rr = 0.0
        for idx, item in enumerate(ranked, start=1):
            if item in relevant_set:
                rr = 1.0 / idx
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def win_rate(model_a_scores: Sequence[float], model_b_scores: Sequence[float]) -> Dict[str, float]:
    """
    Pairwise win rate summary.
    """
    a_scores = np.array(model_a_scores, dtype=float)
    b_scores = np.array(model_b_scores, dtype=float)
    wins = float(np.mean(a_scores > b_scores))
    losses = float(np.mean(a_scores < b_scores))
    ties = float(np.mean(a_scores == b_scores))
    return {"wins": wins, "losses": losses, "ties": ties}


def ablation_deltas(results: Dict[str, float], baseline_name: str = "baseline") -> Dict[str, float]:
    """Metric deltas relative to a named baseline."""
    baseline = results[baseline_name]
    return {name: float(value - baseline) for name, value in results.items()}


if __name__ == "__main__":
    print("LLM Evaluation and Ablation Boilerplate")
    print("=" * 60)

    token_probs = [0.8, 0.7, 0.9, 0.6]
    nll = negative_log_likelihood(token_probs)
    ppl = perplexity_from_nll(nll)
    print(f"Average NLL: {nll:.4f}")
    print(f"Perplexity:  {ppl:.4f}")

    pred = "the capital of france is paris"
    ref = "Paris"
    print(f"\nExact match: {exact_match(pred, ref)}")
    print(
        "Token F1:",
        token_f1(pred.lower().split(), "paris".split()),
    )

    print(f"\npass@k example: {pass_at_k(n=20, c=3, k=5):.4f}")

    retrieved = ["d4", "d2", "d8", "d1", "d7"]
    relevant = ["d2", "d5"]
    print(f"Recall@3: {retrieval_recall_at_k(retrieved, relevant, k=3):.4f}")

    rank_lists = [["a", "b", "c"], ["x", "y", "z"], ["m", "n", "o"]]
    relevant_lists = [["c"], ["y"], ["q"]]
    print(f"MRR: {mean_reciprocal_rank(rank_lists, relevant_lists):.4f}")

    print("\nPreference win rate")
    print(win_rate([4, 5, 3, 2, 5], [3, 5, 4, 1, 2]))

    ablation_scores = {
        "baseline": 0.712,
        "larger_context": 0.728,
        "better_retriever": 0.741,
        "remove_instruction_tuning": 0.688,
    }
    print("\nAblation deltas")
    print(ablation_deltas(ablation_scores))
