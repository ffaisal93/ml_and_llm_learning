"""
Small helpers for frontier-training interview reasoning.

The goal is not to simulate a training run.
The goal is to make trade-offs concrete enough to talk through.
"""

from dataclasses import dataclass
from typing import Dict, List


BYTES_FP16 = 2


@dataclass
class ModelShape:
    num_layers: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int


def kv_cache_bytes_per_sequence(shape: ModelShape, seq_len: int, bytes_per_value: int = BYTES_FP16) -> int:
    """
    Estimate KV-cache size for one sequence.

    Per layer we store keys and values.
    """
    values_per_token = 2 * shape.num_kv_heads * shape.head_dim
    return shape.num_layers * seq_len * values_per_token * bytes_per_value


def active_moe_parameters(num_experts: int, expert_params: int, experts_per_token: int) -> int:
    """Approximate active expert parameters per token."""
    experts_per_token = min(experts_per_token, num_experts)
    return experts_per_token * expert_params


def dense_parameters(hidden_params: int) -> int:
    """Return active dense parameters for a token path."""
    return hidden_params


def ablation_matrix(base_config: Dict[str, str], changes: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Build a one-change-at-a-time ablation matrix.
    """
    rows = [dict(base_config)]
    for field, options in changes.items():
        for option in options:
            row = dict(base_config)
            row[field] = option
            rows.append(row)
    return rows


def strongest_justified_conclusion(mean_gain: float, std_gain: float, changed_variables: int) -> str:
    """
    Return a cautious interview-style conclusion string.
    """
    if changed_variables > 1:
        return "Observed gain is real enough to investigate, but causal attribution is weak because multiple variables changed."
    if mean_gain <= 0:
        return "Current evidence does not support an improvement."
    if std_gain > mean_gain:
        return "Average gain is positive, but variance is too large for a strong claim."
    return "Current evidence supports a real improvement under the tested setup, but generalization beyond this setup still needs more checks."


def demo() -> None:
    mha = ModelShape(num_layers=32, num_query_heads=32, num_kv_heads=32, head_dim=128)
    gqa = ModelShape(num_layers=32, num_query_heads=32, num_kv_heads=8, head_dim=128)
    seq_len = 8192

    print("MHA KV cache bytes:", kv_cache_bytes_per_sequence(mha, seq_len))
    print("GQA KV cache bytes:", kv_cache_bytes_per_sequence(gqa, seq_len))
    print("MoE active params:", active_moe_parameters(num_experts=64, expert_params=10_000_000, experts_per_token=2))
    print(
        strongest_justified_conclusion(
            mean_gain=0.8,
            std_gain=0.2,
            changed_variables=1,
        )
    )


if __name__ == "__main__":
    demo()
