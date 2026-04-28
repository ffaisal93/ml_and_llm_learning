"""
Small helpers for reasoning about large-scale LLM memory and serving trade-offs.
"""


def parameter_memory_gb(num_parameters: float, bytes_per_param: float = 2.0) -> float:
    """
    Approximate parameter memory in GB.

    bytes_per_param:
    - 4 for fp32
    - 2 for fp16/bf16
    """
    return (num_parameters * bytes_per_param) / 1e9


def adam_training_memory_gb(num_parameters: float, bytes_per_param: float = 2.0) -> float:
    """
    Rough training-memory estimate for parameters + gradients + Adam moments.

    This is a simplified mental-model calculator, not a precise profiler.
    """
    parameter_mem = parameter_memory_gb(num_parameters, bytes_per_param)
    gradient_mem = parameter_memory_gb(num_parameters, bytes_per_param)
    moment1_mem = parameter_memory_gb(num_parameters, 4.0)
    moment2_mem = parameter_memory_gb(num_parameters, 4.0)
    return parameter_mem + gradient_mem + moment1_mem + moment2_mem


def kv_cache_memory_gb(
    batch_size: int,
    seq_len: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    bytes_per_value: int = 2,
) -> float:
    """
    Approximate KV cache memory for autoregressive inference.

    Keys and values are both stored.
    """
    total_values = 2 * batch_size * seq_len * num_layers * num_heads * head_dim
    return (total_values * bytes_per_value) / 1e9


if __name__ == "__main__":
    print("Large-Scale LLM Systems Tradeoff Helpers")
    print("=" * 60)

    params = 7_000_000_000
    print(f"7B params in bf16 weights only: {parameter_memory_gb(params, bytes_per_param=2):.2f} GB")
    print(f"7B params rough Adam training memory: {adam_training_memory_gb(params, bytes_per_param=2):.2f} GB")

    kv_mem = kv_cache_memory_gb(
        batch_size=8,
        seq_len=4096,
        num_layers=32,
        num_heads=32,
        head_dim=128,
        bytes_per_value=2,
    )
    print(f"Approx KV cache memory: {kv_mem:.2f} GB")
