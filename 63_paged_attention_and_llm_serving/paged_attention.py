"""
Tiny helpers for paged-attention and serving interviews.

These functions are intentionally simple and pressure-friendly.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List


BYTES_FP16 = 2


def kv_bytes(num_layers: int, num_kv_heads: int, head_dim: int, seq_len: int, bytes_per_value: int = BYTES_FP16) -> int:
    """Estimate KV-cache bytes for one sequence."""
    return num_layers * seq_len * 2 * num_kv_heads * head_dim * bytes_per_value


def contiguous_reservation_waste(reserved_tokens: int, used_tokens: int) -> int:
    """Estimate token slots wasted by pessimistic contiguous reservation."""
    return max(0, reserved_tokens - used_tokens)


def blocks_needed(seq_len: int, block_size: int) -> int:
    """How many fixed-size KV blocks are needed."""
    return (seq_len + block_size - 1) // block_size


def paged_reserved_tokens(seq_len: int, block_size: int) -> int:
    """Rounded-up token capacity under fixed-size blocks."""
    return blocks_needed(seq_len, block_size) * block_size


@dataclass
class Request:
    request_id: str
    remaining_tokens: int


def continuous_batch_schedule(initial_requests: List[Request], arriving_requests: List[Request], max_batch_size: int) -> List[List[str]]:
    """
    Toy continuous-batching scheduler.

    Each step decodes one token for every active request and admits new requests
    whenever capacity is available.
    """
    active: Deque[Request] = deque(initial_requests[:max_batch_size])
    pending: Deque[Request] = deque(arriving_requests)
    steps: List[List[str]] = []

    while active or pending:
        while len(active) < max_batch_size and pending:
            active.append(pending.popleft())

        step_ids: List[str] = []
        next_active: Deque[Request] = deque()

        while active:
            request = active.popleft()
            step_ids.append(request.request_id)
            request.remaining_tokens -= 1
            if request.remaining_tokens > 0:
                next_active.append(request)

        steps.append(step_ids)
        active = next_active

    return steps


def demo() -> None:
    print("KV bytes:", kv_bytes(num_layers=32, num_kv_heads=8, head_dim=128, seq_len=4096))
    print("Contiguous waste:", contiguous_reservation_waste(reserved_tokens=8192, used_tokens=3000))
    print("Blocks needed:", blocks_needed(seq_len=3000, block_size=128))
    print("Paged reserved tokens:", paged_reserved_tokens(seq_len=3000, block_size=128))

    schedule = continuous_batch_schedule(
        initial_requests=[Request("a", 3), Request("b", 1)],
        arriving_requests=[Request("c", 2), Request("d", 1)],
        max_batch_size=2,
    )
    print("Schedule:", schedule)


if __name__ == "__main__":
    demo()
