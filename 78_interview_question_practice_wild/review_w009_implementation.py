import math
import numpy as np


def compute_entropy_blockwise(logits, block_size):
    """Two scalar scans; supports len/index-only lazy inputs and negative infinity."""
    n = len(logits)
    if n == 0 or block_size <= 0:
        raise ValueError("empty input or non-positive block size")
    maximum = -math.inf
    for start in range(0, n, block_size):
        for i in range(start, min(start + block_size, n)):
            x = float(logits[i])
            if math.isnan(x) or x == math.inf:
                raise ValueError("invalid logit")
            maximum = max(maximum, x)
    if maximum == -math.inf:
        raise ValueError("at least one logit must be finite")
    z_sum, weighted_shift = 0.0, 0.0
    for start in range(0, n, block_size):
        for i in range(start, min(start + block_size, n)):
            x = float(logits[i])
            if x == -math.inf:
                continue
            shift = x - maximum
            weight = math.exp(shift)
            z_sum += weight
            # Avoid 0 * -inf after subtraction overflows for extreme finite inputs.
            if weight:
                weighted_shift += weight * shift
    return math.log(z_sum) - weighted_shift / z_sum


assert math.isclose(compute_entropy_blockwise([0, 0], 1), math.log(2))
assert math.isclose(compute_entropy_blockwise([1000, 999], 8), 0.5822031088882179)
assert compute_entropy_blockwise([-math.inf, 4, -math.inf], 1) == 0


class Lazy:
    def __len__(self):
        return 100000

    def __getitem__(self, i):
        if not isinstance(i, int) or not 0 <= i < len(self):
            raise AssertionError("integer indexing only")
        return 0.0

    def __iter__(self):
        raise AssertionError("iteration forbidden")


assert math.isclose(
    compute_entropy_blockwise(Lazy(), 37), math.log(100000), abs_tol=1e-10
)
for bad, block in [
    ([], 1),
    ([0], 0),
    ([math.nan], 1),
    ([math.inf], 1),
    ([-math.inf], 1),
]:
    try:
        compute_entropy_blockwise(bad, block)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid logits accepted")

print("W009: all exact-contract behavior checks passed")
