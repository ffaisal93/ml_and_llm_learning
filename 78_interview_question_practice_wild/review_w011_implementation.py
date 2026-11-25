import numpy as np


def top_p_filter(logits, p, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("finite nonempty vector required")
    if not 0 < p <= 1 or not (temperature == 0 or 0.01 <= temperature <= 100):
        raise ValueError("invalid p or temperature")
    if temperature == 0:
        return [int(np.argmax(z))], np.array([1.0])
    scaled = z / temperature
    weights = np.exp(scaled - scaled.max())
    probabilities = weights / weights.sum(dtype=np.float64)
    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(probabilities[order], dtype=np.float64)
    count = min(
        len(z), max(1, int(np.searchsorted(cumulative, p - 1e-12, side="left")) + 1)
    )
    selected = order[:count]
    kept = probabilities[selected]
    return selected.tolist(), kept / kept.sum(dtype=np.float64)


def top_p_sample(logits, p, temperature=1.0, rng=None):
    ids, probabilities = top_p_filter(logits, p, temperature)
    if temperature == 0:
        return ids[0]
    if rng is None:
        rng = np.random.default_rng()
    u = float(rng.random())
    cumulative = 0.0
    for token, probability in zip(ids, probabilities):
        cumulative += float(probability)
        if cumulative > u:
            return int(token)
    return int(ids[-1])


class FixedRng:
    def __init__(self, u):
        self.u = u
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.u


ids, p = top_p_filter([2, 1, 0.5, -1], 0.8)
assert ids == [0, 1] and np.allclose(p, [0.73105858, 0.26894142])
r = FixedRng(0.5)
assert top_p_sample([2, 1, 0.5, -1], 0.8, rng=r) == 0 and r.calls == 1
assert top_p_sample([2, 1, 0.5, -1], 0.8, rng=FixedRng(0.9)) == 1
assert top_p_sample([1, 3, 3], 0.5, rng=FixedRng(0.5)) == 2
r = FixedRng(0.5)
assert top_p_sample([1, 3, 3], 0.9, temperature=0, rng=r) == 1 and r.calls == 0
assert top_p_filter([0, 0], 0.5)[0] == [0]
assert top_p_filter([0, 0], 1)[0] == [0, 1]
assert top_p_filter([1000, -1000], 1, 0.01)[0] == [0]
for temp in [-1, 0.001, 101, float("nan")]:
    try:
        top_p_filter([0], 0.5, temp)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid temperature accepted")
print(
    "W011: literal examples, ties, tolerance, exact RNG count and temperature boundaries passed"
)
