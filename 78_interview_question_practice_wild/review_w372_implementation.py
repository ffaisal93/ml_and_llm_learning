def feasible_phase(arrivals, period=30.0):
    """Observed indexed arrivals only; cannot certify unobserved future lags."""
    if not arrivals:
        raise ValueError("observations required")
    aligned = [time - period * index for index, time in arrivals]
    lower = max(aligned)
    upper = min(aligned) + period
    return (lower, upper) if lower < upper else None


def poll_counts(arrivals, first, period, polls, cursor=-1):
    counts = []
    for j in range(polls):
        now = first + j * period
        ready = [i for i, t in arrivals if i > cursor and t <= now]
        counts.append(len(ready))
        if ready:
            cursor = max(ready)
    return counts


if __name__ == "__main__":
    fixed = [(k, 30 * k + 47) for k in range(100)]
    assert feasible_phase(fixed) == (47, 77)
    assert poll_counts(fixed, 47, 30, 100) == [1] * 100
    jitter = [(k, 30 * k + (10 if k % 2 == 0 else 25)) for k in range(100)]
    assert feasible_phase(jitter) == (25, 40)
    assert poll_counts(jitter, 25, 30, 100) == [1] * 100
    pairs = [(k, 60 * (k // 2) + 50) for k in range(100)]
    assert feasible_phase(pairs) is None
    assert poll_counts(pairs, 50, 30, 4) == [2, 0, 2, 0]
    # Same finite boot prefix, incompatible future: an allowed batching suffix.
    boot = [(k, 30 * k + 10) for k in range(10)]
    assert feasible_phase(boot) == (10, 40)
    future = boot + [(10, 350), (11, 350)]
    assert feasible_phase(future) is None
    print(
        "W372: phase intervals, exact boundaries, batched impossibility and finite-boot counterexample passed"
    )
