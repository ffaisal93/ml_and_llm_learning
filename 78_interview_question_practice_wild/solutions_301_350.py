"""Prompt-specific runnable examples for W301–W350."""

# BEGIN W301
import math


def staying_dry(n, p):
    if not isinstance(n, int) or isinstance(n, bool) or n < 0 or not 0 <= p <= 1:
        raise ValueError("n nonnegative integer, p in [0,1] required")
    a, b = 1.0, 0.0
    for _ in range(2 * n):
        a, b = p * a + (1 - p) * b, (1 - p) * a
    return a + b


# END W301

# BEGIN W302
import numpy as np
from numbers import Real


def _positive(value):
    if (
        not isinstance(value, Real)
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError("finite positive scalar required")
    return float(value)


def _integer(value):
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or value < 1
    ):
        raise ValueError("positive integer required")
    return int(value)


def sample_hit_positions(n: int, screen_x: float = 1.0, rng=None):
    n, screen_x = _integer(n), _positive(screen_x)
    rng = np.random.default_rng() if rng is None else rng
    angle = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    angle = np.clip(angle, np.nextafter(-np.pi / 2, 0), np.nextafter(np.pi / 2, 0))
    return screen_x * np.tan(angle)


def cauchy_pdf(y, scale: float = 1.0):
    scale = _positive(scale)
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        raise ValueError("NaN evaluation position")
    a = np.abs(y)
    result = np.empty_like(a)
    near = a <= scale
    result[near] = ((1 / scale) / np.pi) / (1 + (a[near] / scale) ** 2)
    result[~near] = (
        ((scale / a[~near]) / a[~near]) / np.pi / (1 + (scale / a[~near]) ** 2)
    )
    return result


def density_histogram(hits, limit: float = 20.0, n_bins: int = 200):
    limit, n_bins = _positive(limit), _integer(n_bins)
    hits = np.asarray(hits, dtype=float)
    if hits.ndim != 1 or hits.size == 0 or np.isnan(hits).any():
        raise ValueError("nonempty one-dimensional hits required")
    edges = np.linspace(-limit, limit, n_bins + 1)
    counts, _ = np.histogram(hits, bins=edges)
    return edges, counts / (hits.size * (edges[1] - edges[0]))


def plot_simulation(
    n: int = 200000, screen_x: float = 1.0, limit: float = 20.0, seed: int = 42
):
    import matplotlib.pyplot as plt

    hits = sample_hit_positions(n, screen_x, np.random.default_rng(seed))
    edges, heights = density_histogram(hits, limit)
    width = edges[1] - edges[0]
    plt.bar(
        edges[:-1],
        heights,
        width=width,
        align="edge",
        alpha=0.5,
        label="total-count histogram",
    )
    grid = np.linspace(-limit, limit, 1000)
    plt.plot(grid, cauchy_pdf(grid, screen_x), label="Cauchy PDF")
    plt.xlabel("vertical hit y")
    plt.ylabel("density")
    plt.legend()
    plt.show()


# END W302

# BEGIN W305


def classify_completion(events):
    best = {}
    for user, movie, percent in events:
        key = (user, movie)
        best[key] = max(best.get(key, 0), percent)
    result = []
    for (user, movie), percent in best.items():
        if percent >= 30:
            bucket = 80 if percent >= 80 else 50 if percent >= 50 else 30
            result.append((user, movie, bucket))
    return sorted(result, key=lambda row: (row[1], row[0]))


# END W305

# BEGIN W323
from collections import defaultdict


def cumulative_deduped_time(data):
    groups = defaultdict(list)
    for row in data:
        start, end = row["start_time"], row["end_time"]
        if end < start:
            raise ValueError("end before start")
        groups[row["run_id"]].append((start, end))
    output = {}
    for run, intervals in groups.items():
        intervals.sort()
        start, end = intervals[0]
        total = 0
        for a, b in intervals[1:]:
            if a <= end:
                end = max(end, b)
            else:
                total += end - start
                start, end = a, b
        output[run] = total + end - start
    return output


# END W323

# BEGIN W332
import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
import os
import tempfile


def run_etl(source, destination, reject_path):
    """Demo CSV schema id,name,amount; first valid ID wins; exact decimal cents."""
    seen = set()
    accepted = []
    rejects = []
    with open(source, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"id", "name", "amount"} <= set(reader.fieldnames or []):
            raise ValueError("missing required columns")
        for line, row in enumerate(reader, 2):
            try:
                key = row["id"].strip()
                name = " ".join(row["name"].split())
                amount = Decimal(row["amount"])
                if (
                    not key
                    or not name
                    or not amount.is_finite()
                    or amount < 0
                    or amount != amount.quantize(Decimal(".01"))
                ):
                    raise ValueError(
                        "required fields or nonnegative cent amount invalid"
                    )
                if key in seen:
                    raise ValueError("duplicate id: first valid row wins")
                seen.add(key)
                accepted.append(
                    {
                        "id": key,
                        "name": name,
                        "amount": str(amount.quantize(Decimal(".01"))),
                    }
                )
            except (ValueError, InvalidOperation, TypeError, AttributeError) as e:
                rejects.append({"line": line, "raw": row, "reason": str(e)})
    # Each destination is atomically replaced. A crash between them needs a manifest
    # or transactional sink if the two files must form one atomic publication.
    for path, rows in [(destination, accepted), (reject_path, rejects)]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".etl-")
        try:
            with os.fdopen(fd, "w") as out:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    return {"accepted": len(accepted), "rejected": len(rejects)}


# END W332
