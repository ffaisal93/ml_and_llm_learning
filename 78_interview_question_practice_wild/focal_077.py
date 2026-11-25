"""W077 exact probability-input CLI."""

import math
import sys


def focal_loss(labels, probabilities, gamma=2.0, alpha=0.25, reduction="mean"):
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Need matching nonempty inputs")
    if gamma < 0 or not 0 <= alpha <= 1:
        raise ValueError("Invalid gamma/alpha")
    losses = []
    for y, p in zip(labels, probabilities):
        if y not in (0, 1) or not 0 <= p <= 1:
            raise ValueError("Invalid label/probability")
        p = min(max(p, 1e-15), 1 - 1e-15)
        pt = p if y else 1 - p
        at = alpha if y else 1 - alpha
        losses.append(-at * (1 - pt) ** gamma * math.log(pt))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return math.fsum(losses)
    if reduction == "mean":
        return math.fsum(losses) / len(losses)
    raise ValueError("Unknown reduction")


if __name__ == "__main__":
    tokens = sys.stdin.read().split()
    n = int(tokens[0])
    gamma = float(tokens[1])
    alpha = float(tokens[2])
    reduction = tokens[3]
    if len(tokens) != 4 + 2 * n:
        raise ValueError("Wrong number of inputs")
    result = focal_loss(
        list(map(int, tokens[4 : 4 + n])),
        list(map(float, tokens[4 + n :])),
        gamma,
        alpha,
        reduction,
    )
    values = result if isinstance(result, list) else [result]
    print(" ".join(f"{value:.6f}" for value in values))
