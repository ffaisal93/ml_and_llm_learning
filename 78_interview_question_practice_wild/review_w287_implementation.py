"""No third-party dependencies; exact replicated-edge Gaussian filter."""

import math
import sys


def gaussian_filter(image, k, sigma):
    if (
        not isinstance(k, int)
        or k < 1
        or k % 2 == 0
        or not math.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError("positive odd K and finite positive sigma required")
    h = len(image)
    w = len(image[0]) if h else 0
    if h == 0 or w == 0 or any(len(row) != w for row in image):
        raise ValueError("nonempty rectangular image required")
    radius = k // 2

    # Equivalent separable kernel; avoids underflow in sigma*sigma for tiny sigma.
    def weight(d):
        scaled = abs(d) / sigma
        return 0.0 if scaled > 40 else math.exp(-0.5 * scaled * scaled)

    weights = [weight(d) for d in range(-radius, radius + 1)]
    norm = sum(weights)
    weights = [v / norm for v in weights]
    temp = [
        [
            sum(
                weights[j] * image[y][min(w - 1, max(0, x + j - radius))]
                for j in range(k)
            )
            for x in range(w)
        ]
        for y in range(h)
    ]
    return [
        [
            sum(
                weights[j] * temp[min(h - 1, max(0, y + j - radius))][x]
                for j in range(k)
            )
            for x in range(w)
        ]
        for y in range(h)
    ]


def solution():
    tokens = sys.stdin.read().split()
    if len(tokens) < 4:
        raise ValueError("expected H W K sigma and pixels")
    h, w, k = map(int, tokens[:3])
    sigma = float(tokens[3])
    if h < 1 or w < 1 or len(tokens) != 4 + h * w:
        raise ValueError("pixel count mismatch")
    values = list(map(float, tokens[4:]))
    image = [values[y * w : (y + 1) * w] for y in range(h)]
    for row in gaussian_filter(image, k, sigma):
        print(" ".join(f"{v:.6f}" for v in row))


if __name__ == "__main__":
    solution()
