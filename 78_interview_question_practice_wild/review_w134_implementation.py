import numpy as np


def cluster_patches(image, p, K, n_iters):
    a = np.asarray(image)
    if a.ndim != 2 or p < 1 or n_iters < 1:
        raise ValueError("invalid image/iteration configuration")
    H, W = a.shape
    if H % p or W % p:
        raise ValueError("image dimensions must divide by p")
    rows, columns = H // p, W // p
    if not 1 <= K <= rows * columns:
        raise ValueError("invalid K")
    # (grid row,pixel row,grid column,pixel column) -> patch order.
    X = (
        a.reshape(rows, p, columns, p)
        .transpose(0, 2, 1, 3)
        .reshape(-1, p * p)
        .astype(np.float64)
    )
    if not np.isfinite(X).all():
        raise ValueError("finite pixels required")
    centers = X[:K].copy()
    previous = None
    for _ in range(n_iters):
        # Accumulate coordinate-wise to match direct squared distances without
        # cancellation or a potentially huge N*K*p*p temporary.
        distances = np.zeros((len(X), K), dtype=np.float64)
        for feature in range(X.shape[1]):
            distances += (X[:, feature, None] - centers[None, :, feature]) ** 2
        labels = distances.argmin(axis=1)
        if previous is not None and np.array_equal(previous, labels):
            break
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, X)
        counts = np.bincount(labels, minlength=K)
        active = counts > 0
        centers[active] = sums[active] / counts[active, None]
        previous = labels.copy()
    return labels.reshape(rows, columns)


images = [
    [[0, 0, 9, 9], [0, 0, 9, 9], [1, 1, 8, 8], [1, 1, 8, 8]],
    [[0, 0, 4, 4], [0, 0, 4, 4], [2, 2, 4, 4], [2, 2, 4, 4]],
    [[5, 5, 5, 5], [5, 5, 5, 5], [7, 7, 0, 0], [7, 7, 0, 0]],
]
assert cluster_patches(np.array(images[0], dtype=np.uint8), 2, 2, 3).tolist() == [
    [0, 1],
    [0, 1],
]
assert cluster_patches(np.array(images[1]), 2, 2, 1).tolist() == [[0, 1], [0, 1]]
assert cluster_patches(np.array(images[2]), 2, 2, 2).tolist() == [[1, 1], [1, 0]]
assert cluster_patches(np.array(images[2]), 2, 2, 1).tolist() == [[0, 0], [0, 0]]
assert cluster_patches(
    np.arange(255 * 255, dtype=float).reshape(255, 255), 15, 10, 2
).shape == (17, 17)
for args in [
    (np.zeros((2, 2, 1)), 1, 1, 1),
    (np.zeros((3, 2)), 2, 1, 1),
    (np.zeros((2, 2)), 0, 1, 1),
    (np.zeros((2, 2)), 1, 5, 1),
    (np.zeros((2, 2)), 1, 1, 0),
]:
    try:
        cluster_patches(*args)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid input accepted")
print("W134: all examples, uint8, shape and invalid-input checks passed")
