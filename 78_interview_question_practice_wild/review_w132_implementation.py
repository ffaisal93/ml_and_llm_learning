import numpy as np


def kmeans(X, k, max_iters=100, tol=1e-4, init_centroids=None, seed=0):
    X = np.asarray(X, dtype=np.float64)
    if (
        X.ndim != 2
        or not 1 <= k <= len(X)
        or max_iters < 1
        or tol < 0
        or not np.isfinite(X).all()
    ):
        raise ValueError("invalid data/configuration")
    if init_centroids is None:
        centers = X[np.random.default_rng(seed).choice(len(X), k, replace=False)].copy()
    else:
        centers = np.asarray(init_centroids, dtype=np.float64).copy()
    if centers.shape != (k, X.shape[1]) or not np.isfinite(centers).all():
        raise ValueError("invalid initial centers")

    def assign():
        # O(n*k) storage, avoiding an n*k*d temporary.
        distances = (
            (X * X).sum(1)[:, None]
            + (centers * centers).sum(1)[None, :]
            - 2 * X @ centers.T
        )
        return distances.argmin(1)

    for _ in range(max_iters):
        labels = assign()
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, X)
        counts = np.bincount(labels, minlength=k)
        updated = centers.copy()
        active = counts > 0
        updated[active] = sums[active] / counts[active, None]
        movement = np.linalg.norm(updated - centers, axis=1).max()
        centers = updated
        if movement <= tol:
            break
    return centers, assign()


c, l = kmeans(
    [[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]],
    2,
    init_centroids=[[0, 0], [10, 10]],
    tol=1e-6,
)
assert np.allclose(c, [[1 / 3, 1 / 3], [31 / 3, 31 / 3]]) and l.tolist() == [
    0,
    0,
    0,
    1,
    1,
    1,
]
c, l = kmeans([[0], [1], [2], [3], [10]], 2, init_centroids=[[0], [1]], max_iters=1)
assert np.allclose(c, [[0], [4]]) and l.tolist() == [0, 0, 0, 1, 1]
c, l = kmeans([[0], [4]], 2, init_centroids=[[2], [2]])
assert np.allclose(c, [[2], [2]]) and l.tolist() == [0, 0]
assert np.array_equal(
    kmeans([[0], [1], [9], [10]], 2, seed=3)[0],
    kmeans([[0], [1], [9], [10]], 2, seed=3)[0],
)
print(
    "W132: provided example, FINAL reassignment, empty clusters and seed determinism passed"
)
