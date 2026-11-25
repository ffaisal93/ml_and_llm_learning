"""Pure uint8 transformations and one bounded process executor per batch."""

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import os
import numpy as np


def _check(img):
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("image must be a uint8 NumPy array")
    if (
        img.ndim not in (2, 3)
        or min(img.shape[:2]) < 1
        or (img.ndim == 3 and img.shape[2] != 3)
    ):
        raise ValueError("image must be nonempty gray or RGB")


def resize(img, height, width):
    _check(img)
    if height < 1 or width < 1:
        raise ValueError("positive output dimensions required")
    rows = np.arange(height) * img.shape[0] // height
    cols = np.arange(width) * img.shape[1] // width
    return img[rows[:, None], cols[None, :]].copy()


def to_grayscale(img):
    _check(img)
    if img.ndim == 2:
        return img.copy()
    a = img.astype(np.int32)
    return (
        (299 * a[:, :, 0] + 587 * a[:, :, 1] + 114 * a[:, :, 2] + 500) // 1000
    ).astype(np.uint8)


def box_blur(img):
    _check(img)
    h, w = img.shape[:2]
    padding = ((1, 1), (1, 1)) if img.ndim == 2 else ((1, 1), (1, 1), (0, 0))
    padded = np.pad(img.astype(np.int32), padding, mode="edge")
    total = np.zeros(img.shape, dtype=np.int32)
    for dy in range(3):
        for dx in range(3):
            total += padded[dy : dy + h, dx : dx + w]
    return ((total + 4) // 9).astype(np.uint8)


def process_one(image, transformations):
    result = image.copy()
    for transform in transformations:
        result = transform(result)
    return result


class BatchImageProcessor:
    def __init__(self, transformations, num_workers=None, executor_factory=None):
        self.num_workers = (os.cpu_count() or 1) if num_workers is None else num_workers
        if self.num_workers < 1:
            raise ValueError("num_workers must be positive")
        self.transformations = tuple(transformations)
        self.executor_factory = executor_factory or ProcessPoolExecutor

    def process(self, images):
        if len(images) == 0:
            return [], {}
        results = [None] * len(images)
        errors = {}
        # Bound submitted work: do not serialize 100,000 huge images at once.
        next_index = 0
        pending = {}
        with self.executor_factory(self.num_workers) as executor:
            while next_index < len(images) or pending:
                while next_index < len(images) and len(pending) < 2 * self.num_workers:
                    i = next_index
                    next_index += 1
                    try:
                        future = executor.submit(
                            process_one, images[i], self.transformations
                        )
                        pending[future] = i
                    except Exception as exc:
                        errors[i] = f"{type(exc).__name__}: {exc}"
                if not pending:
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i = pending.pop(future)
                    try:
                        results[i] = future.result()
                    except Exception as exc:
                        errors[i] = f"{type(exc).__name__}: {exc}"
        return results, errors
