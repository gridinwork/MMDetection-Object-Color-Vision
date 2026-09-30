"""Small K-means palette for object pixels."""

from __future__ import annotations

import numpy as np


def kmeans_colors(pixels: np.ndarray, k: int = 3, iterations: int = 8) -> list[dict]:
    """Cluster BGR pixels. Returns centers sorted by weight."""
    if pixels is None or len(pixels) == 0:
        return []
    data = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
    count = int(data.shape[0])
    clusters = int(max(1, min(k, count)))
    if count == 1 or clusters == 1:
        center = data.mean(axis=0)
        return [{"bgr": center, "ratio": 1.0}]

    rng = np.random.default_rng(0)
    centers = np.empty((clusters, 3), dtype=np.float32)
    centers[0] = data[int(rng.integers(0, count))]
    closest = np.full(count, np.inf, dtype=np.float32)
    for index in range(1, clusters):
        distance = ((data - centers[index - 1]) ** 2).sum(axis=1)
        closest = np.minimum(closest, distance)
        total = float(closest.sum())
        if total <= 1e-6:
            centers[index] = data[int(rng.integers(0, count))]
            continue
        probs = closest / total
        centers[index] = data[int(rng.choice(count, p=probs))]

    labels = np.zeros(count, dtype=np.int32)
    for _ in range(iterations):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(clusters):
            selected = data[labels == index]
            if len(selected):
                centers[index] = selected.mean(axis=0)

    weights = np.bincount(labels, minlength=clusters).astype(np.float32)
    total = float(weights.sum()) or 1.0
    order = np.argsort(-weights)
    palette = []
    for index in order:
        if weights[index] <= 0:
            continue
        palette.append({"bgr": centers[index], "ratio": float(weights[index] / total)})
    return palette
