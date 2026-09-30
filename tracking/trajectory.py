"""Short center-point history for one tracked object."""

from __future__ import annotations

from collections import deque


class Trajectory:
    def __init__(self, maxlen: int = 30):
        self.points: deque[tuple[float, float]] = deque(maxlen=max(2, int(maxlen)))

    def set_maxlen(self, maxlen: int) -> None:
        self.points = deque(self.points, maxlen=max(2, int(maxlen)))

    def add(self, x: float, y: float) -> None:
        self.points.append((float(x), float(y)))

    def as_list(self) -> list[tuple[float, float]]:
        return list(self.points)
