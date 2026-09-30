"""Per-track state, Kalman box filter, and motion label."""

from __future__ import annotations

import numpy as np

from tracking.trajectory import Trajectory


def box_to_z(bbox) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, width, height], dtype=np.float64)


def z_to_box(state) -> np.ndarray:
    cx, cy, width, height = [float(v) for v in state[:4]]
    width = max(1.0, width)
    height = max(1.0, height)
    return np.array(
        [cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5],
        dtype=np.float32,
    )


class KalmanBox:
    """Constant-velocity filter for center, width, and height."""

    def __init__(self, bbox):
        measurement = box_to_z(bbox)
        self.x = np.zeros(8, dtype=np.float64)
        self.x[:4] = measurement
        self.p = np.eye(8, dtype=np.float64) * 10.0
        self.p[4:, 4:] *= 100.0
        self.f = np.eye(8, dtype=np.float64)
        for index in range(4):
            self.f[index, index + 4] = 1.0
        self.q = np.eye(8, dtype=np.float64)
        self.q[:4, :4] *= 1.0
        self.q[4:, 4:] *= 0.01
        self.r = np.eye(4, dtype=np.float64)
        self.h = np.eye(4, 8, dtype=np.float64)

    def predict(self) -> np.ndarray:
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return z_to_box(self.x)

    def update(self, bbox) -> np.ndarray:
        measurement = box_to_z(bbox)
        innovation = measurement - (self.h @ self.x)
        innovation_cov = self.h @ self.p @ self.h.T + self.r
        gain = self.p @ self.h.T @ np.linalg.inv(innovation_cov)
        self.x = self.x + gain @ innovation
        self.p = (np.eye(8) - gain @ self.h) @ self.p
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        return z_to_box(self.x)


def update_motion(track: "TrackState") -> None:
    points = list(track.trail.points)
    if len(points) < 2:
        track.velocity_x = 0.0
        track.velocity_y = 0.0
        track.motion = "STATIC"
        return
    steps = min(5, len(points) - 1)
    dx = (points[-1][0] - points[-1 - steps][0]) / steps
    dy = (points[-1][1] - points[-1 - steps][1]) / steps
    track.velocity_x = float(dx)
    track.velocity_y = float(dy)
    if (dx * dx + dy * dy) ** 0.5 < 1.25:
        track.motion = "STATIC"
    elif abs(dx) >= abs(dy):
        track.motion = "MOVING RIGHT" if dx > 0 else "MOVING LEFT"
    else:
        track.motion = "MOVING DOWN" if dy > 0 else "MOVING UP"


class TrackState:
    def __init__(self, track_id: int, det, trail_length: int):
        self.track_id = int(track_id)
        self.class_name = det.class_name
        self.label = int(det.label)
        self.bbox = np.asarray(det.bbox, dtype=np.float32).copy()
        self.score = float(det.score)
        self.mask = det.mask
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.kalman = KalmanBox(self.bbox)
        self.trail = Trajectory(trail_length)
        cx = float((self.bbox[0] + self.bbox[2]) * 0.5)
        cy = float((self.bbox[1] + self.bbox[3]) * 0.5)
        self.trail.add(cx, cy)
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.motion = "STATIC"
        self.color_name = ""
        self.rgb = None
        self.hsv = None
        self.lab = None
        self.palette = []
        self.palette_age = 999
