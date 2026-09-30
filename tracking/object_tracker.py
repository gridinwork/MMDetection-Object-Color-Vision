"""ByteTrack-style two-stage tracker.

MMDetection supplies detections. This tracker keeps an ID while an object
moves, and it can recover that ID for a few frames after the object disappears.
"""

from __future__ import annotations

import numpy as np

from tracking.track_state import TrackState, update_motion
from utils.logger import get_logger

log = get_logger("tracker")


def iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def greedy_match(tracks, detections, iou_threshold: float):
    pairs = []
    for track_index, track in enumerate(tracks):
        predicted = track.predicted_bbox if getattr(track, "predicted_bbox", None) is not None else track.bbox
        for det_index, det in enumerate(detections):
            if track.class_name != det.class_name:
                continue
            score = max(iou_xyxy(predicted, det.bbox), iou_xyxy(track.bbox, det.bbox))
            if score >= iou_threshold:
                pairs.append((score, track_index, det_index))
    pairs.sort(key=lambda item: item[0], reverse=True)
    used_tracks = set()
    used_dets = set()
    matches = []
    for _score, track_index, det_index in pairs:
        if track_index in used_tracks or det_index in used_dets:
            continue
        used_tracks.add(track_index)
        used_dets.add(det_index)
        matches.append((track_index, det_index))
    unmatched_tracks = [index for index in range(len(tracks)) if index not in used_tracks]
    unmatched_dets = [index for index in range(len(detections)) if index not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


class ObjectTracker:
    def __init__(self, lost_frames_timeout: int = 20, trail_length: int = 30, iou_threshold: float = 0.3):
        self.lost_frames_timeout = max(1, int(lost_frames_timeout))
        self.trail_length = max(2, int(trail_length))
        self.iou_threshold = float(iou_threshold)
        self.tracks: list[TrackState] = []
        self._next_id = 1

    def configure(self, lost_frames_timeout: int, trail_length: int) -> None:
        self.lost_frames_timeout = max(1, int(lost_frames_timeout))
        self.trail_length = max(2, int(trail_length))
        for track in self.tracks:
            track.trail.set_maxlen(self.trail_length)

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 1

    def update(self, detections, high_thr: float, low_thr: float = 0.10):
        try:
            return self._update(detections, high_thr, low_thr)
        except Exception:
            log.exception("Tracking failed; resetting tracks")
            self.reset()
            return []

    def _update(self, detections, high_thr: float, low_thr: float):
        high_thr = float(high_thr)
        low_thr = min(float(low_thr), high_thr)
        for track in self.tracks:
            track.age += 1
            track.predicted_bbox = track.kalman.predict()

        high = [det for det in detections if float(det.score) >= high_thr]
        low = [det for det in detections if low_thr <= float(det.score) < high_thr]

        matches, unmatched_track_ids, unmatched_high = greedy_match(
            self.tracks, high, self.iou_threshold
        )
        for track_index, det_index in matches:
            self._apply(self.tracks[track_index], high[det_index])

        remain = [self.tracks[index] for index in unmatched_track_ids]
        matches_low, still_open, _unmatched_low = greedy_match(remain, low, self.iou_threshold)
        for track_index, det_index in matches_low:
            self._apply(remain[track_index], low[det_index])
        for track_index in still_open:
            remain[track_index].time_since_update += 1

        for det_index in unmatched_high:
            self._birth(high[det_index])

        self.tracks = [
            track for track in self.tracks if track.time_since_update <= self.lost_frames_timeout
        ]
        visible = []
        for track in self.tracks:
            if track.time_since_update == 0 and track.score >= high_thr:
                visible.append(track)
        return visible

    def _birth(self, det) -> None:
        track = TrackState(self._next_id, det, self.trail_length)
        track.predicted_bbox = track.bbox.copy()
        self._next_id += 1
        self.tracks.append(track)

    def _apply(self, track: TrackState, det) -> None:
        track.time_since_update = 0
        track.hits += 1
        track.score = float(det.score)
        track.bbox = np.asarray(det.bbox, dtype=np.float32).copy()
        track.mask = det.mask
        track.class_name = det.class_name
        track.label = int(det.label)
        track.kalman.update(track.bbox)
        cx = float((track.bbox[0] + track.bbox[2]) * 0.5)
        cy = float((track.bbox[1] + track.bbox[3]) * 0.5)
        track.trail.add(cx, cy)
        update_motion(track)
