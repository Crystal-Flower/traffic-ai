"""A small greedy IoU tracker, used only to estimate whether vehicles are moving.

It is not a replacement for ByteTrack/DeepSORT. Its job is to give each vehicle
a short-lived identity so we can tell a queue of stopped vehicles from free flow.
Speed is measured in bounding-box heights per second, a crude but camera-agnostic
proxy (a vehicle that moves less than a fraction of its own height per second is
"stopped").
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .detectors import Detection


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple[float, float, float, float]
    hist: deque = field(default_factory=lambda: deque(maxlen=40))  # (t, cx, cy, h)
    missed: int = 0
    seen_now: bool = True


class SimpleTracker:
    def __init__(self, iou_min: float = 0.10, max_missed: int = 4, gate_heights: float = 1.2):
        self.iou_min, self.max_missed, self.gate = iou_min, max_missed, gate_heights
        self.tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, dets: list[Detection], t: float) -> list[int]:
        """Match detections to tracks; returns a track id per detection (same order)."""
        for tr in self.tracks.values():
            tr.seen_now = False
        pairs = []
        for di, d in enumerate(dets):
            for tid, tr in self.tracks.items():
                iou = _iou(d.box, tr.box)
                if iou >= self.iou_min:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)
        det_to_track: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, di, tid in pairs:
            if di in det_to_track or tid in used_tracks:
                continue
            det_to_track[di] = tid
            used_tracks.add(tid)
        # centroid fallback for leftovers (fast movers whose boxes no longer overlap)
        for di, d in enumerate(dets):
            if di in det_to_track:
                continue
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            best, best_dist = None, self.gate * d.height
            for tid, tr in self.tracks.items():
                if tid in used_tracks:
                    continue
                tcx, tcy = (tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2
                dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = tid, dist
            if best is not None:
                det_to_track[di] = best
                used_tracks.add(best)
        ids: list[int] = []
        for di, d in enumerate(dets):
            tid = det_to_track.get(di)
            if tid is None:
                tid = self._next
                self._next += 1
                self.tracks[tid] = Track(id=tid, box=d.box)
            tr = self.tracks[tid]
            tr.box, tr.missed, tr.seen_now = d.box, 0, True
            tr.hist.append((t, (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2, d.height))
            ids.append(tid)
        for tid in list(self.tracks):
            tr = self.tracks[tid]
            if not tr.seen_now:
                tr.missed += 1
                if tr.missed > self.max_missed:
                    del self.tracks[tid]
        return ids

    def speeds(self, min_span_s: float = 0.8) -> dict[int, float]:
        """Speed in box-heights per second for tracks seen now with enough history."""
        out: dict[int, float] = {}
        for tid, tr in self.tracks.items():
            if not tr.seen_now or len(tr.hist) < 2:
                continue
            t1, x1, y1, h1 = tr.hist[-1]
            # oldest sample that is still within ~3 s of the newest
            base = next((s for s in tr.hist if t1 - s[0] <= 3.0), tr.hist[0])
            t0, x0, y0, h0 = base
            span = t1 - t0
            if span < min_span_s:
                continue
            dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            out[tid] = dist / (0.5 * (h0 + h1)) / span
        return out
