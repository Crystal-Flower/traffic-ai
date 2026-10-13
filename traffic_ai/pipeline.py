"""End-to-end pipeline: video in, density + plan out."""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .config import Config, ApproachCfg, config_to_dict
from .density import ApproachState, DensityEstimator
from .detectors import Detection, Detector
from .signal import SignalPlan, fixed_plan, recommend_from_states
from .tracker import SimpleTracker
from .zones import Zone

GREEN, RED, GREY = (80, 200, 80), (60, 60, 230), (150, 150, 150)
CLASS_COLOR = {"car": (255, 200, 60), "motorcycle": (60, 220, 255), "bus": (255, 120, 220), "truck": (60, 140, 255)}


def _open_source(src: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(src) if src.isdigit() else src)


class ApproachRuntime:
    def __init__(self, cfg: ApproachCfg, config: Config):
        self.cfg = cfg
        self.is_file = not (cfg.source.isdigit() or "://" in cfg.source)
        self.cap = _open_source(cfg.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source for approach {cfg.name}: {cfg.source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.zone = Zone(cfg.zone)
        self.tracker = SimpleTracker()
        self.est = DensityEstimator(cfg.name, cfg.capacity_pce, config.density)
        self.density_cfg = config.density
        self.dets: list[Detection] = []
        self.inside: list[bool] = []
        self.frame: np.ndarray | None = None
        self.state: ApproachState | None = None
        self.ended = False
        self._seek_start()

    def _seek_start(self) -> None:
        if self.is_file and self.cfg.offset_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cfg.offset_frames)

    def read(self, stride: int, loop: bool) -> np.ndarray | None:
        for _ in range(stride - 1):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok and self.is_file and loop:
            self._seek_start() if self.cfg.offset_frames else self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return None
        return frame

    def process(self, frame: np.ndarray, detector: Detector, t: float) -> ApproachState:
        self.frame = frame
        self.zone.ensure_size(frame)
        self.dets = detector(frame)
        self.inside = [self.zone.contains(d.anchor) for d in self.dets]
        in_zone = [d for d, ins in zip(self.dets, self.inside) if ins]
        self.tracker.update(in_zone, t)
        speeds = self.tracker.speeds()
        stopped_fraction = None
        if speeds:
            stopped_fraction = sum(1 for v in speeds.values() if v < self.density_cfg.stopped_speed) / len(speeds)
        self.state = self.est.update(in_zone, t, stopped_fraction, len(speeds))
        return self.state

    def release(self) -> None:
        self.cap.release()


@dataclass
class StepResult:
    step: int
    t: float
    states: dict[str, ApproachState]
    plan: SignalPlan
    composite: np.ndarray
    plan_changed: bool = False


def _put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_approach(ap: ApproachRuntime, tile: tuple[int, int]) -> np.ndarray:
    img = ap.frame.copy()
    st = ap.state
    color = RED if st.level == "HEAVY" else GREEN
    ap.zone.draw(img, color)
    for d, ins in zip(ap.dets, ap.inside):
        x1, y1, x2, y2 = map(int, d.box)
        c = CLASS_COLOR.get(d.cls, (255, 255, 255)) if ins else GREY
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
    img = cv2.resize(img, tile, interpolation=cv2.INTER_AREA)
    cs = st.counts
    _put(img, f"{ap.cfg.name}  {st.level}  occ {st.occupancy:.2f}", (8, 20), 0.6, color, 2)
    _put(img, f"car {cs['car']}  moto {cs['motorcycle']}  bus {cs['bus']}  truck {cs['truck']}  PCE {st.pce:.1f}", (8, tile[1] - 10), 0.45)
    return img


def compose(aps: list[ApproachRuntime], plan: SignalPlan, fixed: SignalPlan, t: float, tile=(480, 270), cols=2, note: str = "") -> np.ndarray:
    tiles = [draw_approach(a, tile) for a in aps]
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    banner = np.zeros((96, grid.shape[1], 3), np.uint8)
    banner[:] = (30, 30, 30)
    greens = "   ".join(f"{p.name} green {p.green}s (fixed {fixed.green_of(p.name)}s)" for p in plan.phases)
    _put(banner, f"t={t:6.1f}s   signal mode: {plan.mode.upper()}   cycle {plan.cycle:.0f}s   {greens}", (8, 24), 0.55)
    _put(banner, "; ".join(plan.reasons)[:150], (8, 52), 0.5, (200, 230, 255))
    if note:
        _put(banner, note, (8, 80), 0.45, (120, 200, 255))
    return np.vstack([grid, banner])


class Pipeline:
    def __init__(self, cfg: Config, detector: Detector, loop: bool = True, note: str = ""):
        self.cfg, self.detector, self.loop, self.note = cfg, detector, loop, note
        self.aps = [ApproachRuntime(a, cfg) for a in cfg.approaches]
        fps = next((a.fps for a in self.aps if a.fps > 1), 0.0) or cfg.fps_fallback
        self.fps = fps
        self.dt = cfg.stride / fps
        self.fixed = fixed_plan(cfg.signal)
        self.plan = self.fixed
        self._last_plan_t = -1e9
        self.plan_history: list[dict] = []
        self.timeline: list[dict] = []

    def iter_steps(self, max_steps: int | None = None) -> Iterator[StepResult]:
        step = 0
        while max_steps is None or step < max_steps:
            t = step * self.dt
            for a in self.aps:
                frame = a.read(self.cfg.stride, self.loop)
                if frame is None:
                    return
                a.process(frame, self.detector, t)
            states = {a.cfg.name: a.state for a in self.aps}
            changed = False
            if t - self._last_plan_t >= self.cfg.signal.plan_interval_s:
                new = recommend_from_states(self.cfg.signal, states, self.plan)
                changed = [p.green for p in new.phases] != [p.green for p in self.plan.phases]
                self.plan, self._last_plan_t = new, t
                self.plan_history.append({"t": round(t, 2), **new.to_dict()})
            self.timeline.append({"t": round(t, 2), **{k: {"occ": round(s.occupancy, 3), "level": s.level, "pce": s.pce, "counts": s.counts} for k, s in states.items()}})
            yield StepResult(step, t, states, self.plan, compose(self.aps, self.plan, self.fixed, t, note=self.note), changed)
            step += 1

    def close(self) -> None:
        for a in self.aps:
            a.release()

    def summary(self, wall_s: float, steps: int) -> dict:
        per = {}
        for a in self.aps:
            n = max(1, len(self.timeline))
            occs = [row[a.cfg.name]["occ"] for row in self.timeline]
            heavy = sum(1 for row in self.timeline if row[a.cfg.name]["level"] == "HEAVY")
            tot = {k: 0 for k in ("car", "motorcycle", "bus", "truck")}
            for row in self.timeline:
                for k, v in row[a.cfg.name]["counts"].items():
                    tot[k] += v
            per[a.cfg.name] = {
                "mean_occupancy": round(float(np.mean(occs)), 3) if occs else 0.0,
                "peak_occupancy": round(float(np.max(occs)), 3) if occs else 0.0,
                "pct_time_heavy": round(100.0 * heavy / n, 1),
                "mean_vehicles_in_zone": {k: round(v / n, 2) for k, v in tot.items()},
            }
        return {
            "steps": steps,
            "video_seconds": round(steps * self.dt, 1),
            "wall_seconds": round(wall_s, 1),
            "processing_fps_per_approach": round(steps / wall_s, 2) if wall_s > 0 else None,
            "approaches": per,
            "final_plan": self.plan.to_dict(),
            "plans_computed": len(self.plan_history),
            "config": config_to_dict(self.cfg),
            "note": self.note,
        }


def run_pipeline(
    cfg: Config,
    detector: Detector,
    out_dir: str | Path,
    max_steps: int | None = None,
    save_video: bool = True,
    loop: bool = True,
    note: str = "",
    progress: Callable[[int, StepResult], None] | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline(cfg, detector, loop=loop, note=note)
    writer = None
    csv_f = open(out / "metrics.csv", "w", newline="")
    cw = csv.writer(csv_f)
    cw.writerow(["t_s", "approach", "car", "motorcycle", "bus", "truck", "pce", "occupancy_raw", "occupancy", "level", "stopped_fraction"])
    t0 = time.time()
    steps = 0
    try:
        for res in pipe.iter_steps(max_steps):
            steps += 1
            for name, s in res.states.items():
                cw.writerow([round(res.t, 2), name, s.counts["car"], s.counts["motorcycle"], s.counts["bus"], s.counts["truck"], round(s.pce, 2), round(s.occupancy_raw, 3), round(s.occupancy, 3), s.level, "" if s.stopped_fraction is None else round(s.stopped_fraction, 2)])
            if save_video:
                if writer is None:
                    h, w = res.composite.shape[:2]
                    writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, pipe.fps / cfg.stride), (w, h))
                writer.write(res.composite)
            if progress:
                progress(steps, res)
    finally:
        pipe.close()
        csv_f.close()
        if writer is not None:
            writer.release()
    summary = pipe.summary(time.time() - t0, steps)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "plans.json").write_text(json.dumps(pipe.plan_history, indent=2))
    return summary
