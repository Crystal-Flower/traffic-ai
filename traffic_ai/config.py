"""Configuration objects. Everything tunable lives here, loaded from YAML."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")

# Passenger-car-equivalent weights. These are ASSUMED defaults: tune them to the
# standard your city uses and to the vehicle mix at the junction.
DEFAULT_PCE = {"car": 1.0, "motorcycle": 0.5, "bus": 3.0, "truck": 3.0}


@dataclass
class ApproachCfg:
    name: str
    source: str  # video file, RTSP/HTTP URL, or webcam index as a string
    zone: list[list[float]]  # polygon of normalized (x, y) points in 0..1
    capacity_pce: float = 12.0  # PCE units that fill the monitored zone (calibrate!)
    offset_frames: int = 0  # start offset; only used to reuse one video for a demo


@dataclass
class DensityCfg:
    pce: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PCE))
    ema_alpha: float = 0.25  # smoothing of the occupancy signal per processed frame
    heavy_enter: float = 0.60  # occupancy that flags HEAVY (assumption, tune)
    heavy_exit: float = 0.40  # occupancy below which HEAVY clears (hysteresis)
    min_dwell_s: float = 3.0  # condition must hold this long before state flips
    # Stopped-queue rule: high-ish occupancy AND most vehicles not moving.
    stopped_speed: float = 0.15  # bbox-heights per second below which a vehicle is "stopped"
    stopped_fraction_heavy: float = 0.6
    stopped_occupancy_min: float = 0.35
    min_tracks_for_speed: int = 3


@dataclass
class SignalCfg:
    phases: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "NS", "approaches": ["N", "S"]},
            {"name": "EW", "approaches": ["E", "W"]},
        ]
    )
    yellow_s: float = 3.0
    all_red_s: float = 2.0
    min_green_s: float = 10.0
    max_green_s: float = 60.0
    min_cycle_s: float = 60.0
    max_cycle_s: float = 120.0
    fixed_green_s: float = 35.0  # the baseline fixed-time plan, per phase
    max_step_s: float = 10.0  # max change of any green between consecutive plans
    plan_interval_s: float = 10.0  # how often the live pipeline recomputes a plan
    deadband: float = 0.12  # phase demands closer than this to the mean count as equal (stops jitter)


@dataclass
class DetectorCfg:
    kind: str = "dnn"  # "dnn" (OpenCV + YOLOv4-tiny, no PyTorch) or "ultralytics"
    conf: float = 0.25
    model: str = "yolov8n.pt"  # ultralytics only
    imgsz: int = 640  # ultralytics only
    models_dir: str = "models"  # dnn only


@dataclass
class Config:
    approaches: list[ApproachCfg]
    density: DensityCfg = field(default_factory=DensityCfg)
    signal: SignalCfg = field(default_factory=SignalCfg)
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    stride: int = 2  # process every Nth frame
    fps_fallback: float = 25.0


def _build(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    approaches = [_build(ApproachCfg, a) for a in raw.get("approaches", [])]
    if not approaches:
        raise ValueError("Config needs at least one approach")
    cfg = Config(
        approaches=approaches,
        density=_build(DensityCfg, raw.get("density")),
        signal=_build(SignalCfg, raw.get("signal")),
        detector=_build(DetectorCfg, raw.get("detector")),
        stride=int(raw.get("stride", 2)),
        fps_fallback=float(raw.get("fps_fallback", 25.0)),
    )
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    names = [a.name for a in cfg.approaches]
    if len(set(names)) != len(names):
        raise ValueError("Approach names must be unique")
    for a in cfg.approaches:
        if len(a.zone) < 3:
            raise ValueError(f"Zone for {a.name} needs at least 3 points")
        for x, y in a.zone:
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError(f"Zone points for {a.name} must be normalized to 0..1")
        if a.capacity_pce <= 0:
            raise ValueError(f"capacity_pce for {a.name} must be positive")
    s = cfg.signal
    for ph in s.phases:
        for a in ph["approaches"]:
            if a not in names:
                raise ValueError(f"Phase {ph['name']} references unknown approach {a}")
    if not (0 < s.min_green_s <= s.max_green_s):
        raise ValueError("Need 0 < min_green_s <= max_green_s")
    lost = len(s.phases) * (s.yellow_s + s.all_red_s)
    if s.min_cycle_s > s.max_cycle_s:
        raise ValueError("min_cycle_s must be <= max_cycle_s")
    if len(s.phases) * s.max_green_s + lost < s.min_cycle_s:
        raise ValueError("max_green_s too small to reach min_cycle_s")
    if cfg.density.heavy_exit >= cfg.density.heavy_enter:
        raise ValueError("heavy_exit must be below heavy_enter (hysteresis)")
    if cfg.stride < 1:
        raise ValueError("stride must be >= 1")


DEFAULT_ZONE = [[0.0, 0.30], [1.0, 0.30], [1.0, 1.0], [0.0, 1.0]]


def demo_config(video: str) -> Config:
    """Four approaches that all reuse ONE video with different start offsets.

    DEMO ONLY: real deployments use one camera per approach. This lets a single
    clip drive the whole pipeline, and the UI labels it as such.
    """
    approaches = [
        ApproachCfg(name=n, source=video, zone=[list(p) for p in DEFAULT_ZONE], offset_frames=off)
        for n, off in (("N", 0), ("E", 90), ("S", 180), ("W", 270))
    ]
    cfg = Config(approaches=approaches)
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    def conv(o):
        if is_dataclass(o):
            return {f.name: conv(getattr(o, f.name)) for f in fields(o)}
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, list):
            return [conv(v) for v in o]
        return o

    return conv(cfg)
