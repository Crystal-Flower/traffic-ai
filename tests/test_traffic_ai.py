import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_ai.config import Config, ApproachCfg, DensityCfg, SignalCfg, demo_config, load_config, validate
from traffic_ai.density import DensityEstimator
from traffic_ai.detectors import Detection, ScriptedDetector, UltralyticsDetector
from traffic_ai.pipeline import run_pipeline
from traffic_ai.signal import adaptive_plan, fixed_plan, lost_time
from traffic_ai.sim import simulate
from traffic_ai.tracker import SimpleTracker
from traffic_ai.zones import Zone

ROOT = Path(__file__).resolve().parents[1]


def det(cls="car", x=100, y=100, w=40, h=40, conf=0.9):
    return Detection(cls, conf, (x, y, x + w, y + h))


# ---------- config ----------
def test_example_config_loads_and_validates():
    cfg = load_config(ROOT / "configs" / "junction_example.yaml")
    assert [a.name for a in cfg.approaches] == ["N", "E", "S", "W"]


def test_config_rejects_bad_input(tmp_path):
    cfg = demo_config("x.mp4")
    cfg.density.heavy_exit = 0.9
    with pytest.raises(ValueError, match="hysteresis"):
        validate(cfg)
    cfg = demo_config("x.mp4")
    cfg.approaches[0].zone = [[0, 0], [2, 0], [1, 1]]
    with pytest.raises(ValueError, match="normalized"):
        validate(cfg)
    bad = tmp_path / "bad.yaml"
    bad.write_text("approaches:\n  - {name: N, source: a, zone: [[0,0],[1,0],[1,1]], typo_key: 1}\n")
    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(bad)


# ---------- zones ----------
def test_zone_contains():
    z = Zone([[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]])
    z.ensure_size(np.zeros((100, 200, 3), np.uint8))
    assert z.contains((100, 80))
    assert not z.contains((100, 20))


# ---------- density ----------
def test_pce_weighting_and_counts():
    est = DensityEstimator("N", 10.0, DensityCfg())
    s = est.update([det("car"), det("bus"), det("motorcycle")], t=0.0)
    assert s.counts == {"car": 1, "motorcycle": 1, "bus": 1, "truck": 0}
    assert s.pce == pytest.approx(1.0 + 3.0 + 0.5)
    assert s.occupancy_raw == pytest.approx(0.45)


def test_heavy_needs_dwell_and_has_hysteresis():
    cfg = DensityCfg(ema_alpha=1.0, min_dwell_s=3.0, heavy_enter=0.6, heavy_exit=0.4)
    est = DensityEstimator("N", 10.0, cfg)
    dets7 = [det()] * 7  # occupancy 0.7
    assert est.update(dets7, 0.0).level == "NORMAL"  # not yet: dwell
    assert est.update(dets7, 2.0).level == "NORMAL"
    assert est.update(dets7, 3.5).level == "HEAVY"
    # in the hysteresis band (0.5) it must stay HEAVY, and not flap
    for t in (4, 5, 6, 7, 8):
        assert est.update([det()] * 5, float(t)).level == "HEAVY"
    # only a sustained drop below heavy_exit clears it
    assert est.update([det()] * 2, 9.0).level == "HEAVY"
    assert est.update([det()] * 2, 12.5).level == "NORMAL"


def test_brief_spike_does_not_flip_state():
    cfg = DensityCfg(ema_alpha=1.0, min_dwell_s=3.0)
    est = DensityEstimator("N", 10.0, cfg)
    est.update([det()] * 9, 0.0)
    assert est.update([det()] * 1, 1.0).level == "NORMAL"
    assert est.update([det()] * 9, 2.0).level == "NORMAL"


def test_stopped_queue_triggers_heavy_at_moderate_occupancy():
    cfg = DensityCfg(ema_alpha=1.0, min_dwell_s=2.0)
    est = DensityEstimator("N", 10.0, cfg)
    d5 = [det()] * 5  # occupancy 0.5, below heavy_enter
    est.update(d5, 0.0, stopped_fraction=0.9, n_speed=5)
    s = est.update(d5, 2.5, stopped_fraction=0.9, n_speed=5)
    assert s.level == "HEAVY" and "queue" in s.reason
    est2 = DensityEstimator("N", 10.0, cfg)
    est2.update(d5, 0.0, stopped_fraction=0.1, n_speed=5)
    assert est2.update(d5, 2.5, stopped_fraction=0.1, n_speed=5).level == "NORMAL"  # free-flow


# ---------- tracker ----------
def test_tracker_speed_stationary_vs_moving():
    tr = SimpleTracker()
    for i in range(10):
        t = i * 0.4
        dets = [det(x=100, y=100), det(x=300 + 30 * i, y=100 + 10 * i)]
        tr.update(dets, t)
    sp = tr.speeds()
    assert len(sp) == 2
    vals = sorted(sp.values())
    assert vals[0] < 0.05
    assert vals[1] > 0.5


# ---------- signal planner ----------
def test_fixed_plan_is_symmetric():
    p = fixed_plan(SignalCfg())
    assert p.phases[0].green == p.phases[1].green == 35
    assert p.cycle == 80


def test_adaptive_gives_more_green_to_busier_phase_and_respects_bounds():
    cfg = SignalCfg()
    p = adaptive_plan(cfg, {"N": 0.9, "S": 0.8, "E": 0.1, "W": 0.1})
    assert p.green_of("NS") > p.green_of("EW")
    for ph in p.phases:
        assert cfg.min_green_s <= ph.green <= cfg.max_green_s
    assert cfg.min_cycle_s <= p.cycle <= cfg.max_cycle_s


def test_equal_demand_gives_equal_greens():
    p = adaptive_plan(SignalCfg(), {"N": 0.5, "S": 0.5, "E": 0.5, "W": 0.5})
    assert abs(p.green_of("NS") - p.green_of("EW")) <= 1


def test_deadband_ignores_small_noise():
    p = adaptive_plan(SignalCfg(), {"N": 0.52, "S": 0.5, "E": 0.45, "W": 0.5})
    assert abs(p.green_of("NS") - p.green_of("EW")) <= 1


def test_planner_property_bounds_and_rate_limit():
    cfg = SignalCfg()
    rng = random.Random(7)
    prev = None
    for _ in range(300):
        d = {a: rng.uniform(0, 1.5) for a in "NSEW"}
        p = adaptive_plan(cfg, d, prev)
        for ph in p.phases:
            assert cfg.min_green_s <= ph.green <= cfg.max_green_s, (d, p.to_dict())
            if prev is not None:
                assert abs(ph.green - prev.green_of(ph.name)) <= cfg.max_step_s + 1e-9, (d, p.to_dict())
        assert p.cycle == sum(ph.green + ph.yellow + ph.all_red for ph in p.phases)
        prev = p
    # without history the cycle bounds hold as well
    for _ in range(200):
        d = {a: rng.uniform(0, 1.5) for a in "NSEW"}
        p = adaptive_plan(cfg, d)
        assert cfg.min_cycle_s - 1 <= p.cycle <= cfg.max_cycle_s + 1


def test_three_phase_junction_supported():
    cfg = SignalCfg(phases=[{"name": "A", "approaches": ["N"]}, {"name": "B", "approaches": ["E"]}, {"name": "C", "approaches": ["S", "W"]}])
    p = adaptive_plan(cfg, {"N": 0.9, "E": 0.1, "S": 0.1, "W": 0.1})
    assert p.green_of("A") > p.green_of("B")
    assert len(p.phases) == 3


# ---------- simulator ----------
def test_simulation_is_deterministic_and_conservative():
    a = simulate("ns_peak", "adaptive", 3)
    b = simulate("ns_peak", "adaptive", 3)
    assert a.avg_delay_s == b.avg_delay_s
    assert a.served <= a.arrivals


def test_adaptive_beats_fixed_when_demand_is_lopsided():
    f = np.mean([simulate("ns_peak", "fixed", s).avg_delay_s for s in range(5)])
    a = np.mean([simulate("ns_peak", "adaptive", s).avg_delay_s for s in range(5)])
    assert a < 0.5 * f


def test_adaptive_not_worse_when_balanced():
    f = np.mean([simulate("balanced", "fixed", s).avg_delay_s for s in range(8)])
    a = np.mean([simulate("balanced", "adaptive", s).avg_delay_s for s in range(8)])
    assert a <= f * 1.05


# ---------- detector adapter (no PyTorch needed: fake model) ----------
class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self):
        self.xyxy = _T([[0, 0, 10, 10], [5, 5, 30, 30], [1, 1, 9, 9]])
        self.cls = _T([2, 0, 7])
        self.conf = _T([0.9, 0.8, 0.7])

    def __len__(self):
        return 3


class _Res:
    names = {0: "person", 2: "car", 7: "truck"}
    boxes = _Boxes()


class _FakeModel:
    def predict(self, frame, conf, imgsz, verbose):
        return [_Res()]


def test_ultralytics_adapter_filters_to_vehicles():
    d = UltralyticsDetector.__new__(UltralyticsDetector)
    d.model, d.conf, d.imgsz = _FakeModel(), 0.25, 640
    out = d(np.zeros((50, 50, 3), np.uint8))
    assert [x.cls for x in out] == ["car", "truck"]


# ---------- pipeline end to end (scripted detector, synthetic video) ----------
def _make_video(path, n=80, size=(320, 240)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(n):
        w.write(np.full((size[1], size[0], 3), 90, np.uint8))
    w.release()


def test_pipeline_end_to_end_heavy_north_extends_ns_green(tmp_path):
    vid = tmp_path / "v.mp4"
    _make_video(vid)
    cfg = demo_config(str(vid))
    for a in cfg.approaches:
        a.offset_frames = 0
    cfg.stride = 1
    cfg.density.min_dwell_s = 1.0
    cfg.density.ema_alpha = 0.8

    calls = {"i": 0}

    def fn(i):
        # frames alternate approaches N,E,S,W in that order; N gets heavy traffic, others light
        k = i % 4
        if k == 0:
            return [det("car", 20 + 25 * j, 150) for j in range(10)] + [det("bus", 200, 120)]
        return [det("car", 100, 150)] if k == 1 else []

    detector = ScriptedDetector(fn)
    summary = run_pipeline(cfg, detector, tmp_path / "out", max_steps=40, note="test")
    assert (tmp_path / "out" / "metrics.csv").exists()
    assert (tmp_path / "out" / "annotated.mp4").stat().st_size > 0
    n = summary["approaches"]["N"]
    assert n["pct_time_heavy"] > 20
    assert summary["approaches"]["S"]["pct_time_heavy"] == 0
    plans = json.loads((tmp_path / "out" / "plans.json").read_text())
    last = plans[-1]["phases"]
    green = {p["name"]: p["green_s"] for p in last}
    assert green["NS"] > green["EW"]
