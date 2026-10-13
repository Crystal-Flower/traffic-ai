"""Junction simulator: fixed-time plan vs the adaptive planner.

A deliberately simple queue model (one-second steps, Poisson arrivals, saturation
discharge during green). It exists to show, reproducibly and with stated
assumptions, how the planner behaves under different demand patterns. It is NOT
evidence about real-world congestion reduction: that needs field data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import SignalCfg
from .density import ApproachState
from .signal import SignalPlan, adaptive_plan, demand_from_state, fixed_plan

APPROACHES = ("N", "S", "E", "W")

# Assumptions (stated in the README): ~1800 veh/h per approach lane group at saturation,
# 2 s start-up lost time per green, monitored-zone capacity of 25 vehicles.
SAT_FLOW = 0.5  # vehicles per second
STARTUP_LOST_S = 2
ZONE_CAPACITY = 25.0
EMA = 0.5

Rate = Callable[[str, float], float]  # (approach, t) -> vehicles per second


def _sc_balanced(a: str, t: float) -> float:
    return 0.12


def _sc_ns_peak(a: str, t: float) -> float:
    return 0.22 if a in ("N", "S") else 0.08


def _sc_shifting(a: str, t: float) -> float:
    ns_first = t < 1800
    heavy = (a in ("N", "S")) == ns_first
    return 0.22 if heavy else 0.08


def _sc_surge(a: str, t: float) -> float:
    if a == "N" and 1200 <= t < 1800:
        return 0.32
    return 0.12


def _sc_moderate_skew(a: str, t: float) -> float:
    # Not overloaded: the fixed plan copes, so this shows a modest, realistic gain.
    return 0.16 if a in ("N", "S") else 0.10


SCENARIOS: dict[str, Rate] = {
    "balanced": _sc_balanced,
    "moderate_skew": _sc_moderate_skew,
    "ns_peak": _sc_ns_peak,
    "shifting_peak": _sc_shifting,
    "north_surge": _sc_surge,
}


@dataclass
class SimResult:
    scenario: str
    controller: str
    seed: int
    avg_delay_s: float
    max_queue: int
    served: int
    arrivals: int
    queue_series: list[float]


def simulate(scenario: str, controller: str, seed: int, cfg: SignalCfg | None = None, duration_s: int = 3600) -> SimResult:
    cfg = cfg or SignalCfg()
    rate = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    q = {a: 0 for a in APPROACHES}
    acc = {a: 0.0 for a in APPROACHES}
    ema = {a: 0.0 for a in APPROACHES}
    plan: SignalPlan = fixed_plan(cfg)
    phase_apps = {p["name"]: p["approaches"] for p in cfg.phases}
    t = 0
    queue_time = 0.0
    arrivals = served = max_q = 0
    series: list[float] = []

    def tick(serving: list[str], discharging: bool) -> None:
        nonlocal t, queue_time, arrivals, served, max_q
        for a in APPROACHES:
            k = int(rng.poisson(rate(a, t)))
            q[a] += k
            arrivals += k
        if discharging:
            for a in serving:
                acc[a] += SAT_FLOW
                n = min(int(acc[a]), q[a])
                q[a] -= n
                served += n
                acc[a] -= int(acc[a])
                if q[a] == 0:
                    acc[a] = 0.0
        queue_time += sum(q.values())
        max_q = max(max_q, max(q.values()))
        if t % 10 == 0:
            series.append(float(sum(q.values())))
        t += 1

    first = True
    while t < duration_s:
        if controller == "adaptive" and not first:
            demand = {}
            for a in APPROACHES:
                ema[a] = EMA * (q[a] / ZONE_CAPACITY) + (1 - EMA) * ema[a]
                heavy = ema[a] >= 0.6
                st = ApproachState(a, t, {}, 0.0, 0.0, ema[a], "HEAVY" if heavy else "NORMAL", None)
                demand[a] = demand_from_state(st)
            plan = adaptive_plan(cfg, demand, plan)
        first = False
        for ph in plan.phases:
            serving = phase_apps[ph.name]
            for s in range(ph.green):
                if t >= duration_s:
                    break
                tick(serving, discharging=s >= STARTUP_LOST_S)
            for _ in range(int(ph.yellow + ph.all_red)):
                if t >= duration_s:
                    break
                tick(serving, discharging=False)
    return SimResult(scenario, controller, seed, queue_time / max(1, arrivals), max_q, served, arrivals, series)


def compare(seeds: int = 20, cfg: SignalCfg | None = None, duration_s: int = 3600) -> dict:
    """Mean and std over seeds for each scenario and controller."""
    out: dict = {}
    for sc in SCENARIOS:
        row: dict = {}
        for ctl in ("fixed", "adaptive"):
            rs = [simulate(sc, ctl, s, cfg, duration_s) for s in range(seeds)]
            d = np.array([r.avg_delay_s for r in rs])
            row[ctl] = {
                "avg_delay_s_mean": float(d.mean()),
                "avg_delay_s_std": float(d.std()),
                "max_queue_mean": float(np.mean([r.max_queue for r in rs])),
                "served_mean": float(np.mean([r.served for r in rs])),
                "arrivals_mean": float(np.mean([r.arrivals for r in rs])),
                "queue_series_seed0": rs[0].queue_series,
            }
        f, a = row["fixed"]["avg_delay_s_mean"], row["adaptive"]["avg_delay_s_mean"]
        row["delay_change_pct"] = 100.0 * (a - f) / f if f else 0.0
        out[sc] = row
    return out


def plot_comparison(results: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for i, (ctl, color) in enumerate((("fixed", "#8a8a8a"), ("adaptive", "#2c9c5a"))):
        m = [results[n][ctl]["avg_delay_s_mean"] for n in names]
        s = [results[n][ctl]["avg_delay_s_std"] for n in names]
        ax[0].bar(x + (i - 0.5) * 0.38, m, 0.38, yerr=s, label=ctl, color=color, capsize=3)
    ax[0].set_xticks(x, names, rotation=15)
    ax[0].set_ylabel("average delay per vehicle (s)")
    ax[0].set_title("Fixed-time vs adaptive (simulation, mean ± std over seeds)")
    ax[0].legend()
    sc = "shifting_peak"
    for ctl, color in (("fixed", "#8a8a8a"), ("adaptive", "#2c9c5a")):
        ser = results[sc][ctl]["queue_series_seed0"]
        ax[1].plot(np.arange(len(ser)) * 10 / 60, ser, label=ctl, color=color)
    ax[1].set_xlabel("time (min)")
    ax[1].set_ylabel("vehicles queued at the junction")
    ax[1].set_title(f"Queue over time, scenario: {sc} (seed 0)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
