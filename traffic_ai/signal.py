"""Signal-timing planner.

Turns per-approach demand into a safety-bounded plan: green split per phase, with
fixed yellow and all-red intervals. The planner never emits a green below the
minimum or above the maximum, never changes a phase by more than ``max_step_s``
per plan, and keeps the cycle inside [min_cycle_s, max_cycle_s].

This is a recommendation engine. Driving a real signal controller needs the
authority's approval and a certified controller interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import SignalCfg
from .density import ApproachState


@dataclass
class PhasePlan:
    name: str
    green: int
    yellow: float
    all_red: float


@dataclass
class SignalPlan:
    cycle: float
    phases: list[PhasePlan]
    mode: str  # "fixed" or "adaptive"
    demand: dict[str, float] = field(default_factory=dict)  # per phase
    reasons: list[str] = field(default_factory=list)

    def green_of(self, name: str) -> int:
        return next(p.green for p in self.phases if p.name == name)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "cycle_s": self.cycle,
            "phases": [{"name": p.name, "green_s": p.green, "yellow_s": p.yellow, "all_red_s": p.all_red} for p in self.phases],
            "demand": {k: round(v, 3) for k, v in self.demand.items()},
            "reasons": self.reasons,
        }


def lost_time(cfg: SignalCfg) -> float:
    return len(cfg.phases) * (cfg.yellow_s + cfg.all_red_s)


def fixed_plan(cfg: SignalCfg) -> SignalPlan:
    g = int(round(cfg.fixed_green_s))
    phases = [PhasePlan(p["name"], g, cfg.yellow_s, cfg.all_red_s) for p in cfg.phases]
    return SignalPlan(cycle=sum(p.green + p.yellow + p.all_red for p in phases), phases=phases, mode="fixed")


def demand_from_state(state: ApproachState) -> float:
    """Demand in roughly 0..1.5: smoothed occupancy, plus a bonus when flagged HEAVY."""
    return min(1.5, state.occupancy + (0.15 if state.level == "HEAVY" else 0.0))


def phase_demands(cfg: SignalCfg, demand_by_approach: dict[str, float]) -> dict[str, float]:
    """A phase's demand is that of its most demanding approach (the critical approach)."""
    out = {}
    for ph in cfg.phases:
        vals = [demand_by_approach[a] for a in ph["approaches"] if a in demand_by_approach]
        out[ph["name"]] = max(vals) if vals else 0.0
    return out


def _fit(desired: list[float], lo: list[float], hi: list[float], total: float) -> list[float]:
    """Shift desired values by a common offset, clamped to [lo, hi], so they sum to total."""
    total = min(max(total, sum(lo)), sum(hi))
    a, b = -1000.0, 1000.0
    for _ in range(80):
        mid = (a + b) / 2
        s = sum(min(max(d + mid, l), h) for d, l, h in zip(desired, lo, hi))
        if s < total:
            a = mid
        else:
            b = mid
    shift = (a + b) / 2
    return [min(max(d + shift, l), h) for d, l, h in zip(desired, lo, hi)]


def _round_keep_sum(vals: list[float], lo: list[float], hi: list[float]) -> list[int]:
    target = int(round(sum(vals)))
    base = [int(math.floor(v)) for v in vals]
    order = sorted(range(len(vals)), key=lambda i: vals[i] - base[i], reverse=True)
    for i in order[: max(0, target - sum(base))]:
        base[i] += 1
    return [min(max(b, math.ceil(l)), math.floor(h)) for b, l, h in zip(base, lo, hi)]


def adaptive_plan(cfg: SignalCfg, demand_by_approach: dict[str, float], prev: SignalPlan | None = None) -> SignalPlan:
    n = len(cfg.phases)
    lost = lost_time(cfg)
    pdem = phase_demands(cfg, demand_by_approach)
    dvals = [pdem[p["name"]] for p in cfg.phases]

    # Cycle grows with overall load: shorter when light (less waiting), longer when heavy (less lost time).
    load = min(1.0, sum(dvals) / n)
    cycle_target = cfg.min_cycle_s + (cfg.max_cycle_s - cfg.min_cycle_s) * load
    green_target = cycle_target - lost

    lo = [cfg.min_green_s] * n
    hi = [cfg.max_green_s] * n
    if prev is not None:
        for i, ph in enumerate(cfg.phases):
            pg = prev.green_of(ph["name"])
            lo[i] = max(cfg.min_green_s, pg - cfg.max_step_s)
            hi[i] = min(cfg.max_green_s, pg + cfg.max_step_s)
    # keep the cycle inside its bounds when the per-phase limits allow it
    tmin, tmax = cfg.min_cycle_s - lost, cfg.max_cycle_s - lost
    green_target = min(max(green_target, max(tmin, sum(lo))), min(tmax, sum(hi))) if max(tmin, sum(lo)) <= min(tmax, sum(hi)) else min(max(green_target, sum(lo)), sum(hi))

    # Deadband: demands within `deadband` of the mean are treated as equal, so measurement
    # noise in balanced traffic does not make the split oscillate.
    mean_d = sum(dvals) / n
    dv = [mean_d if abs(d - mean_d) < cfg.deadband else d for d in dvals]
    w = [d + 0.05 for d in dv]
    desired = [green_target * wi / sum(w) for wi in w]
    fitted = _fit(desired, lo, hi, green_target)
    greens = _round_keep_sum(fitted, lo, hi)

    base = fixed_plan(cfg)
    phases = [PhasePlan(p["name"], g, cfg.yellow_s, cfg.all_red_s) for p, g in zip(cfg.phases, greens)]
    plan = SignalPlan(cycle=sum(p.green + p.yellow + p.all_red for p in phases), phases=phases, mode="adaptive", demand=pdem)
    for p in phases:
        delta = p.green - base.green_of(p.name)
        if delta:
            verb = "extend" if delta > 0 else "shorten"
            plan.reasons.append(f"{verb} {p.name} green {base.green_of(p.name)}->{p.green}s ({delta:+d}); demand {pdem[p.name]:.2f}")
    if not plan.reasons:
        plan.reasons.append("demand balanced: keep the baseline split")
    return plan


def recommend_from_states(cfg: SignalCfg, states: dict[str, ApproachState], prev: SignalPlan | None) -> SignalPlan:
    demand = {a: demand_from_state(s) for a, s in states.items()}
    plan = adaptive_plan(cfg, demand, prev)
    heavy = sorted(a for a, s in states.items() if s.level == "HEAVY")
    if heavy:
        plan.reasons.insert(0, "HEAVY traffic on " + ", ".join(heavy))
    return plan
