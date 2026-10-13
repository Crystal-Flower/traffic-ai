"""Traffic density estimation and NORMAL / HEAVY classification per approach."""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import VEHICLE_CLASSES, DensityCfg
from .detectors import Detection


@dataclass
class ApproachState:
    name: str
    t: float
    counts: dict[str, int]
    pce: float  # passenger-car-equivalent load in the zone right now
    occupancy_raw: float  # pce / capacity
    occupancy: float  # smoothed
    level: str  # "NORMAL" or "HEAVY"
    stopped_fraction: float | None  # None when too few tracked vehicles to say
    reason: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class DensityEstimator:
    def __init__(self, name: str, capacity_pce: float, cfg: DensityCfg):
        self.name, self.capacity, self.cfg = name, capacity_pce, cfg
        self.level = "NORMAL"
        self._ema: float | None = None
        self._pending_since: float | None = None

    def _stopped_queue(self, occ: float, stopped_fraction: float | None, n_speed: int) -> bool:
        c = self.cfg
        return (
            stopped_fraction is not None
            and n_speed >= c.min_tracks_for_speed
            and occ >= c.stopped_occupancy_min
            and stopped_fraction >= c.stopped_fraction_heavy
        )

    def update(
        self,
        dets_in_zone: list[Detection],
        t: float,
        stopped_fraction: float | None = None,
        n_speed: int = 0,
    ) -> ApproachState:
        c = self.cfg
        counts = {k: 0 for k in VEHICLE_CLASSES}
        pce = 0.0
        for d in dets_in_zone:
            counts[d.cls] = counts.get(d.cls, 0) + 1
            pce += c.pce.get(d.cls, 1.0)
        raw = pce / self.capacity
        self._ema = raw if self._ema is None else c.ema_alpha * raw + (1 - c.ema_alpha) * self._ema
        occ = self._ema
        stopped_q = self._stopped_queue(occ, stopped_fraction, n_speed)

        want_heavy = occ >= c.heavy_enter or stopped_q
        want_normal = occ <= c.heavy_exit and not stopped_q
        reason = ""
        if self.level == "NORMAL":
            if want_heavy:
                if self._pending_since is None:
                    self._pending_since = t
                if t - self._pending_since >= c.min_dwell_s:
                    self.level, self._pending_since = "HEAVY", None
                    reason = "stopped queue" if stopped_q and occ < c.heavy_enter else "high occupancy"
            else:
                self._pending_since = None
        else:
            if want_normal:
                if self._pending_since is None:
                    self._pending_since = t
                if t - self._pending_since >= c.min_dwell_s:
                    self.level, self._pending_since = "NORMAL", None
                    reason = "cleared"
            else:
                self._pending_since = None
        return ApproachState(self.name, t, counts, pce, raw, occ, self.level, stopped_fraction, reason)
