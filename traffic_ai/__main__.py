"""Command line: python -m traffic_ai <command>"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .config import Config, config_to_dict, demo_config, load_config
from .detectors import fetch_models, make_detector


def _cfg_from_args(a) -> tuple[Config, str]:
    note = ""
    if a.config:
        cfg = load_config(a.config)
    else:
        if not a.source:
            sys.exit("Give --config FILE or --source VIDEO")
        cfg = demo_config(a.source)
        note = "DEMO MODE: one video reused for all 4 approaches at different start offsets"
    if a.detector:
        cfg.detector.kind = a.detector
    if a.stride:
        cfg.stride = a.stride
    return cfg, note


def cmd_run(a) -> None:
    from .pipeline import run_pipeline

    cfg, note = _cfg_from_args(a)
    det = make_detector(cfg.detector)
    steps = a.steps

    def prog(n, res):
        if n % 10 == 0 or res.plan_changed:
            greens = " ".join(f"{p.name}={p.green}s" for p in res.plan.phases)
            lv = " ".join(f"{k}:{s.level[0]}{s.occupancy:.2f}" for k, s in res.states.items())
            print(f"[{n:4d}] t={res.t:6.1f}s  {lv}   plan {greens}{'  <- changed' if res.plan_changed else ''}")

    summary = run_pipeline(cfg, det, a.out, max_steps=steps, save_video=not a.no_video, note=note, progress=prog)
    print(json.dumps({k: summary[k] for k in ("steps", "video_seconds", "wall_seconds", "processing_fps_per_approach", "approaches", "final_plan")}, indent=2))
    print(f"\nresults in {a.out}/  (annotated.mp4, metrics.csv, plans.json, summary.json)")


def cmd_simulate(a) -> None:
    from .sim import compare, plot_comparison

    res = compare(seeds=a.seeds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'scenario':15s} {'fixed (s)':>14s} {'adaptive (s)':>16s} {'change':>9s}")
    for sc, row in res.items():
        f, ad = row["fixed"], row["adaptive"]
        print(f"{sc:15s} {f['avg_delay_s_mean']:8.1f}±{f['avg_delay_s_std']:<5.1f} {ad['avg_delay_s_mean']:9.1f}±{ad['avg_delay_s_std']:<5.1f} {row['delay_change_pct']:+8.1f}%")
    slim = {sc: {k: ({kk: vv for kk, vv in v.items() if kk != 'queue_series_seed0'} if isinstance(v, dict) else v) for k, v in row.items()} for sc, row in res.items()}
    (out / "simulation.json").write_text(json.dumps(slim, indent=2))
    plot_comparison(res, str(out / "simulation.png"))
    print(f"\nwrote {out}/simulation.json and simulation.png  (seeds={a.seeds}; SIMULATION, see README for assumptions)")


def cmd_preview_zones(a) -> None:
    from .pipeline import ApproachRuntime, draw_approach
    from .density import ApproachState

    cfg, _ = _cfg_from_args(a)
    tiles = []
    for ap_cfg in cfg.approaches:
        from .pipeline import ApproachRuntime

        ap = ApproachRuntime(ap_cfg, cfg)
        f = ap.read(1, True)
        ap.zone.ensure_size(f)
        ap.frame, ap.dets, ap.inside = f, [], []
        ap.state = ApproachState(ap_cfg.name, 0.0, {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}, 0.0, 0.0, 0.0, "NORMAL", None)
        tiles.append(draw_approach(ap, (480, 270)))
        ap.release()
    import numpy as np

    rows = [np.hstack(tiles[i : i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i : i + 2]))) for i in range(0, len(tiles), 2)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, np.vstack(rows))
    print(f"zone preview saved to {a.out}. Edit the zone polygons in your config (normalized 0..1) and preview again.")


def cmd_config(a) -> None:
    cfg = demo_config(a.source)
    import yaml

    d = config_to_dict(cfg)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(d, sort_keys=False))
    print(f"wrote starter config to {a.out}. Give each approach its own camera source and a zone polygon.")


def main() -> None:
    p = argparse.ArgumentParser(prog="traffic_ai", description="CV traffic monitoring + adaptive signal timing")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--config", help="YAML config (one camera source per approach)")
        sp.add_argument("--source", help="single video for DEMO MODE (reused for all approaches)")
        sp.add_argument("--detector", choices=["dnn", "ultralytics"])
        sp.add_argument("--stride", type=int, help="process every Nth frame")

    sp = sub.add_parser("fetch-models", help="download YOLOv4-tiny files (~24 MB) for the dnn detector")
    sp.add_argument("--dir", default="models")
    sp.set_defaults(fn=lambda a: fetch_models(a.dir))

    sp = sub.add_parser("run", help="process video(s), write annotated video + metrics + plans")
    common(sp)
    sp.add_argument("--out", default="outputs/run")
    sp.add_argument("--steps", type=int, help="stop after N processed steps")
    sp.add_argument("--no-video", action="store_true")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("simulate", help="fixed-time vs adaptive junction simulation")
    sp.add_argument("--seeds", type=int, default=20)
    sp.add_argument("--out", default="outputs/sim")
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser("preview-zones", help="draw the approach zones on one frame per camera")
    common(sp)
    sp.add_argument("--out", default="outputs/zones.jpg")
    sp.set_defaults(fn=cmd_preview_zones)

    sp = sub.add_parser("make-config", help="write a starter YAML config from a video")
    sp.add_argument("--source", required=True)
    sp.add_argument("--out", default="configs/starter.yaml")
    sp.set_defaults(fn=cmd_config)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
