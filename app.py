"""Streamlit dashboard:  streamlit run app.py"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from traffic_ai.config import demo_config, load_config
from traffic_ai.detectors import make_detector
from traffic_ai.pipeline import Pipeline
from traffic_ai.sim import SCENARIOS, compare, plot_comparison

st.set_page_config(page_title="Traffic AI", layout="wide")
st.title("Computer-vision traffic monitoring and adaptive signal timing")
st.caption("Detects cars, buses, motorcycles and trucks, estimates density per approach, classifies NORMAL vs HEAVY, and recommends signal timing.")

DEMO_DIR = Path("demo")
demo_files = sorted(str(p) for p in DEMO_DIR.glob("*") if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}) if DEMO_DIR.exists() else []

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Demo: one video for all 4 approaches", "Config file (one camera per approach)"])
    video_path, cfg_path = None, None
    if mode.startswith("Demo"):
        up = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov", "mkv"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"traffic_upload_{up.name}"
            tmp.write_bytes(up.getvalue())
            video_path = str(tmp)
        elif demo_files:
            video_path = st.selectbox("...or a bundled clip", demo_files)
        st.caption("Demo mode reuses ONE video for all four approaches at different start offsets. Real use needs one camera per approach.")
    else:
        cfg_path = st.text_input("Config path", "configs/junction_example.yaml")
    st.header("Detector")
    kind = st.selectbox("Backend", ["dnn", "ultralytics"], help="dnn: OpenCV + YOLOv4-tiny, no PyTorch. ultralytics: YOLOv8, needs `pip install ultralytics`.")
    conf = st.slider("Confidence", 0.10, 0.80, 0.25, 0.05)
    stride = st.slider("Process every Nth frame", 1, 10, 3)
    steps = st.slider("Steps to process", 10, 300, 40)
    st.header("Classification")
    heavy_enter = st.slider("HEAVY above occupancy", 0.30, 1.00, 0.60, 0.05)
    heavy_exit = st.slider("Back to NORMAL below", 0.05, 0.90, 0.40, 0.05)
    capacity = st.slider("Zone capacity (PCE units)", 2.0, 40.0, 12.0, 1.0, help="Calibrate per camera: how many car-equivalents fill the zone.")
    st.header("Signal limits")
    min_green = st.slider("Minimum green (s)", 5, 30, 10)
    max_green = st.slider("Maximum green (s)", 30, 90, 60)

tab_live, tab_sim, tab_about = st.tabs(["Live monitoring", "Signal simulation", "About and limits"])

with tab_live:
    run = st.button("Run monitoring", type="primary")
    if run:
        try:
            if cfg_path:
                cfg = load_config(cfg_path)
                note = ""
            else:
                if not video_path:
                    st.error("Choose or upload a video first.")
                    st.stop()
                cfg = demo_config(video_path)
                note = "DEMO MODE: one video reused for all 4 approaches at different start offsets"
            cfg.detector.kind, cfg.detector.conf, cfg.stride = kind, conf, stride
            cfg.density.heavy_enter, cfg.density.heavy_exit = heavy_enter, heavy_exit
            for a in cfg.approaches:
                a.capacity_pce = capacity
            cfg.signal.min_green_s, cfg.signal.max_green_s = float(min_green), float(max_green)
            from traffic_ai.config import validate

            validate(cfg)
            detector = make_detector(cfg.detector)
            pipe = Pipeline(cfg, detector, loop=True, note=note)
        except Exception as e:  # show config / model problems in the UI
            st.error(f"Could not start: {e}")
            st.stop()

        left, right = st.columns([3, 2])
        frame_slot = left.empty()
        prog = st.progress(0.0)
        metric_slot = right.empty()
        plan_slot = right.empty()
        chart_slot = st.empty()
        rows = []
        t0 = time.time()
        for res in pipe.iter_steps(steps):
            frame_slot.image(res.composite[:, :, ::-1], channels="RGB")
            with metric_slot.container():
                cols = st.columns(len(res.states))
                for c, (name, s) in zip(cols, res.states.items()):
                    c.metric(name, s.level, f"occ {s.occupancy:.2f}", delta_color="inverse" if s.level == "HEAVY" else "off")
            plan_df = pd.DataFrame(
                [{"phase": p.name, "adaptive green (s)": p.green, "fixed green (s)": pipe.fixed.green_of(p.name), "change (s)": p.green - pipe.fixed.green_of(p.name)} for p in res.plan.phases]
            )
            plan_slot.dataframe(plan_df, hide_index=True)
            rows.append({"t (s)": round(res.t, 1), **{k: s.occupancy for k, s in res.states.items()}})
            chart_slot.line_chart(pd.DataFrame(rows).set_index("t (s)"), height=220)
            prog.progress((res.step + 1) / steps)
        pipe.close()
        summary = pipe.summary(time.time() - t0, len(rows))
        st.success(f"Processed {summary['steps']} steps ({summary['video_seconds']} s of video) in {summary['wall_seconds']} s.")
        st.subheader("Latest recommendation")
        st.json(summary["final_plan"])
        st.subheader("Per-approach summary")
        st.dataframe(pd.DataFrame(summary["approaches"]).T)
        st.download_button("Download plans (JSON)", json.dumps(pipe.plan_history, indent=2), "plans.json")
        st.download_button("Download timeline (JSON)", json.dumps(pipe.timeline), "timeline.json")
    else:
        st.info("Pick a source in the sidebar and press **Run monitoring**.")

with tab_sim:
    st.write(
        "Fixed-time vs adaptive planner on a simple queue model. **This is a simulation under stated assumptions, not a measurement of real traffic.**"
    )
    seeds = st.slider("Random seeds (more = smoother averages)", 3, 30, 10)
    if st.button("Run simulation"):
        with st.spinner("Simulating 1 hour of traffic per scenario per controller ..."):
            res = compare(seeds=seeds)
        table = pd.DataFrame(
            [
                {
                    "scenario": sc,
                    "fixed delay (s)": round(r["fixed"]["avg_delay_s_mean"], 1),
                    "adaptive delay (s)": round(r["adaptive"]["avg_delay_s_mean"], 1),
                    "change (%)": round(r["delay_change_pct"], 1),
                }
                for sc, r in res.items()
            ]
        )
        st.dataframe(table, hide_index=True)
        out = Path(tempfile.gettempdir()) / "sim.png"
        plot_comparison(res, str(out))
        st.image(str(out))
    st.caption(
        "Assumptions: 1800 vehicles/hour saturation flow per approach (0.5/s), 2 s start-up lost time per green, 3 s yellow + 2 s all-red, "
        "monitored zone capacity 25 vehicles, Poisson arrivals, adaptive planner sees the queue-based occupancy (a real camera estimate is noisier)."
    )

with tab_about:
    st.markdown(
        """
**Pipeline:** video → vehicle detection (car, motorcycle, bus, truck) → keep vehicles inside each approach zone → PCE-weighted occupancy (smoothed) →
NORMAL/HEAVY with hysteresis and dwell time (plus a stopped-queue rule using a small tracker) → signal plan within safe limits.

**Known limits (please read):**
- Occupancy needs a **per-camera capacity calibration**; the default is a placeholder.
- Detection recall depends on camera height, resolution, night and rain. The bundled `dnn` model is small and misses vehicles in low-resolution video; `ultralytics` (YOLOv8) is better.
- Speed is a crude proxy (bounding-box heights per second), not calibrated km/h.
- The planner produces a **recommendation** bounded by min/max green and step limits. Driving a real signal controller needs the traffic authority's approval and a certified interface.
- The simulation compares controllers under assumptions; it is not proof of real-world improvement.
        """
    )
