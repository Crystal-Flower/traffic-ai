# Traffic AI: computer-vision traffic monitoring and adaptive signal timing

Problem statement: a city wants a CCTV-based system that detects cars, buses, motorcycles and trucks at a busy junction,
estimates traffic density, decides whether traffic is **normal or heavy**, and **recommends or dynamically adjusts signal timing**.

## What it does

```
CCTV video (one per approach)
   -> vehicle detection (car, motorcycle, bus, truck)
   -> keep vehicles inside each approach's zone polygon
   -> PCE-weighted occupancy, smoothed            (density)
   -> NORMAL / HEAVY with hysteresis + dwell time (+ stopped-queue rule from a small tracker)
   -> signal planner: green split per phase within safe limits (recommendation or dynamic plan)
   -> annotated video, CSV metrics, plan history, dashboard
```

| Requirement | Where |
|---|---|
| Continuous video capture (file, RTSP/HTTP, webcam) | `traffic_ai/pipeline.py` (OpenCV `VideoCapture`) |
| Detect cars, buses, motorcycles, trucks | `traffic_ai/detectors.py` (YOLOv4-tiny via OpenCV, or YOLOv8 via `ultralytics`) |
| Estimate density | `traffic_ai/density.py` (passenger-car-equivalent occupancy per zone) |
| Normal vs heavy | `traffic_ai/density.py` (hysteresis, dwell time, stopped-queue rule) |
| Recommend / adjust signal timing | `traffic_ai/signal.py` (bounded green split, cycle and rate limits) |
| Show it improves flow | `traffic_ai/sim.py` (fixed-time vs adaptive simulation) |
| Dashboard | `app.py` (Streamlit) |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m traffic_ai fetch-models                       # ~24 MB YOLOv4-tiny files, no PyTorch needed
# put a traffic video in demo/ (see demo/README.md), then:
python -m traffic_ai run --source demo/highway.avi --steps 40 --out outputs/run
python -m traffic_ai simulate --seeds 20                # fixed vs adaptive, writes outputs/sim/
streamlit run app.py                                    # dashboard
pytest -q                                               # 19 tests
```

Outputs of `run`: `annotated.mp4` (four approach tiles + signal banner), `metrics.csv`, `plans.json`, `summary.json`.

### Use your own junction (one camera per approach)

```bash
python -m traffic_ai make-config --source demo/highway.avi --out configs/mine.yaml   # starter file
# edit: each approach gets its own `source` and a `zone` polygon (normalized 0..1)
python -m traffic_ai preview-zones --config configs/mine.yaml --out outputs/zones.jpg
python -m traffic_ai run --config configs/mine.yaml
```

`configs/junction_example.yaml` is a fully commented example. **Calibrate `capacity_pce`** per camera: the number of
car-equivalents that fill the monitored zone bumper to bumper. Occupancy = PCE in zone / capacity.

### More accurate detector (needs PyTorch)

```bash
pip install ultralytics
python -m traffic_ai run --source demo/highway.avi --detector ultralytics
```

## How decisions are made

* **Density:** each vehicle inside a zone counts by its class weight (defaults: car 1, motorcycle 0.5, bus 3, truck 3; these are
  **assumptions**, set them to your city's standard). The sum is divided by the zone's capacity and smoothed with an EMA.
* **HEAVY:** occupancy above `heavy_enter` (0.60) for at least `min_dwell_s`; it clears only below `heavy_exit` (0.40) for the dwell time,
  so it does not flap. A second rule flags HEAVY at moderate occupancy when most tracked vehicles are stopped (a queue).
* **Signal plan:** a phase's demand is that of its busiest approach. Green time is split in proportion to demand, then bounded:
  min/max green, cycle between `min_cycle_s` and `max_cycle_s`, yellow and all-red fixed, at most `max_step_s` change per plan, and a
  deadband so sensor noise in balanced traffic does not cause oscillation. The cycle lengthens under load and shortens when light.
* Every recommendation carries plain-language reasons ("extend NS green 35->45s; demand 1.50").

## Results and what they do and do not show

**Simulation (`python -m traffic_ai simulate --seeds 20`, one simulated hour per run, average delay per vehicle, mean ± std over 20 seeds):**

| Scenario | Fixed-time | Adaptive | Change |
|---|---|---|---|
| balanced demand | 20.8 ± 0.6 s | 19.4 ± 0.7 s | -7.0% |
| moderate N-S skew | 23.3 ± 1.2 s | 21.1 ± 0.6 s | -9.4% |
| north surge (10 min) | 48.0 ± 10.3 s | 30.0 ± 5.0 s | -37.6% |
| shifting peak (N-S then E-W) | 84.6 ± 15.6 s | 32.4 ± 3.7 s | -61.7% |
| N-S peak (fixed plan overloaded) | 129.9 ± 28.3 s | 22.8 ± 0.8 s | -82.4% |

Read this correctly: the large gains occur where the fixed plan is *overloaded on one axis* (an arrangement I chose, so the fixed
plan's queue grows unchecked). The realistic, modest cases are the first two rows. Assumptions: 1800 veh/h saturation flow per approach,
2 s start-up lost time, 3 s yellow + 2 s all-red, monitored capacity 25 vehicles, Poisson arrivals. The adaptive planner sees queue-based
occupancy, which is cleaner than a real camera estimate. **This is a simulation, not evidence of real-world improvement.** During
development the first version of the planner was *worse* than fixed-time in balanced traffic (+11%); adding the deadband fixed that.

**Real-video test (development):** the pipeline ran end to end on a public 320x240 highway clip (four approaches reusing one clip at
different offsets), about 2 processed frames/s per approach on one CPU core. `conf` was lowered from 0.30 to 0.25 after a measurement
showed about 35% more detections at no speed cost. Detection recall on that low-resolution clip is limited: the small model misses
some vehicles and occasionally labels a car as a truck (which inflates PCE). The HEAVY demonstration on real detections used a
deliberately small zone capacity (a "stress" setting); it demonstrates the mechanism, not a measured congestion level.

## Known limitations

* Capacity and thresholds need per-camera calibration; the defaults are placeholders.
* The `ultralytics` (YOLOv8) backend is implemented and its adapter is unit-tested with a fake model, but **it was not run with real
  YOLOv8 weights in the development environment** (PyTorch could not be installed there). Run it once on your machine before relying on it.
* Speed is a proxy (bounding-box heights per second), not km/h. Night, rain, glare and camera shake are not evaluated.
* One approach = one camera. Demo mode reuses a single video for all four approaches and is labeled as such in the UI and video.
* The planner outputs a plan. Applying it to a real controller needs the traffic authority's approval and a certified interface.
  Pedestrian phases, emergency-vehicle preemption and corridor coordination are not modelled.

## Project layout

```
traffic_ai/  config.py detectors.py zones.py tracker.py density.py signal.py pipeline.py sim.py __main__.py
app.py       Streamlit dashboard
configs/     junction_example.yaml
tests/       test_traffic_ai.py (19 tests)
outputs_sample/  sample results from development runs (simulation plot, annotated frames, stress-demo video)
```
