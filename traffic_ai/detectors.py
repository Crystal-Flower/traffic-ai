"""Vehicle detectors. Two interchangeable backends behind one call signature.

* ``DnnDetector``: OpenCV DNN + YOLOv4-tiny. No PyTorch needed; runs on a laptop CPU.
* ``UltralyticsDetector``: YOLOv8/YOLO11 through the ``ultralytics`` package.
  Higher accuracy; use this on a machine where you can install PyTorch.

Both return only car / motorcycle / bus / truck detections.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .config import VEHICLE_CLASSES, DetectorCfg


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def anchor(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the vehicle touches the road."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2.0, y2)

    @property
    def height(self) -> float:
        return max(1.0, self.box[3] - self.box[1])


Detector = Callable[[np.ndarray], list[Detection]]

DARKNET_FILES = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
    "coco.names": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names",
}


def fetch_models(models_dir: str | Path = "models") -> Path:
    """Download the YOLOv4-tiny files used by ``DnnDetector`` (about 24 MB)."""
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, url in DARKNET_FILES.items():
        target = d / name
        if target.exists() and target.stat().st_size > 0:
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, target)  # noqa: S310 (fixed https URLs above)
    return d


class DnnDetector:
    def __init__(self, models_dir: str | Path = "models", conf: float = 0.30, nms: float = 0.4, size: int = 416):
        d = Path(models_dir)
        weights, cfg, names = d / "yolov4-tiny.weights", d / "yolov4-tiny.cfg", d / "coco.names"
        for p in (weights, cfg, names):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run: python -m traffic_ai fetch-models")
        self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = [ln.strip() for ln in names.read_text().splitlines() if ln.strip()]
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.conf, self.nms, self.size = conf, nms, size

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        boxes: list[list[int]] = []
        confs: list[float] = []
        classes: list[str] = []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                name = self.names[cid]
                if name not in VEHICLE_CLASSES:
                    continue
                conf = float(det[4] * scores[cid])
                if conf < self.conf:
                    continue
                cx, cy, bw, bh = det[0] * w, det[1] * h, det[2] * w, det[3] * h
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confs.append(conf)
                classes.append(name)
        if not boxes:
            return []
        keep = np.array(cv2.dnn.NMSBoxes(boxes, confs, self.conf, self.nms)).flatten()
        result = []
        for i in keep:
            x, y, bw, bh = boxes[int(i)]
            result.append(Detection(classes[int(i)], confs[int(i)], (float(x), float(y), float(x + bw), float(y + bh))))
        return result


class UltralyticsDetector:
    def __init__(self, model: str = "yolov8n.pt", conf: float = 0.30, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install ultralytics to use the 'ultralytics' detector") from e
        self.model = YOLO(model)
        self.conf, self.imgsz = conf, imgsz

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        names = res.names
        out: list[Detection] = []
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        for b, c, s in zip(xyxy, cls, conf):
            name = names[int(c)]
            if name in VEHICLE_CLASSES:
                out.append(Detection(name, float(s), tuple(float(v) for v in b)))
        return out


def make_detector(cfg: DetectorCfg) -> Detector:
    if cfg.kind == "dnn":
        return DnnDetector(cfg.models_dir, conf=cfg.conf)
    if cfg.kind == "ultralytics":
        return UltralyticsDetector(cfg.model, conf=cfg.conf, imgsz=cfg.imgsz)
    raise ValueError(f"Unknown detector kind {cfg.kind!r} (use 'dnn' or 'ultralytics')")


class ScriptedDetector:
    """Test double: returns whatever ``fn(frame_index)`` says."""

    def __init__(self, fn: Callable[[int], Sequence[Detection]]):
        self.fn = fn
        self.i = 0

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        out = list(self.fn(self.i))
        self.i += 1
        return out
