"""Approach zones: the road area whose vehicles count toward an approach."""
from __future__ import annotations

import cv2
import numpy as np


class Zone:
    def __init__(self, poly_norm: list[list[float]]):
        self.poly_norm = poly_norm
        self.size: tuple[int, int] | None = None
        self.pts: np.ndarray | None = None

    def ensure_size(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self.size != (w, h):
            self.size = (w, h)
            self.pts = np.array([[x * w, y * h] for x, y in self.poly_norm], dtype=np.float32)

    def contains(self, point: tuple[float, float]) -> bool:
        assert self.pts is not None, "call ensure_size(frame) first"
        return cv2.pointPolygonTest(self.pts.reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) >= 0

    def draw(self, img: np.ndarray, color: tuple[int, int, int], alpha: float = 0.22) -> None:
        assert self.pts is not None
        poly = self.pts.astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
        cv2.polylines(img, [poly], True, color, 2)
