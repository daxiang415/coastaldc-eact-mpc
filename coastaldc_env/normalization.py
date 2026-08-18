"""Observation normalization helpers."""

from __future__ import annotations

import numpy as np


class RunningScaler:
    """Fixed min-max scaling to roughly [0, 1] given known physical bounds."""

    def __init__(self, lows: np.ndarray, highs: np.ndarray):
        self.lows = np.asarray(lows, dtype=np.float64)
        self.highs = np.asarray(highs, dtype=np.float64)
        span = self.highs - self.lows
        span[span <= 0] = 1.0
        self.span = span

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return ((x - self.lows) / self.span).astype(np.float32)


def cyclical(value: float, period: float) -> tuple[float, float]:
    """Encode a periodic feature as (sin, cos)."""
    ang = 2.0 * np.pi * value / period
    return float(np.sin(ang)), float(np.cos(ang))
