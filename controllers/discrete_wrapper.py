"""SustainDC-style discrete-action ablation.

Wraps any continuous controller and quantizes each action dimension to 3 levels
({-1, 0, +1} for workload/setpoint, {0, 0.5, 1} for pump), reproducing the coarse
MultiDiscrete(3,3,3) interface of SustainDC-style environments (27 joint actions).
Used for the continuous-vs-discrete ablation.
"""

from __future__ import annotations

import numpy as np

LEVELS = np.array([
    [-1.0, 0.0, 1.0],   # a_workload
    [-1.0, 0.0, 1.0],   # a_setpoint
    [0.0, 0.5, 1.0],    # a_pump
])


def quantize(action: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.float32)
    for i in range(3):
        out[i] = LEVELS[i][np.argmin(np.abs(LEVELS[i] - action[i]))]
    return out


class DiscretizedController:
    def __init__(self, base_controller):
        self.base = base_controller
        self.name = f"discrete_{base_controller.name}"

    def reset(self):
        if hasattr(self.base, "reset"):
            self.base.reset()

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        return quantize(np.asarray(self.base.act(obs, info)))
