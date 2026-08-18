"""No-control baseline: no workload shifting, fixed setpoint, fixed pump flow."""

from __future__ import annotations

import numpy as np


class NoControlController:
    name = "no_control"

    def __init__(self, pump_level: float = 0.6, setpoint_c: float = 22.0):
        self.pump_level = pump_level
        self.setpoint_action = 2.0 * (setpoint_c - 18.0) / 9.0 - 1.0

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        return np.array(
            [0.0, self.setpoint_action, self.pump_level], dtype=np.float32)
