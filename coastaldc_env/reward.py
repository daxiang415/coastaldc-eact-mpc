"""Multi-objective reward for energy, emissions, service, and safe control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RewardWeights:
    w_grid: float = 1.0
    w_co2: float = 2.0
    w_total: float = 0.2
    w_sla: float = 0.0
    w_thermal: float = 0.0
    w_thermal_risk: float = 0.0
    w_unused_wind: float = 0.0
    w_smooth: float = 0.5
    w_pump: float = 0.0
    w_safety_override: float = 0.0


@dataclass
class RewardNormalizers:
    """Per-step scales used to bring each term to O(1)."""
    grid_scale: float = 10.0          # MWh residual grid purchase / step
    co2_scale: float = 5000.0         # kg / step
    total_scale: float = 10.0         # total facility electricity / step
    sla_scale: float = 2.0            # MWh violated / step
    thermal_scale: float = 2.0        # K over limit
    thermal_risk_scale: float = 1.0   # K over soft safety boundary
    unused_wind_scale: float = 10.0   # MWh / step
    pump_scale: float = 0.4           # MWh / step
    safety_override_scale: float = 1.0


class RewardFunction:
    def __init__(self, weights: RewardWeights | None = None,
                 normalizers: RewardNormalizers | None = None):
        self.w = weights or RewardWeights()
        self.n = normalizers or RewardNormalizers()

    def sla_cost(self, violation_mwh: float) -> float:
        return self.w.w_sla * violation_mwh / self.n.sla_scale

    def __call__(self, e_grid_mwh: float, co2_kg: float, sla_violation_mwh: float,
                 thermal_violation_k: float, unused_wind_mwh: float,
                 action: np.ndarray, prev_action: np.ndarray,
                 e_total_mwh: float, pump_energy_mwh: float,
                 thermal_risk_k: float = 0.0,
                 safety_override: float = 0.0) -> tuple[float, dict]:
        w, n = self.w, self.n
        smooth = float(np.mean((np.asarray(action) - np.asarray(prev_action)) ** 2))
        terms = {
            "grid": w.w_grid * e_grid_mwh / n.grid_scale,
            "co2": w.w_co2 * co2_kg / n.co2_scale,
            "total": w.w_total * e_total_mwh / n.total_scale,
            "sla": self.sla_cost(sla_violation_mwh),
            "thermal": w.w_thermal * thermal_violation_k / n.thermal_scale,
            "thermal_risk": w.w_thermal_risk * thermal_risk_k / n.thermal_risk_scale,
            "unused_wind": w.w_unused_wind * unused_wind_mwh / n.unused_wind_scale,
            "smooth": w.w_smooth * smooth,
            "pump": w.w_pump * pump_energy_mwh / n.pump_scale,
            "safety_override": (
                w.w_safety_override * safety_override / n.safety_override_scale),
        }
        reward = -float(sum(terms.values()))
        return reward, terms
