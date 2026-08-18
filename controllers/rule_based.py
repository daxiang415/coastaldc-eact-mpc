"""Rule-based continuous controller (RBC).

Implements the README rules with continuous (proportional) outputs:
  - high wind & backlog        -> increase flexible execution (recover)
  - low wind & high carbon     -> defer flexible workload within SLA limit
  - cold sea (and cool room)   -> relax pump / slightly raise setpoint
  - room temp near limit       -> lower setpoint, increase pump flow
"""

from __future__ import annotations

import numpy as np

from coastaldc_env.continuous_env import ObsIndex as O


class RuleBasedController:
    name = "rule_based"

    def __init__(self,
                 wind_high: float = 0.5, wind_low: float = 0.25,
                 carbon_high: float = 0.45,      # normalized (kg/MWh / 1000)
                 sst_cold: float = 12.0 / 30.0,
                 t_room_alarm: float = 0.68,     # normalized trigger (~28.9 C, below the 30 C limit)
                 t_room_ok: float = 0.55,        # only relax cooling when the room is comfortably cool
                 base_pump: float = 0.5,
                 terminal_recovery_threshold: float = 24.0 / 168.0):
        self.p = dict(wind_high=wind_high, wind_low=wind_low, carbon_high=carbon_high,
                      sst_cold=sst_cold, t_room_alarm=t_room_alarm, t_room_ok=t_room_ok,
                      base_pump=base_pump,
                      terminal_recovery_threshold=terminal_recovery_threshold)

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        p = self.p
        wind = obs[O.WIND]
        carbon = obs[O.CARBON]
        backlog = obs[O.BACKLOG]
        pressure = obs[O.DEADLINE_PRESSURE]
        remaining = obs[O.REMAINING] if len(obs) > O.REMAINING else 1.0
        sst = obs[O.SST]
        t_room = obs[O.T_ROOM]

        # --- workload ---
        a_workload = 0.0
        if wind > p["wind_high"] and backlog > 0.01:
            a_workload = min(1.0, 2.0 * (wind - p["wind_high"]) + backlog)      # recover
        elif wind < p["wind_low"] and carbon > p["carbon_high"]:
            carbon_excess = max(0.0, carbon - p["carbon_high"])
            wind_deficit = max(0.0, p["wind_low"] - wind)
            a_workload = -min(1.0, 2.0 * carbon_excess + 2.0 * wind_deficit)    # defer
        # deadline pressure overrides deferral
        if pressure > 0.7:
            a_workload = max(a_workload, min(1.0, pressure))
        if remaining <= p["terminal_recovery_threshold"] and backlog > 1e-6:
            a_workload = 1.0

        # --- cooling setpoint & pump ---
        # Absolute target action: convert the observed normalized setpoint back
        # to the action range so the neutral command holds its current target.
        a_setpoint = float(np.clip(2.0 * obs[O.T_SET] - 1.0, -1.0, 1.0))
        a_pump = p["base_pump"]
        if sst < p["sst_cold"] and t_room < p["t_room_ok"]:
            a_setpoint = 1.0 / 3.0   # target 24 C
            a_pump = max(0.3, p["base_pump"] - 0.2)
        if t_room > p["t_room_alarm"]:
            urgency = min(1.0, (t_room - p["t_room_alarm"]) / max(1.0 - p["t_room_alarm"], 1e-6))
            a_setpoint = -5.0 / 9.0  # target 20 C
            a_pump = min(1.0, p["base_pump"] + 0.5 * urgency + 0.2)

        return np.array([a_workload, a_setpoint, a_pump], dtype=np.float32)
