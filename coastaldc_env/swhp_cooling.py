"""SWHP cooling with a lumped IT-equipment inlet-air temperature proxy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CoolingConfig:
    # setpoint bounds (supply/chilled-water proxy, °C)
    t_set_min: float = 18.0
    t_set_max: float = 27.0
    t_set_init: float = 22.0
    dt_set_max: float = 0.5           # max setpoint change per hour, °C

    # Representative IT-equipment inlet-air thermal model (ASHRAE A1).
    t_inlet_init: float = 24.0
    t_inlet_recommended_min_c: float = 18.0
    t_inlet_recommended_max_c: float = 27.0
    t_inlet_allowable_min_c: float = 15.0
    t_inlet_allowable_max_c: float = 32.0
    alpha: float = 0.30               # °C per MWh of IT heat (thermal mass ~ 1/alpha)
    beta: float = 0.30                # °C per MWh of removed heat
    ambient_leak: float = 0.02        # passive coupling toward 22 °C

    # cooling delivery: Q_cool = conductance * (T_room - T_set) * effectiveness(m_dot)
    conductance_mw_per_k: float = 2.5

    # COP model
    cop_ref: float = 3.5              # at T_set=22, T_sea=15
    cop_dtset: float = 0.12           # dCOP / dT_set  (higher setpoint -> higher COP)
    cop_dtsea: float = -0.10          # dCOP / dT_sea  (warmer sea -> lower COP)
    cop_min: float = 1.5
    cop_max: float = 7.0
    hx_loss_k: float = 1.5            # heat-exchanger approach loss, reduces effective lift benefit

    # seawater pump
    pump_rated_mw: float = 0.4        # rated pump electric power at m_dot_rated
    m_dot_min: float = 0.2            # normalized min flow
    m_dot_rated: float = 1.0
    eff_min: float = 0.55             # HX effectiveness at minimum flow

    # auxiliary (fans etc.), fraction of IT power
    aux_fraction: float = 0.03

    # Optional two-step equipment-allowable safety shield.
    enforce_thermal_safety: bool = True

    @property
    def t_room_init(self) -> float:
        """Legacy read alias for pre-redesign callers."""
        return self.t_inlet_init

    @property
    def t_room_soft_max(self) -> float:
        """Legacy read alias; the canonical name is the recommended maximum."""
        return self.t_inlet_recommended_max_c

    @property
    def t_room_max(self) -> float:
        """Legacy read alias; the canonical name is the allowable maximum."""
        return self.t_inlet_allowable_max_c

    @property
    def thermal_safety_margin_k(self) -> float:
        """Legacy read alias; the unsupported fixed margin has been removed."""
        return 0.0


class SWHPCoolingModel:
    """Continuous cooling subsystem with setpoint and pump-flow control."""

    def __init__(self, config: CoolingConfig | None = None):
        self.cfg = config or CoolingConfig()
        self.reset()

    def reset(self):
        self.t_set = self.cfg.t_set_init
        self.t_inlet = self.cfg.t_inlet_init

    @property
    def t_room(self) -> float:
        """Legacy state alias for historical controllers and artifacts."""
        return self.t_inlet

    @t_room.setter
    def t_room(self, value: float) -> None:
        self.t_inlet = float(value)

    def cop(self, t_sea: float) -> float:
        c = self.cfg
        t_sea_eff = t_sea + c.hx_loss_k
        cop = c.cop_ref + c.cop_dtset * (self.t_set - 22.0) + c.cop_dtsea * (t_sea_eff - 15.0)
        return float(np.clip(cop, c.cop_min, c.cop_max))

    def _flow_and_effectiveness(self, a_pump: float) -> tuple[float, float, float]:
        c = self.cfg
        a_p = float(np.clip(a_pump, 0.0, 1.0))
        m_dot = c.m_dot_min + a_p * (c.m_dot_rated - c.m_dot_min)
        p_pump = c.pump_rated_mw * (m_dot / c.m_dot_rated) ** 3
        effectiveness = c.eff_min + (1.0 - c.eff_min) * (
            m_dot - c.m_dot_min) / (c.m_dot_rated - c.m_dot_min)
        return m_dot, p_pump, effectiveness

    def action_to_setpoint(self, a_setpoint: float) -> float:
        c = self.cfg
        a = float(np.clip(a_setpoint, -1.0, 1.0))
        return c.t_set_min + 0.5 * (a + 1.0) * (c.t_set_max - c.t_set_min)

    def setpoint_to_action(self, t_set: float) -> float:
        c = self.cfg
        return float(np.clip(
            2.0 * (t_set - c.t_set_min) / (c.t_set_max - c.t_set_min) - 1.0,
            -1.0, 1.0))

    def _predict_from_state(self, t_inlet: float, t_set: float, a_pump: float,
                            q_it_mwh: float) -> tuple[float, float]:
        c = self.cfg
        _, _, effectiveness = self._flow_and_effectiveness(a_pump)
        q_cool = max(
            0.0, c.conductance_mw_per_k * (t_inlet - t_set)
        ) * effectiveness
        q_cool = min(q_cool, 2.0 * q_it_mwh + 5.0)
        t_next = (t_inlet
                  + c.alpha * q_it_mwh
                  - c.beta * q_cool
                  - c.ambient_leak * (t_inlet - 22.0))
        return float(t_next), float(q_cool)

    def _predict_temperature(self, t_set: float, a_pump: float,
                             q_it_mwh: float) -> tuple[float, float]:
        return self._predict_from_state(self.t_inlet, t_set, a_pump, q_it_mwh)

    def _controls_are_viable(self, t_set: float, a_pump: float,
                             q_it_mwh: float, q_it_next_max: float,
                             limit: float) -> bool:
        c = self.cfg
        t_next, _ = self._predict_temperature(t_set, a_pump, q_it_mwh)
        if t_next > limit + 1e-8:
            return False
        next_t_set = max(c.t_set_min, t_set - c.dt_set_max)
        t_after_worst, _ = self._predict_from_state(
            t_next, next_t_set, 1.0, q_it_next_max)
        return t_after_worst <= limit + 1e-8

    def _rate_limited_controls(
        self, a_setpoint: float, a_pump: float
    ) -> tuple[float, float, float]:
        c = self.cfg
        requested_sp = float(np.clip(a_setpoint, -1.0, 1.0))
        requested_pump = float(np.clip(a_pump, 0.0, 1.0))
        desired_t_set = self.action_to_setpoint(requested_sp)
        lower_t_set = max(c.t_set_min, self.t_set - c.dt_set_max)
        upper_t_set = min(c.t_set_max, self.t_set + c.dt_set_max)
        requested_t_set = float(np.clip(
            desired_t_set, lower_t_set, upper_t_set))
        return (
            self.setpoint_to_action(requested_t_set),
            requested_pump,
            requested_t_set,
        )

    def _project_safe_controls(self, a_setpoint: float, a_pump: float,
                               q_it_mwh: float,
                               q_it_next_max: float) -> tuple[float, float, float, bool, bool]:
        """Project controls into the current and next-step robust viability set."""
        c = self.cfg
        requested_sp, requested_pump, requested_t_set = (
            self._rate_limited_controls(a_setpoint, a_pump))
        lower_t_set = max(c.t_set_min, self.t_set - c.dt_set_max)
        limit = c.t_inlet_allowable_max_c

        requested_viable = self._controls_are_viable(
            requested_t_set, requested_pump, q_it_mwh, q_it_next_max, limit)
        if requested_viable:
            return (requested_sp, requested_pump, requested_t_set,
                    False, requested_viable)

        if not self._controls_are_viable(
                lower_t_set, 1.0, q_it_mwh, q_it_next_max, limit):
            return (self.setpoint_to_action(lower_t_set), 1.0,
                    lower_t_set, True, False)

        # Find the warmest rate-limited setpoint that is viable at full flow.
        low_t, high_t = lower_t_set, requested_t_set
        for _ in range(12):
            mid_t = 0.5 * (low_t + high_t)
            if self._controls_are_viable(
                    mid_t, 1.0, q_it_mwh, q_it_next_max, limit):
                low_t = mid_t
            else:
                high_t = mid_t
        target_t_set = low_t

        if self._controls_are_viable(
                target_t_set, requested_pump, q_it_mwh, q_it_next_max, limit):
            applied_pump = requested_pump
        else:
            low_p, high_p = requested_pump, 1.0
            for _ in range(12):
                mid_p = 0.5 * (low_p + high_p)
                if self._controls_are_viable(
                        target_t_set, mid_p, q_it_mwh, q_it_next_max, limit):
                    high_p = mid_p
                else:
                    low_p = mid_p
            applied_pump = high_p

        applied_sp = self.setpoint_to_action(target_t_set)
        return applied_sp, float(applied_pump), target_t_set, True, True

    def step(self, a_setpoint: float, a_pump: float, q_it_mwh: float,
             t_sea: float, q_it_next_max: float | None = None):
        """Advance one hour.

        Returns dict with cooling electricity, pump electricity, aux electricity,
        removed heat, inlet temperature, and ASHRAE range exceedance.
        """
        c = self.cfg

        # 1. apply actuator limits, with optional allowable-boundary projection
        if q_it_next_max is None:
            q_it_next_max = q_it_mwh
        if c.enforce_thermal_safety:
            a_sp, a_p, target_t_set, safety_intervened, safety_feasible = (
                self._project_safe_controls(
                    a_setpoint, a_pump, q_it_mwh, q_it_next_max))
        else:
            a_sp, a_p, target_t_set = self._rate_limited_controls(
                a_setpoint, a_pump)
            safety_intervened = False
            safety_feasible = self._controls_are_viable(
                target_t_set, a_p, q_it_mwh, q_it_next_max,
                c.t_inlet_allowable_max_c)
        self.t_set = target_t_set

        # 2. pump flow
        m_dot, p_pump, effectiveness = self._flow_and_effectiveness(a_p)

        # 3. delivered cooling
        q_cool = max(
            0.0, c.conductance_mw_per_k * (self.t_inlet - self.t_set)
        ) * effectiveness
        q_cool = min(q_cool, 2.0 * q_it_mwh + 5.0)  # physical cap

        # 4. representative inlet-air temperature evolution (first order)
        t_next = (self.t_inlet
                  + c.alpha * q_it_mwh
                  - c.beta * q_cool
                  - c.ambient_leak * (self.t_inlet - 22.0))
        self.t_inlet = float(t_next)

        # 5. electricity
        cop = self.cop(t_sea)
        e_cool = q_cool / cop
        e_aux = c.aux_fraction * q_it_mwh

        allowable_exceedance = max(
            0.0, self.t_inlet - c.t_inlet_allowable_max_c)
        recommended_exceedance = max(
            0.0, self.t_inlet - c.t_inlet_recommended_max_c)

        return {
            "e_cooling_mwh": e_cool,
            "e_pump_mwh": p_pump,
            "e_aux_mwh": e_aux,
            "q_cool_mwh": q_cool,
            "cop": cop,
            "t_set": self.t_set,
            "t_inlet": self.t_inlet,
            "recommended_exceedance_c": recommended_exceedance,
            "allowable_exceedance_c": allowable_exceedance,
            # Legacy aliases remain for old callers; new outputs use inlet names.
            "t_room": self.t_inlet,
            "m_dot": m_dot,
            "thermal_violation_k": allowable_exceedance,
            "thermal_risk_k": recommended_exceedance,
            "applied_a_setpoint": a_sp,
            "applied_a_pump": a_p,
            "safety_intervened": safety_intervened,
            "safety_feasible": safety_feasible,
        }
