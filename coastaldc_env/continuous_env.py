"""CoastalDCContinuousEnv: continuous-control Gymnasium environment for a single
coastal AI data centre with seawater-source heat-pump cooling and offshore-wind matching.

Action (Box, 3-dim):
    a_workload in [-1, 1] : defer (-) / neutral (0) / recover (+) flexible workload
    a_setpoint in [-1, 1] : absolute 18-27 C target, rate-limited by 0.5 C/h
    a_pump     in [ 0, 1] : normalized seawater pump flow

Timestep: 1 hour. Episode: 168 steps (7 days).
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from coastaldc_env.normalization import cyclical
from coastaldc_env.offshore_wind import WindConfig, match_wind
from coastaldc_env.reward import RewardFunction, RewardNormalizers, RewardWeights
from coastaldc_env.swhp_cooling import CoolingConfig, SWHPCoolingModel
from coastaldc_env.synthetic import COUNTRY_PARAMS, generate_country_inputs
from coastaldc_env.workload import WorkloadConfig, WorkloadModel

COUNTRIES = list(COUNTRY_PARAMS.keys())

EPISODE_HOURS = 168
FORECAST_BUCKETS = 4          # next-24h forecasts compressed to 4 x 6-hour means


class ObsIndex:
    """Indices into the observation vector (kept in sync with _observe)."""
    IT_ARRIVAL = 0
    FIXED_LOAD = 1
    FLEX_ARRIVAL = 2
    BACKLOG = 3
    DEADLINE_PRESSURE = 4
    SST = 5
    COP = 6
    WIND = 7
    WIND_FC = slice(8, 8 + FORECAST_BUCKETS)
    CARBON = 8 + FORECAST_BUCKETS                     # 12
    CARBON_FC = slice(13, 13 + FORECAST_BUCKETS)
    PRICE = 13 + FORECAST_BUCKETS                     # 17
    T_SET = 18
    T_INLET = 19
    T_ROOM = T_INLET  # legacy observation-index alias
    PREV_ACTION = slice(20, 23)
    TIME = slice(23, 28)
    REMAINING = 28

REQUIRED_COLUMNS = ["fixed_load_mw", "flexible_arrival_mw", "sst_c", "wind_mw",
                    "carbon_kg_per_mwh", "price_usd_per_mwh"]


class CoastalDCContinuousEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 country: str = "JPN",
                 data_dir: str | None = None,
                 data: pd.DataFrame | None = None,
                 episode_hours: int = EPISODE_HOURS,
                 workload_config: WorkloadConfig | None = None,
                 cooling_config: CoolingConfig | None = None,
                 wind_config: WindConfig | None = None,
                 reward_weights: RewardWeights | None = None,
                 reward_normalizers: RewardNormalizers | None = None,
                 random_episode_start: bool = True,
                 use_oracle_workload_projection: bool = True,
                 use_oracle_forecast_observations: bool = True,
                 seed: int | None = None):
        super().__init__()
        if country not in COUNTRIES:
            raise ValueError(f"country must be one of {COUNTRIES}, got '{country}'")
        self.country = country
        self.episode_hours = episode_hours
        self.random_episode_start = random_episode_start
        self.use_oracle_workload_projection = bool(use_oracle_workload_projection)
        self.use_oracle_forecast_observations = bool(
            use_oracle_forecast_observations)

        self.wl_cfg = workload_config or WorkloadConfig()
        self.cool_cfg = cooling_config or CoolingConfig()
        self.wind_cfg = wind_config or WindConfig()

        self.workload = WorkloadModel(self.wl_cfg)
        self.cooling = SWHPCoolingModel(self.cool_cfg)
        self.reward_fn = RewardFunction(reward_weights, reward_normalizers)

        self.data = self._load_data(data, data_dir)
        if len(self.data) < episode_hours + 24:
            raise ValueError("Input data too short for one episode plus 24h forecast horizon")

        # --- spaces ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._obs_dim = 12 + 2 * FORECAST_BUCKETS + 3 + 5 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._t0 = 0
        self._step_idx = 0
        self._prev_action = np.zeros(3, dtype=np.float32)

    # ------------------------------------------------------------------ data
    def _load_data(self, data: pd.DataFrame | None, data_dir: str | None) -> pd.DataFrame:
        if data is not None:
            return self._prepare_data(data, "Provided data")

        if data_dir is None:
            here = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(here), "data", "processed")
        path = os.path.join(data_dir, f"hourly_inputs_{self.country}.csv")
        if os.path.exists(path):
            return self._prepare_data(pd.read_csv(path), path)

        # fallback: deterministic synthetic placeholder (same generator as build script)
        return self._prepare_data(generate_country_inputs(
            self.country,
            it_capacity_mw=self.wl_cfg.it_capacity_mw,
            wind_capacity_mw=self.wind_cfg.wind_capacity_mw,
        ), "synthetic inputs")

    @staticmethod
    def _prepare_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(f"{label} missing columns: {missing}")
        prepared = data.copy().reset_index(drop=True)
        if "timestamp" in prepared.columns:
            timestamps = pd.to_datetime(prepared["timestamp"], utc=True, errors="raise")
            prepared["timestamp"] = timestamps.dt.tz_convert(None)
            if prepared["timestamp"].duplicated().any():
                raise ValueError(f"{label} contains duplicate timestamps")
            if not prepared["timestamp"].is_monotonic_increasing:
                raise ValueError(f"{label} timestamps must be increasing")
        return prepared

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        max_start = len(self.data) - self.episode_hours - 24
        if options and "start_hour" in options:
            self._t0 = int(options["start_hour"])
            if not (0 <= self._t0 <= max_start):
                raise ValueError(f"start_hour must be in [0, {max_start}]")
        elif self.random_episode_start:
            self._t0 = int(self._rng.integers(0, max_start + 1))
        else:
            self._t0 = 0

        self._step_idx = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self.workload.reset()
        self.cooling.reset()
        self._episode_metrics = _new_metrics()

        return self._observe(), self._base_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape != (3,):
            raise ValueError(f"Action must have shape (3,), got {action.shape}")
        requested_action = np.clip(action, self.action_space.low, self.action_space.high)

        row = self.data.iloc[self._t0 + self._step_idx]

        # 1. workload
        if self.use_oracle_workload_projection:
            future_spare = self._future_workload_spare_profile()
            workload_action, workload_intervened, workload_feasible = (
                self.workload.project_recoverable_action(
                    float(requested_action[0]), float(row.fixed_load_mw),
                    float(row.flexible_arrival_mw), future_spare))
        else:
            future_spare = np.zeros(0, dtype=float)
            workload_action = float(requested_action[0])
            workload_intervened = False
            workload_feasible = True
        executed_it_mw, sla_violation_mwh, executed_flex = self.workload.step(
            workload_action, float(row.fixed_load_mw),
            float(row.flexible_arrival_mw))

        # 2. cooling (IT power -> heat, 1h step so MW == MWh)
        enforce_safety = self.cooling.cfg.enforce_thermal_safety
        self.cooling.cfg.enforce_thermal_safety = False
        try:
            unshielded_sp, unshielded_pump, _, _, _ = (
                self.cooling._project_safe_controls(
                    float(requested_action[1]), float(requested_action[2]),
                    executed_it_mw, self.wl_cfg.it_capacity_mw))
        finally:
            self.cooling.cfg.enforce_thermal_safety = enforce_safety
        cool = self.cooling.step(
            float(requested_action[1]), float(requested_action[2]),
            q_it_mwh=executed_it_mw, t_sea=float(row.sst_c),
            q_it_next_max=self.wl_cfg.it_capacity_mw)
        unshielded_action = np.array([
            workload_action, unshielded_sp, unshielded_pump,
        ], dtype=np.float32)
        applied_action = np.array([
            workload_action,
            cool["applied_a_setpoint"],
            cool["applied_a_pump"],
        ], dtype=np.float32)
        safety_override = float(np.mean(np.abs(requested_action - applied_action)))
        thermal_safety_override = float(np.mean(np.abs(
            unshielded_action[1:] - applied_action[1:])))
        rate_limit_override = float(np.mean(np.abs(
            requested_action[1:] - unshielded_action[1:])))

        will_truncate = self._step_idx + 1 >= self.episode_hours
        terminal_unserved_mwh = (
            self.workload.settle_terminal_backlog() if will_truncate else 0.0)
        total_sla_violation_mwh = sla_violation_mwh + terminal_unserved_mwh

        # 3. energy balance and wind matching
        e_total = executed_it_mw + cool["e_cooling_mwh"] + cool["e_pump_mwh"] + cool["e_aux_mwh"]
        wind = match_wind(e_total, float(row.wind_mw),
                          float(row.carbon_kg_per_mwh), float(row.price_usd_per_mwh))

        # 4. reward
        reward, terms = self.reward_fn(
            e_grid_mwh=wind["e_grid_mwh"],
            co2_kg=wind["co2_kg"],
            sla_violation_mwh=total_sla_violation_mwh,
            thermal_violation_k=cool["thermal_violation_k"],
            unused_wind_mwh=wind["e_wind_unused_mwh"],
            action=applied_action, prev_action=self._prev_action,
            e_total_mwh=e_total,
            pump_energy_mwh=cool["e_pump_mwh"],
            thermal_risk_k=cool["thermal_risk_k"],
            safety_override=safety_override,
        )

        # 5. bookkeeping
        m = self._episode_metrics
        m["e_grid_mwh"] += wind["e_grid_mwh"]
        m["co2_kg"] += wind["co2_kg"]
        m["cost_usd"] += wind["cost_usd"]
        m["e_cooling_mwh"] += cool["e_cooling_mwh"]
        m["e_pump_mwh"] += cool["e_pump_mwh"]
        m["e_wind_used_mwh"] += wind["e_wind_used_mwh"]
        m["e_wind_unused_mwh"] += wind["e_wind_unused_mwh"]
        m["e_total_mwh"] += e_total
        m["sla_violation_mwh"] += total_sla_violation_mwh
        m["sla_violation_hours"] += int(total_sla_violation_mwh > 1e-9)
        m["terminal_unserved_mwh"] += terminal_unserved_mwh
        m["thermal_violation_hours"] += int(cool["thermal_violation_k"] > 0)
        m["safety_interventions"] += int(cool["safety_intervened"])
        m["safety_infeasible_hours"] += int(not cool["safety_feasible"])
        m["workload_interventions"] += int(workload_intervened)
        m["workload_infeasible_hours"] += int(not workload_feasible)
        m["action_override"] += safety_override
        m["thermal_safety_override"] += thermal_safety_override
        m["rate_limit_override"] += rate_limit_override
        m["max_t_inlet"] = max(m["max_t_inlet"], cool["t_inlet"])
        m["sum_t_inlet"] += cool["t_inlet"]
        m["action_smoothness"] += float(np.mean((applied_action - self._prev_action) ** 2))

        self._prev_action = applied_action.copy()
        self._step_idx += 1
        terminated = False
        truncated = will_truncate

        info = self._base_info()
        info.update({
            "reward_terms": terms,
            "e_total_mwh": e_total,
            "executed_flexible_mw": executed_flex,
            "deadline_recovery_mwh": self.workload.last_mandatory_recovery_mwh,
            "deadline_sla_violation_mwh": sla_violation_mwh,
            "workload_intervened": workload_intervened,
            "workload_feasible": workload_feasible,
            "oracle_workload_projection_enabled": self.use_oracle_workload_projection,
            "future_workload_spare_mwh": float(future_spare.sum()),
            "terminal_unserved_mwh": terminal_unserved_mwh,
            "sla_violation_mwh": total_sla_violation_mwh,
            "requested_action": requested_action.copy(),
            "unshielded_action": unshielded_action.copy(),
            "applied_action": applied_action.copy(),
            "thermal_safety_override": thermal_safety_override,
            "rate_limit_override": rate_limit_override,
            **cool, **wind,
        })
        if truncated:
            info["episode_metrics"] = self.episode_summary()

        return self._observe(), reward, terminated, truncated, info

    # -------------------------------------------------------------- helpers
    def _observe(self) -> np.ndarray:
        t = min(self._t0 + self._step_idx, len(self.data) - 25)
        row = self.data.iloc[t]
        cap = self.wl_cfg.it_capacity_mw
        wcap = self.wind_cfg.wind_capacity_mw

        if self.use_oracle_forecast_observations:
            wind_fc = self._forecast("wind_mw", t) / max(wcap, 1e-9)
            ci_fc = self._forecast("carbon_kg_per_mwh", t) / 1000.0
        else:
            wind_fc = np.zeros(FORECAST_BUCKETS, dtype=float)
            ci_fc = np.zeros(FORECAST_BUCKETS, dtype=float)

        if "timestamp" in self.data.columns:
            timestamp = pd.Timestamp(row.timestamp)
            hs, hc = cyclical(timestamp.hour, 24)
            ds, dc = cyclical(timestamp.dayofweek, 7)
            days_in_year = 366 if timestamp.is_leap_year else 365
            season = np.sin(2 * np.pi * (timestamp.dayofyear - 1) / days_in_year)
        else:
            hs, hc = cyclical(t % 24, 24)
            ds, dc = cyclical((t // 24) % 7, 7)
            season = np.sin(2 * np.pi * ((t // 24) % 365) / 365)

        c = self.cool_cfg
        obs = np.array([
            (row.fixed_load_mw + row.flexible_arrival_mw) / cap,      # normalized IT arrival
            row.fixed_load_mw / cap,
            row.flexible_arrival_mw / cap,
            self.workload.backlog_mwh / cap,
            self.workload.deadline_pressure(),
            row.sst_c / 30.0,
            self.cooling.cop(float(row.sst_c)) / c.cop_max,
            row.wind_mw / max(wcap, 1e-9),
            *wind_fc,
            row.carbon_kg_per_mwh / 1000.0,
            *ci_fc,
            row.price_usd_per_mwh / 200.0,
            (self.cooling.t_set - c.t_set_min) / (c.t_set_max - c.t_set_min),
            (
                self.cooling.t_inlet - c.t_inlet_allowable_min_c
            ) / (
                c.t_inlet_allowable_max_c - c.t_inlet_allowable_min_c
            ),
            *self._prev_action,
            hs, hc, ds, dc, season,
            max(0.0, (self.episode_hours - self._step_idx) / self.episode_hours),
        ], dtype=np.float32)
        return obs

    def _forecast(self, col: str, t: int) -> np.ndarray:
        """Next-24h forecast compressed into FORECAST_BUCKETS bucket means (perfect foresight)."""
        horizon = self.data[col].values[t + 1: t + 25]
        return horizon.reshape(FORECAST_BUCKETS, -1).mean(axis=1)

    def _future_workload_spare_profile(self) -> np.ndarray:
        remaining_after = self.episode_hours - self._step_idx - 1
        horizon = min(self.wl_cfg.max_delay_hours, max(0, remaining_after))
        if horizon <= 0:
            return np.zeros(0, dtype=float)
        start = self._t0 + self._step_idx + 1
        rows = self.data.iloc[start:start + horizon]
        spare = np.maximum(
            0.0,
            self.wl_cfg.it_capacity_mw
            - rows["fixed_load_mw"].to_numpy(dtype=float)
            - rows["flexible_arrival_mw"].to_numpy(dtype=float),
        )
        return spare

    def _base_info(self) -> dict:
        return {
            "country": self.country,
            "step": self._step_idx,
            "data_hour": self._t0 + self._step_idx,
            "backlog_mwh": self.workload.backlog_mwh,
            "t_set": self.cooling.t_set,
            "t_inlet": self.cooling.t_inlet,
            "t_room": self.cooling.t_inlet,
            "oracle_forecast_observations_enabled": (
                self.use_oracle_forecast_observations),
        }

    def episode_summary(self) -> dict:
        m = dict(self._episode_metrics)
        n = max(self._step_idx, 1)
        m["avg_t_inlet"] = m.pop("sum_t_inlet") / n
        m["max_t_room"] = m["max_t_inlet"]
        m["avg_t_room"] = m["avg_t_inlet"]
        m["action_smoothness"] /= n
        m["mean_action_override"] = m.pop("action_override") / n
        m["mean_thermal_safety_override"] = (
            m.pop("thermal_safety_override") / n)
        m["mean_rate_limit_override"] = m.pop("rate_limit_override") / n
        m["wind_utilization_pct"] = 100.0 * m["e_wind_used_mwh"] / max(
            m["e_wind_used_mwh"] + m["e_wind_unused_mwh"], 1e-9)
        m["final_backlog_mwh"] = self.workload.backlog_mwh
        return m

    def exogenous_window(self, extra_hours: int = 24) -> pd.DataFrame:
        """Exogenous inputs for the current episode (used by MPC with perfect foresight)."""
        return self.data.iloc[self._t0: self._t0 + self.episode_hours + extra_hours].reset_index(drop=True)


def _new_metrics() -> dict:
    return {
        "e_grid_mwh": 0.0, "co2_kg": 0.0, "cost_usd": 0.0,
        "e_cooling_mwh": 0.0, "e_pump_mwh": 0.0,
        "e_wind_used_mwh": 0.0, "e_wind_unused_mwh": 0.0, "e_total_mwh": 0.0,
        "sla_violation_mwh": 0.0, "sla_violation_hours": 0,
        "terminal_unserved_mwh": 0.0,
        "thermal_violation_hours": 0,
        "max_t_inlet": -np.inf, "sum_t_inlet": 0.0,
        "safety_interventions": 0, "safety_infeasible_hours": 0,
        "workload_interventions": 0, "workload_infeasible_hours": 0,
        "action_override": 0.0,
        "thermal_safety_override": 0.0, "rate_limit_override": 0.0,
        "action_smoothness": 0.0,
    }
