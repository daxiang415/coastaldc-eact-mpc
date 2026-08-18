"""Rolling-horizon MPC controllers with shared physics, objective, and constraints.

``MPCController`` uses full future exogenous inputs and is retained as the
perfect-forecast reference. ``InformationMatchedMPCController`` receives only
the compressed wind/carbon forecast values exposed to SAC plus current state.
Both optimize piecewise-constant [workload, setpoint, pump] action blocks.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import minimize

from coastaldc_env.continuous_env import ObsIndex
from coastaldc_env.offshore_wind import match_wind


class MPCController:
    name = "oracle_mpc"

    def __init__(self, env, horizon: int = 24, replan_every: int = 4,
                 maxiter: int = 10, control_block_hours: int = 4,
                 gamma: float = 0.995):
        if horizon <= 0 or replan_every <= 0 or maxiter <= 0 or control_block_hours <= 0:
            raise ValueError("MPC horizon, intervals, and maxiter must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("MPC gamma must be in (0, 1]")
        self.env = env
        self.horizon = horizon
        self.replan_every = replan_every
        self.maxiter = maxiter
        self.control_block_hours = control_block_hours
        self.gamma = gamma
        self.reset()

    def reset(self):
        self._plan = None
        self._plan_cursor = 0

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        if self._plan is None or self._plan_cursor >= self.replan_every:
            self._plan = self._optimize(obs)
            self._plan_cursor = 0
        a = self._plan[self._plan_cursor]
        self._plan_cursor += 1
        return a.astype(np.float32)

    # ------------------------------------------------------------------
    def _prediction_inputs(self, obs: np.ndarray) -> dict[str, np.ndarray | int]:
        """Return full perfect-forecast inputs for the reference MPC."""
        env = self.env
        t = env._step_idx
        window = env.exogenous_window(extra_hours=self.horizon)
        H = min(self.horizon, len(window) - t - 1)
        future = window.iloc[t:]
        return {
            "H": H,
            "fixed": future["fixed_load_mw"].to_numpy(dtype=float)[:H],
            "flex": future["flexible_arrival_mw"].to_numpy(dtype=float)[:H],
            "sst": future["sst_c"].to_numpy(dtype=float)[:H],
            "wind": future["wind_mw"].to_numpy(dtype=float)[:H],
            "ci": future["carbon_kg_per_mwh"].to_numpy(dtype=float)[:H],
            "price": future["price_usd_per_mwh"].to_numpy(dtype=float)[:H],
            "all_fixed": future["fixed_load_mw"].to_numpy(dtype=float),
            "all_flex": future["flexible_arrival_mw"].to_numpy(dtype=float),
        }

    def _optimize(self, obs: np.ndarray) -> np.ndarray:
        env = self.env
        inputs = self._prediction_inputs(obs)
        H = int(inputs["H"])
        n_blocks = int(np.ceil(H / self.control_block_hours))

        wl0 = copy.deepcopy(env.workload)
        cool0 = copy.deepcopy(env.cooling)
        prev_a0 = env._prev_action.copy()
        rw = env.reward_fn

        fixed = np.asarray(inputs["fixed"], dtype=float)
        flex = np.asarray(inputs["flex"], dtype=float)
        sst = np.asarray(inputs["sst"], dtype=float)
        wind = np.asarray(inputs["wind"], dtype=float)
        ci = np.asarray(inputs["ci"], dtype=float)
        price = np.asarray(inputs["price"], dtype=float)
        all_fixed = np.asarray(inputs["all_fixed"], dtype=float)
        all_flex = np.asarray(inputs["all_flex"], dtype=float)

        def expand_actions(x: np.ndarray) -> np.ndarray:
            blocked = x.reshape(n_blocks, 3)
            return np.repeat(blocked, self.control_block_hours, axis=0)[:H]

        def rollout_cost(x: np.ndarray) -> float:
            acts = expand_actions(x)
            wl = copy.deepcopy(wl0)
            cl = copy.deepcopy(cool0)
            prev_a = prev_a0
            total = 0.0
            for k in range(H):
                requested = np.clip(acts[k], [-1, -1, 0], [1, 1, 1])
                remaining_hours = max(1, env.episode_hours - (env._step_idx + k))
                future_h = min(
                    wl.cfg.max_delay_hours, remaining_hours - 1)
                if future_h > 0:
                    future_start = k + 1
                    future_spare = np.maximum(
                        0.0,
                        wl.cfg.it_capacity_mw
                        - all_fixed[future_start:future_start + future_h]
                        - all_flex[future_start:future_start + future_h],
                    )
                else:
                    future_spare = np.zeros(0, dtype=float)
                workload_action, _, _ = wl.project_recoverable_action(
                    requested[0], fixed[k], flex[k], future_spare)
                exec_it, sla_v, _ = wl.step(workload_action, fixed[k], flex[k])
                c = cl.step(requested[1], requested[2],
                            q_it_mwh=exec_it, t_sea=sst[k],
                            q_it_next_max=wl.cfg.it_capacity_mw)
                applied = np.array([
                    workload_action,
                    c["applied_a_setpoint"],
                    c["applied_a_pump"],
                ])
                safety_override = float(np.mean(np.abs(requested - applied)))
                e_total = exec_it + c["e_cooling_mwh"] + c["e_pump_mwh"] + c["e_aux_mwh"]
                w = match_wind(e_total, wind[k], ci[k], price[k])
                r, _ = rw(e_grid_mwh=w["e_grid_mwh"], co2_kg=w["co2_kg"],
                          sla_violation_mwh=sla_v,
                          thermal_violation_k=c["thermal_violation_k"],
                          unused_wind_mwh=w["e_wind_unused_mwh"],
                          action=applied, prev_action=prev_a,
                          e_total_mwh=e_total,
                          pump_energy_mwh=c["e_pump_mwh"],
                          thermal_risk_k=c["thermal_risk_k"],
                          safety_override=safety_override)
                total -= (self.gamma ** k) * r
                prev_a = applied
            # terminal backlog penalty: leftover work must eventually run
            total += rw.sla_cost(wl.backlog_mwh)
            return total

        full_guess = np.tile([0.0, 0.0, 0.5], (H, 1))
        if self._plan is not None and len(self._plan) > self.replan_every:
            warm = self._plan[self.replan_every:]
            n_warm = min(len(warm), H)
            full_guess[:n_warm] = warm[:n_warm]
        block_guess = []
        for start in range(0, H, self.control_block_hours):
            block_guess.append(full_guess[start:start + self.control_block_hours].mean(axis=0))
        x0 = np.asarray(block_guess).reshape(-1)

        bounds = [(-1, 1), (-1, 1), (0, 1)] * n_blocks
        res = minimize(rollout_cost, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": self.maxiter, "eps": 5e-3})
        return expand_actions(res.x)


class InformationMatchedMPCController(MPCController):
    """MPC with exactly SAC's wind/carbon forecast resolution and no future rows."""

    name = "im_mpc"

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        if self.horizon > 24:
            raise ValueError("Information-matched MPC horizon must not exceed 24 hours")

    @staticmethod
    def _expand_six_hour_buckets(values: np.ndarray, horizon: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (4,):
            raise ValueError("Information-matched MPC requires four 6-hour forecast bins")
        return np.repeat(values, 6)[:horizon]

    def _prediction_inputs(self, obs: np.ndarray) -> dict[str, np.ndarray | int]:
        """Build forecasts from the current SAC observation without future-data access."""
        env = self.env
        obs = np.asarray(obs, dtype=float)
        if obs.shape != env.observation_space.shape:
            raise ValueError("Observation shape does not match the environment")

        H = min(self.horizon, env.episode_hours - env._step_idx)
        cap = env.wl_cfg.it_capacity_mw
        wind_capacity = env.wind_cfg.wind_capacity_mw
        forecast_length = H + env.wl_cfg.max_delay_hours

        wind_bins = obs[ObsIndex.WIND_FC] * wind_capacity
        carbon_bins = obs[ObsIndex.CARBON_FC] * 1000.0
        wind = self._expand_six_hour_buckets(wind_bins, H)
        carbon = self._expand_six_hour_buckets(carbon_bins, H)

        fixed_now = float(obs[ObsIndex.FIXED_LOAD] * cap)
        flex_now = float(obs[ObsIndex.FLEX_ARRIVAL] * cap)
        sst_now = float(obs[ObsIndex.SST] * 30.0)
        price_now = float(obs[ObsIndex.PRICE] * 200.0)
        return {
            "H": H,
            "fixed": np.full(H, fixed_now),
            "flex": np.full(H, flex_now),
            "sst": np.full(H, sst_now),
            "wind": wind,
            "ci": carbon,
            "price": np.full(H, price_now),
            "all_fixed": np.full(forecast_length, fixed_now),
            "all_flex": np.full(forecast_length, flex_now),
        }
