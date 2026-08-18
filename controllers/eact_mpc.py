"""Causal constrained MPC variants with adaptive forecast-error tightening."""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from coastaldc_env.forecasting import (
    CausalRidgeForecaster,
    ForecastStressConfig,
    OnlineResidualAdaptor,
    ResidualScaledForecastStress,
)
from coastaldc_env.offshore_wind import match_wind
from controllers.rule_based import RuleBasedController

ForecastMode = Literal["nominal", "static", "adaptive"]


@dataclass
class _RolloutResult:
    cost: float
    constraints: np.ndarray
    actions: np.ndarray


class CausalConstrainedMPCController:
    """Shared constrained optimizer for nominal, static, and adaptive MPC."""

    name = "causal_mpc"
    mode: ForecastMode = "nominal"

    def __init__(
        self,
        env,
        *,
        forecast_dir: str | None = None,
        forecaster: CausalRidgeForecaster | None = None,
        calibration_residuals: np.ndarray | None = None,
        calibration_beta: dict[str, float] | None = None,
        history_tail: pd.DataFrame | None = None,
        horizon: int = 24,
        maxiter: int = 20,
        control_block_hours: int = 6,
        gamma: float = 0.995,
        confidence: float = 0.90,
        residual_window: int = 720,
        intervention_weight: float = 0.1,
        constraint_tolerance: float = 1e-4,
        adaptive_bias_correction: bool = True,
        adaptive_beta_floor: float = 0.0,
        forecast_stress_mode: str = "none",
        forecast_stress_scale: float = 0.0,
        forecast_stress_start_step: int = 0,
        forecast_stress_seed: int = 20260717,
    ):
        if horizon <= 0 or maxiter <= 0 or control_block_hours <= 0:
            raise ValueError("MPC horizon, maxiter, and action block must be positive")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("MPC gamma must be in (0, 1]")
        if intervention_weight < 0.0 or constraint_tolerance <= 0.0:
            raise ValueError("Intervention weight and constraint tolerance are invalid")
        if not 0.0 <= adaptive_beta_floor <= 1.0:
            raise ValueError("Adaptive beta floor must be in [0, 1]")
        self.env = env
        self.horizon = int(horizon)
        self.maxiter = int(maxiter)
        self.control_block_hours = int(control_block_hours)
        self.gamma = float(gamma)
        self.confidence = float(confidence)
        self.residual_window = int(residual_window)
        self.intervention_weight = float(intervention_weight)
        self.constraint_tolerance = float(constraint_tolerance)
        self.adaptive_bias_correction = bool(adaptive_bias_correction)
        self.adaptive_beta_floor = float(adaptive_beta_floor)
        self.forecast_stress_config = ForecastStressConfig(
            mode=forecast_stress_mode,
            scale=float(forecast_stress_scale),
            start_step=int(forecast_stress_start_step),
            seed=int(forecast_stress_seed),
        )

        if forecaster is None:
            if forecast_dir is None:
                raise ValueError("forecast_dir or an injected forecaster is required")
            (forecaster, calibration_residuals, calibration_beta,
             history_tail) = self._load_artifacts(forecast_dir)
        self.forecaster = forecaster
        if tuple(self.forecaster.columns) != tuple(self._forecast_columns):
            raise ValueError("Forecast artifact columns do not match EACT-MPC inputs")
        self._initial_residuals = (
            None if calibration_residuals is None
            else np.asarray(calibration_residuals, dtype=float).copy())
        self._calibration_beta = calibration_beta or {
            column: 0.1 for column in self._forecast_columns}
        self._history_tail = (
            pd.DataFrame(columns=["timestamp", *self._forecast_columns])
            if history_tail is None else history_tail.copy().reset_index(drop=True))
        self._forecast_stressor = None
        if (self.forecast_stress_config.mode != "none"
                and self.forecast_stress_config.scale > 0.0):
            if self._initial_residuals is None:
                raise ValueError(
                    "Forecast stress requires calibration residuals for scaling")
            self._forecast_stressor = ResidualScaledForecastStress(
                self._forecast_columns,
                self._initial_residuals,
                bounds=self.forecaster.bounds,
                config=self.forecast_stress_config,
                window=self.residual_window,
            )
        self._fallback = RuleBasedController()
        self.solver_history: list[dict] = []
        self.reset()

    @property
    def _forecast_columns(self) -> tuple[str, ...]:
        return (
            "fixed_load_mw",
            "flexible_arrival_mw",
            "sst_c",
            "wind_mw",
            "carbon_kg_per_mwh",
        )

    def _load_artifacts(self, forecast_dir: str):
        country = self.env.country
        model_path = os.path.join(forecast_dir, f"causal_ridge_{country}.npz")
        residual_path = os.path.join(
            forecast_dir, f"calibration_residuals_{country}.npz")
        forecaster = CausalRidgeForecaster.load(model_path)
        with np.load(residual_path, allow_pickle=False) as archive:
            columns = tuple(str(value) for value in archive["columns"].tolist())
            if columns != tuple(forecaster.columns):
                raise ValueError("Forecast model and calibration columns differ")
            residuals = archive["residuals"].astype(float)
            beta_values = archive["beta"].astype(float)
            beta = dict(zip(columns, beta_values))
            history = pd.DataFrame(
                archive["history_values"].astype(float), columns=columns)
            history.insert(
                0, "timestamp", pd.to_datetime(archive["history_timestamps"]))
        return forecaster, residuals, beta, history

    def reset(self):
        self._last_feasible_plan: np.ndarray | None = None
        self._last_registered_origin: int | None = None
        self._last_observed_index: int | None = None
        self._cached_inputs: dict | None = None
        self._last_forecast_stress = {
            "forecast_stress_mode": self.forecast_stress_config.mode,
            "forecast_stress_active": False,
            "forecast_stress_mean_abs": 0.0,
            "forecast_stress_max_abs": 0.0,
        }
        self._fallback.reset()
        self.solver_history.clear()
        self._adaptor = None
        if self.mode in ("static", "adaptive"):
            if self._initial_residuals is None:
                raise ValueError(f"{self.mode} MPC requires calibration residuals")
            beta = self._calibration_beta
            if self.mode == "adaptive" and self.adaptive_bias_correction:
                beta = {
                    column: max(float(value), self.adaptive_beta_floor)
                    for column, value in self._calibration_beta.items()
                }
            self._adaptor = OnlineResidualAdaptor(
                self._forecast_columns,
                self.forecaster.config.horizon,
                beta=beta,
                window=self.residual_window,
                confidence=self.confidence,
                initial_residuals=self._initial_residuals,
                bias_correction=self.adaptive_bias_correction,
            )

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        inputs = self._prediction_inputs()
        start = time.perf_counter()
        plan, success, diagnostics = self._optimize(inputs)
        elapsed = time.perf_counter() - start
        diagnostics["solve_time_s"] = elapsed
        diagnostics.update(self._last_forecast_stress)
        diagnostics["adaptive_beta_floor"] = self.adaptive_beta_floor

        if success:
            self._last_feasible_plan = plan.copy()
            action = plan[0]
            diagnostics["fallback"] = "none"
        elif (self.env.episode_hours - self.env._step_idx
              <= self.env.wl_cfg.max_delay_hours):
            action = np.array([1.0, -1.0, 1.0], dtype=np.float32)
            diagnostics["fallback"] = "safe_recovery"
        elif self._last_feasible_plan is not None and len(self._last_feasible_plan) > 1:
            shifted = np.vstack([
                self._last_feasible_plan[1:], self._last_feasible_plan[-1]])
            self._last_feasible_plan = shifted
            action = shifted[0]
            diagnostics["fallback"] = "shifted_plan"
        else:
            action = self._fallback.act(obs, info)
            diagnostics["fallback"] = "rule_based"
        self.solver_history.append(diagnostics)
        return np.clip(
            np.asarray(action, dtype=np.float32),
            self.env.action_space.low,
            self.env.action_space.high,
        )

    def _prediction_inputs(self) -> dict[str, np.ndarray | int | bool]:
        data_index = self.env._t0 + self.env._step_idx
        forecast_origin = data_index + 1
        if self._last_registered_origin == forecast_origin and self._cached_inputs is not None:
            return self._cached_inputs

        if self.mode == "adaptive" and self._last_observed_index != data_index:
            row = self.env.data.iloc[data_index]
            actual = {column: float(row[column]) for column in self._forecast_columns}
            self._adaptor.observe(data_index, actual)
            self._last_observed_index = data_index

        max_lag = max(self.forecaster.config.lags)
        observed_start = max(0, data_index - max_lag + 1)
        observed = self.env.data.iloc[observed_start:data_index + 1][
            ["timestamp", *self._forecast_columns]
            if "timestamp" in self.env.data.columns else list(self._forecast_columns)]
        needed_history = max(0, max_lag - len(observed))
        history = pd.concat([
            self._history_tail.iloc[-needed_history:] if needed_history else
            self._history_tail.iloc[0:0],
            observed,
        ], ignore_index=True)
        if "timestamp" in history.columns:
            history = history.drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        origin_in_history = len(history)
        base = self.forecaster.predict(history, origin=origin_in_history)
        if self._forecast_stressor is not None:
            base, self._last_forecast_stress = self._forecast_stressor.apply(
                base,
                origin=forecast_origin,
                episode_step=self.env._step_idx,
            )
        else:
            self._last_forecast_stress = {
                "forecast_stress_mode": self.forecast_stress_config.mode,
                "forecast_stress_active": False,
                "forecast_stress_mean_abs": 0.0,
                "forecast_stress_max_abs": 0.0,
            }

        if self.mode == "adaptive":
            corrected = self._adaptor.register_forecast(forecast_origin, base)
            lower_error, upper_error = self._adaptor.one_sided_bounds()
        elif self.mode == "static":
            corrected = self._adaptor.correct(base)
            lower_error, upper_error = self._adaptor.one_sided_bounds()
        else:
            corrected = {key: value.copy() for key, value in base.items()}
            lower_error = {key: np.zeros_like(value) for key, value in base.items()}
            upper_error = {key: np.zeros_like(value) for key, value in base.items()}

        future = {
            "fixed_load_mw": corrected["fixed_load_mw"] + upper_error["fixed_load_mw"],
            "flexible_arrival_mw": (
                corrected["flexible_arrival_mw"]
                + upper_error["flexible_arrival_mw"]),
            "sst_c": corrected["sst_c"] + upper_error["sst_c"],
            "wind_mw": corrected["wind_mw"] - lower_error["wind_mw"],
            "carbon_kg_per_mwh": (
                corrected["carbon_kg_per_mwh"]
                + upper_error["carbon_kg_per_mwh"]),
        }
        for column in future:
            low, high = self.forecaster.bounds.get(column, (None, None))
            future[column] = np.clip(
                future[column],
                -np.inf if low is None else low,
                np.inf if high is None else high,
            )
        # Marginal upper errors for fixed and flexible load are strongly
        # correlated. Keep their joint risk scenario inside the physical IT
        # arrival support instead of adding two impossible marginal extremes.
        capacity = float(self.env.wl_cfg.it_capacity_mw)
        future["fixed_load_mw"] = np.clip(
            future["fixed_load_mw"], 0.0, capacity)
        future["flexible_arrival_mw"] = np.clip(
            future["flexible_arrival_mw"],
            0.0,
            np.maximum(0.0, capacity - future["fixed_load_mw"]),
        )

        remaining = self.env.episode_hours - self.env._step_idx
        if remaining <= self.env.wl_cfg.max_delay_hours:
            # See the complete terminal recovery window so flexible arrivals
            # cannot be deferred past the artificial episode boundary.
            H = remaining
        else:
            H = min(self.horizon, remaining)
        row = self.env.data.iloc[data_index]
        required_all = H + self.env.wl_cfg.max_delay_hours

        def include_current(column: str, length: int) -> np.ndarray:
            needed_future = max(0, length - 1)
            values = future[column]
            if needed_future > len(values):
                values = np.pad(values, (0, needed_future - len(values)), mode="edge")
            return np.concatenate([[float(row[column])], values[:needed_future]])

        inputs: dict[str, np.ndarray | int | bool] = {
            "H": H,
            "fixed": include_current("fixed_load_mw", H),
            "flex": include_current("flexible_arrival_mw", H),
            "sst": include_current("sst_c", H),
            "wind": include_current("wind_mw", H),
            "ci": include_current("carbon_kg_per_mwh", H),
            "all_fixed": include_current("fixed_load_mw", required_all),
            "all_flex": include_current("flexible_arrival_mw", required_all),
            "episode_terminal": H == remaining,
        }
        self._last_registered_origin = forecast_origin
        self._cached_inputs = inputs
        return inputs

    def _optimize(self, inputs: dict) -> tuple[np.ndarray, bool, dict]:
        H = int(inputs["H"])
        n_blocks = int(np.ceil(H / self.control_block_hours))
        initial_guesses = self._initial_guesses(H, n_blocks)
        bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * n_blocks
        rollout_cache: dict[bytes, _RolloutResult] = {}

        def evaluate(x: np.ndarray) -> _RolloutResult:
            values = np.asarray(x, dtype=float)
            key = values.tobytes()
            if key not in rollout_cache:
                rollout_cache[key] = self._simulate(values, inputs)
            return rollout_cache[key]

        x0 = initial_guesses[0][1]
        result = minimize(
            lambda x: evaluate(x).cost,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": lambda x: evaluate(x).constraints}],
            options={"maxiter": self.maxiter, "ftol": 1e-6, "disp": False},
        )
        candidates = [("solver", result.x, evaluate(result.x))]
        candidates.extend(
            (source, guess, evaluate(guess))
            for source, guess in initial_guesses)
        feasible_candidates = [
            candidate for candidate in candidates
            if (np.isfinite(candidate[2].cost)
                and float(np.min(candidate[2].constraints))
                >= -self.constraint_tolerance)
        ]
        if feasible_candidates:
            plan_source, _, rollout = min(
                feasible_candidates, key=lambda candidate: candidate[2].cost)
            accepted = True
        else:
            plan_source, _, rollout = max(
                candidates,
                key=lambda candidate: float(np.min(candidate[2].constraints)))
            accepted = False
        min_constraint = float(np.min(rollout.constraints))
        diagnostics = {
            "accepted": accepted,
            "solver_success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "objective": float(rollout.cost),
            "min_constraint": min_constraint,
            "plan_source": plan_source,
        }
        return rollout.actions, accepted, diagnostics

    def _initial_guesses(self, H: int, n_blocks: int) -> list[tuple[str, np.ndarray]]:
        safe_full = np.tile([1.0, -1.0, 1.0], (H, 1))
        guesses: list[tuple[str, np.ndarray]] = []
        if self._last_feasible_plan is not None:
            shifted = np.vstack([
                self._last_feasible_plan[1:], self._last_feasible_plan[-1]])
            count = min(H, len(shifted))
            warm_full = safe_full.copy()
            warm_full[:count] = shifted[:count]
            guesses.append(("warm_start", self._compress_actions(
                warm_full, n_blocks)))
        guesses.append(("safe_initial", self._compress_actions(
            safe_full, n_blocks)))
        # The first item seeds SLSQP; remaining items are retained as feasible
        # candidates if the numerical optimizer exits at an infeasible point.
        return guesses

    def _compress_actions(self, full: np.ndarray, n_blocks: int) -> np.ndarray:
        blocked = [
            full[start:start + self.control_block_hours].mean(axis=0)
            for start in range(0, len(full), self.control_block_hours)
        ]
        if len(blocked) != n_blocks:
            raise RuntimeError("Action-block construction failed")
        return np.asarray(blocked, dtype=float).reshape(-1)

    def _simulate(self, x: np.ndarray, inputs: dict) -> _RolloutResult:
        H = int(inputs["H"])
        n_blocks = int(np.ceil(H / self.control_block_hours))
        blocked = np.asarray(x, dtype=float).reshape(n_blocks, 3)
        actions = np.repeat(blocked, self.control_block_hours, axis=0)[:H]
        wl = copy.deepcopy(self.env.workload)
        cooling = copy.deepcopy(self.env.cooling)
        cooling.cfg.enforce_thermal_safety = False
        previous_action = self.env._prev_action.copy()
        reward_fn = self.env.reward_fn
        constraints: list[float] = []
        total_cost = 0.0
        cap = self.env.wl_cfg.it_capacity_mw
        limit = self.env.cool_cfg.t_inlet_recommended_max_c

        for k in range(H):
            requested = np.clip(actions[k], [-1.0, -1.0, 0.0], [1.0, 1.0, 1.0])
            future_end = min(
                len(inputs["all_fixed"]), k + 1 + self.env.wl_cfg.max_delay_hours)
            future_spare = np.maximum(
                0.0,
                cap
                - np.asarray(inputs["all_fixed"])[k + 1:future_end]
                - np.asarray(inputs["all_flex"])[k + 1:future_end],
            )
            projected_workload, _, _ = wl.project_recoverable_action(
                requested[0], inputs["fixed"][k], inputs["flex"][k], future_spare)
            executed_it, sla_violation, _ = wl.step(
                requested[0], inputs["fixed"][k], inputs["flex"][k])

            cool = cooling.step(
                requested[1], requested[2], executed_it, inputs["sst"][k],
                q_it_next_max=cap)
            applied = np.array([
                requested[0],
                cool["applied_a_setpoint"],
                cool["applied_a_pump"],
            ])
            projected = np.array([
                projected_workload,
                cool["applied_a_setpoint"],
                cool["applied_a_pump"],
            ])
            intervention_distance = float(np.mean(np.abs(applied - projected)))

            e_total = (
                executed_it + cool["e_cooling_mwh"]
                + cool["e_pump_mwh"] + cool["e_aux_mwh"])
            wind = match_wind(
                e_total, inputs["wind"][k], inputs["ci"][k], 0.0)
            reward, _ = reward_fn(
                e_grid_mwh=wind["e_grid_mwh"],
                co2_kg=wind["co2_kg"],
                sla_violation_mwh=sla_violation,
                thermal_violation_k=cool["thermal_violation_k"],
                unused_wind_mwh=wind["e_wind_unused_mwh"],
                action=applied,
                prev_action=previous_action,
                e_total_mwh=e_total,
                pump_energy_mwh=cool["e_pump_mwh"],
                thermal_risk_k=cool["thermal_risk_k"],
                safety_override=intervention_distance,
            )
            total_cost += (self.gamma ** k) * (
                -reward + self.intervention_weight * intervention_distance)
            previous_action = applied
            constraints.extend([
                float(executed_it),
                float(cap - executed_it),
                float(self.constraint_tolerance - sla_violation),
                float(limit - cool["t_inlet"]),
            ])

        if bool(inputs["episode_terminal"]):
            constraints.append(float(self.constraint_tolerance - wl.backlog_mwh))
        else:
            queue = wl.queue.copy()
            for j in range(H, min(
                    len(inputs["all_fixed"]), H + self.env.wl_cfg.max_delay_hours)):
                spare = max(
                    0.0,
                    cap - inputs["all_fixed"][j] - inputs["all_flex"][j])
                wl._drain_queue_oldest_first(queue, spare)
                constraints.append(float(
                    self.constraint_tolerance
                    - (queue[-1] if queue.size else 0.0)))
                if queue.size:
                    queue[1:] = queue[:-1]
                    queue[0] = 0.0
            constraints.append(float(self.constraint_tolerance - queue.sum()))

        total_cost += reward_fn.sla_cost(wl.backlog_mwh)
        return _RolloutResult(
            cost=float(total_cost),
            constraints=np.asarray(constraints, dtype=float),
            actions=actions,
        )


class NominalCausalMPCController(CausalConstrainedMPCController):
    name = "nominal_causal_mpc"
    mode: ForecastMode = "nominal"


class StaticRobustMPCController(CausalConstrainedMPCController):
    name = "static_robust_mpc"
    mode: ForecastMode = "static"


class EACTMPCController(CausalConstrainedMPCController):
    name = "eact_mpc"
    mode: ForecastMode = "adaptive"


class EACTNoBiasMPCController(EACTMPCController):
    """Adaptive residual quantiles without forecast-bias correction."""

    name = "eact_no_bias"

    def __init__(self, env, **kwargs):
        kwargs["adaptive_bias_correction"] = False
        super().__init__(env, **kwargs)


class EACTNoInterventionMPCController(EACTMPCController):
    """EACT ablation without the safety-intervention objective term."""

    name = "eact_no_intervention"

    def __init__(self, env, **kwargs):
        kwargs["intervention_weight"] = 0.0
        super().__init__(env, **kwargs)


class OracleConstrainedMPCController(CausalConstrainedMPCController):
    """Perfect-forecast reference with the same objective and constraints."""

    name = "oracle_constrained_mpc"
    mode: ForecastMode = "nominal"

    def _prediction_inputs(self) -> dict[str, np.ndarray | int | bool]:
        data_index = self.env._t0 + self.env._step_idx
        if self._last_registered_origin == data_index and self._cached_inputs is not None:
            return self._cached_inputs

        remaining = self.env.episode_hours - self.env._step_idx
        if remaining <= self.env.wl_cfg.max_delay_hours:
            H = remaining
        else:
            H = min(self.horizon, remaining)
        required_all = H + self.env.wl_cfg.max_delay_hours

        def actual(column: str, length: int) -> np.ndarray:
            values = self.env.data.iloc[
                data_index:data_index + length][column].to_numpy(dtype=float)
            if len(values) < length:
                values = np.pad(values, (0, length - len(values)), mode="edge")
            return values

        inputs: dict[str, np.ndarray | int | bool] = {
            "H": H,
            "fixed": actual("fixed_load_mw", H),
            "flex": actual("flexible_arrival_mw", H),
            "sst": actual("sst_c", H),
            "wind": actual("wind_mw", H),
            "ci": actual("carbon_kg_per_mwh", H),
            "all_fixed": actual("fixed_load_mw", required_all),
            "all_flex": actual("flexible_arrival_mw", required_all),
            "episode_terminal": H == remaining,
        }
        self._last_registered_origin = data_index
        self._cached_inputs = inputs
        return inputs
