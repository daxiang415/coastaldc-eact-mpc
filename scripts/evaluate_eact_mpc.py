"""Evaluate paper-facing causal MPC variants on paired 2025 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES, CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.forecasting import FORECAST_STRESS_MODES  # noqa: E402
from coastaldc_env.reward import RewardWeights  # noqa: E402
from coastaldc_env.swhp_cooling import CoolingConfig  # noqa: E402
from controllers.eact_mpc import (  # noqa: E402
    EACTMPCController,
    EACTNoBiasMPCController,
    EACTNoInterventionMPCController,
    NominalCausalMPCController,
    OracleConstrainedMPCController,
    StaticRobustMPCController,
)
from controllers.no_control import NoControlController  # noqa: E402
from controllers.rule_based import RuleBasedController  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed_multiyear", "2025")
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias")
THERMAL_METRIC_SCHEMA = "ashrae_a1_inlet_v1"
THERMAL_STATE_SEMANTICS = (
    "representative_lumped_it_equipment_inlet_air_temperature_proxy")
CONTROLLER_NAMES = (
    "no_control", "rule_based", "nominal", "static", "eact",
    "eact_no_bias", "eact_no_intervention", "oracle")
DEFAULT_CONTROLLERS = CONTROLLER_NAMES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=COUNTRIES)
    parser.add_argument("--controllers", nargs="*", choices=CONTROLLER_NAMES,
                        default=list(DEFAULT_CONTROLLERS))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument(
        "--progress-every", type=int, default=168,
        help="Flush a progress line after this many environment steps; zero disables",
    )
    parser.add_argument(
        "--start-hours", nargs="*", type=int,
        help="Explicit paired episode starts; overrides --episodes",
    )
    parser.add_argument(
        "--start-timestamps", nargs="*",
        help="Exact timestamps resolved independently in each country file",
    )
    parser.add_argument("--continuous-year", action="store_true")
    parser.add_argument(
        "--oracle-workload-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the environment safety layer to inspect true future workload",
    )
    parser.add_argument(
        "--thermal-safety-shield",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project cooling actions into the two-step thermal viability set",
    )
    parser.add_argument(
        "--cooling-conductance-multiplier",
        type=float,
        default=1.0,
        help=(
            "Available heat-transfer conductance relative to the nominal "
            "CoolingConfig value"),
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--maxiter", type=int, default=20)
    parser.add_argument("--block-hours", type=int, default=6)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-4)
    parser.add_argument("--intervention-weight", type=float, default=0.1)
    parser.add_argument("--weight-grid", type=float, default=1.0)
    parser.add_argument("--weight-co2", type=float, default=2.0)
    parser.add_argument("--weight-total", type=float, default=0.2)
    parser.add_argument("--weight-smooth", type=float, default=0.5)
    parser.add_argument(
        "--adaptive-beta-floor", type=float, default=0.10,
        help=(
            "Minimum EACT EWMA update rate; 0.10 reduces old bias weight "
            "to about 10 percent within the 24-hour MPC horizon"),
    )
    parser.add_argument(
        "--forecast-stress", choices=FORECAST_STRESS_MODES, default="none")
    parser.add_argument("--forecast-stress-scale", type=float, default=0.0)
    parser.add_argument("--forecast-stress-start-step", type=int, default=0)
    parser.add_argument("--forecast-stress-seed", type=int, default=20260717)
    parser.add_argument("--out-dir", default=os.path.join(
        ROOT, "results", "eact_mpc_evaluation"))
    parser.add_argument("--tag", default="v1")
    return parser


def objective_weights_from_args(args) -> RewardWeights:
    values = {
        "w_grid": float(args.weight_grid),
        "w_co2": float(args.weight_co2),
        "w_total": float(args.weight_total),
        "w_smooth": float(args.weight_smooth),
    }
    if any(not np.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("Objective weights must be finite and nonnegative")
    return RewardWeights(**values)


def cooling_config_from_args(args) -> CoolingConfig:
    multiplier = float(args.cooling_conductance_multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(
            "Cooling conductance multiplier must be finite and positive")
    nominal = CoolingConfig()
    return CoolingConfig(
        conductance_mw_per_k=(
            nominal.conductance_mw_per_k * multiplier),
        enforce_thermal_safety=args.thermal_safety_shield,
    )


def build_controllers(env, args) -> list:
    shared = dict(
        forecast_dir=args.forecast_dir,
        horizon=args.horizon,
        maxiter=args.maxiter,
        control_block_hours=args.block_hours,
        gamma=args.gamma,
        confidence=args.confidence,
        constraint_tolerance=args.constraint_tolerance,
        intervention_weight=args.intervention_weight,
        adaptive_beta_floor=args.adaptive_beta_floor,
        forecast_stress_mode=args.forecast_stress,
        forecast_stress_scale=args.forecast_stress_scale,
        forecast_stress_start_step=args.forecast_stress_start_step,
        forecast_stress_seed=args.forecast_stress_seed,
    )
    requested = set(args.controllers)
    controllers = []
    if "no_control" in requested:
        controllers.append(NoControlController())
    if "rule_based" in requested:
        controllers.append(RuleBasedController())
    if "nominal" in requested:
        controllers.append(NominalCausalMPCController(env, **shared))
    if "static" in requested:
        controllers.append(StaticRobustMPCController(env, **shared))
    if "eact" in requested:
        controllers.append(EACTMPCController(env, **shared))
    if "eact_no_bias" in requested:
        controllers.append(EACTNoBiasMPCController(env, **shared))
    if "eact_no_intervention" in requested:
        controllers.append(EACTNoInterventionMPCController(env, **shared))
    if "oracle" in requested:
        controllers.append(OracleConstrainedMPCController(env, **shared))
    return controllers


def resolve_start_timestamps(
    data: pd.DataFrame,
    timestamps: list[str],
    episode_hours: int,
) -> list[int]:
    """Resolve exact timestamps to valid episode start indices."""
    if "timestamp" not in data.columns:
        raise ValueError("Timestamp starts require a timestamp column")
    parsed = pd.to_datetime(data["timestamp"], errors="raise")
    if parsed.duplicated().any():
        raise ValueError("Episode data contain duplicate timestamps")
    lookup = {timestamp: index for index, timestamp in enumerate(parsed)}
    starts = []
    for value in timestamps:
        timestamp = pd.Timestamp(value)
        if timestamp not in lookup:
            raise ValueError(f"Start timestamp is absent from episode data: {value}")
        starts.append(int(lookup[timestamp]))
    return validate_start_hours(starts, len(data), episode_hours)


def validate_start_hours(
    starts: list[int], data_hours: int, episode_hours: int
) -> list[int]:
    """Validate explicit starts without silently clipping or deduplicating."""
    values = [int(value) for value in starts]
    if not values:
        raise ValueError("Explicit episode starts cannot be empty")
    if len(set(values)) != len(values):
        raise ValueError("Explicit episode starts must be unique")
    max_start = data_hours - episode_hours
    invalid = [value for value in values if value < 0 or value > max_start]
    if invalid:
        raise ValueError(
            f"Episode starts outside [0, {max_start}]: {invalid}")
    return values


def resolve_episode_starts(env, args) -> list[int | None]:
    if args.continuous_year:
        if args.start_hours is not None or args.start_timestamps is not None:
            raise ValueError("Continuous-year evaluation always starts at hour 0")
        return [0]
    if args.start_hours is not None and args.start_timestamps is not None:
        raise ValueError("Use either --start-hours or --start-timestamps, not both")
    if args.start_hours is not None:
        return validate_start_hours(
            args.start_hours, len(env.data), env.episode_hours)
    if args.start_timestamps is not None:
        return resolve_start_timestamps(
            env.data, args.start_timestamps, env.episode_hours)
    return [None] * args.episodes


def run_detailed_episode(
    env,
    controller,
    *,
    seed: int,
    start_hour: int | None,
    constraint_tolerance: float = 1e-4,
    progress_every: int = 0,
    progress_label: str = "",
) -> dict:
    options = None if start_hour is None else {"start_hour": start_hour}
    obs, info = env.reset(seed=seed, options=options)
    reset_info = dict(info)
    start_timestamp = (
        pd.Timestamp(env.data.iloc[env._t0]["timestamp"]).isoformat()
        if "timestamp" in env.data.columns else None
    )
    if hasattr(controller, "reset"):
        controller.reset()

    step_rows = []
    requested_actions = []
    applied_actions = []
    done = False
    while not done:
        action = np.asarray(controller.act(obs, info), dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)
        applied = np.asarray(info.get("applied_action", action), dtype=np.float32)
        requested_actions.append(action)
        applied_actions.append(applied)
        thermal = thermal_step_metrics(
            float(info["t_inlet"]), env.cool_cfg, constraint_tolerance)
        step_rows.append({
            "step": len(step_rows),
            "reward": float(reward),
            "e_grid_mwh": float(info["e_grid_mwh"]),
            "co2_kg": float(info["co2_kg"]),
            "e_total_mwh": float(info["e_total_mwh"]),
            "e_cooling_mwh": float(info["e_cooling_mwh"]),
            "e_pump_mwh": float(info["e_pump_mwh"]),
            "e_wind_used_mwh": float(info["e_wind_used_mwh"]),
            "e_wind_unused_mwh": float(info["e_wind_unused_mwh"]),
            "sla_violation_mwh": float(info["sla_violation_mwh"]),
            **thermal,
            "safety_intervention": int(info["safety_intervened"]),
            "safety_feasible": int(info["safety_feasible"]),
            "safety_infeasible": int(not info["safety_feasible"]),
            "workload_intervention": int(info["workload_intervened"]),
            "workload_feasible": int(info["workload_feasible"]),
            "action_override": float(np.mean(np.abs(action - applied))),
            "workload_override": float(abs(action[0] - applied[0])),
            "thermal_override": float(np.mean(np.abs(action[1:] - applied[1:]))),
            "thermal_safety_override": float(info["thermal_safety_override"]),
            "rate_limit_override": float(info["rate_limit_override"]),
            "requested_workload": float(action[0]),
            "requested_setpoint": float(action[1]),
            "requested_pump": float(action[2]),
            "applied_workload": float(applied[0]),
            "applied_setpoint": float(applied[1]),
            "applied_pump": float(applied[2]),
            "backlog_mwh": float(info["backlog_mwh"]),
        })
        obs = next_obs
        done = terminated or truncated
        if progress_every > 0 and (
                len(step_rows) % progress_every == 0 or done):
            print(
                f"progress | {progress_label} | "
                f"{len(step_rows)}/{env.episode_hours} h",
                flush=True,
            )

    return {
        "reset_info": reset_info,
        "start_timestamp": start_timestamp,
        "metrics": env.episode_summary(),
        "steps": pd.DataFrame(step_rows),
        "requested_actions": np.asarray(requested_actions),
        "applied_actions": np.asarray(applied_actions),
    }


def thermal_step_metrics(
    t_inlet_c: float,
    cooling_config: CoolingConfig,
    constraint_tolerance: float,
) -> dict[str, float | int]:
    if not np.isfinite(t_inlet_c):
        raise ValueError("Inlet temperature must be finite")
    if not np.isfinite(constraint_tolerance) or constraint_tolerance <= 0.0:
        raise ValueError("Constraint tolerance must be finite and positive")
    rec_min = cooling_config.t_inlet_recommended_min_c
    rec_max = cooling_config.t_inlet_recommended_max_c
    allow_max = cooling_config.t_inlet_allowable_max_c
    recommended_excess = max(0.0, float(t_inlet_c) - rec_max)
    allowable_excess = max(0.0, float(t_inlet_c) - allow_max)
    return {
        "t_inlet_c": float(t_inlet_c),
        "recommended_compliant": int(
            rec_min - constraint_tolerance
            <= t_inlet_c
            <= rec_max + constraint_tolerance),
        "recommended_exceedance_event": int(
            recommended_excess > constraint_tolerance),
        "recommended_exceedance_c": recommended_excess,
        "allowable_exceedance_event": int(
            allowable_excess > constraint_tolerance),
        "allowable_exceedance_c": allowable_excess,
        "allowable_headroom_c": float(allow_max - t_inlet_c),
    }


def aggregate_thermal_metrics(steps: pd.DataFrame) -> dict[str, float]:
    if steps.empty:
        raise ValueError("Thermal metrics require at least one hourly row")
    required = {
        "t_inlet_c",
        "recommended_compliant",
        "recommended_exceedance_event",
        "recommended_exceedance_c",
        "allowable_exceedance_event",
        "allowable_exceedance_c",
        "allowable_headroom_c",
    }
    missing = sorted(required.difference(steps.columns))
    if missing:
        raise ValueError(f"Thermal metric columns are missing: {missing}")
    temperatures = steps["t_inlet_c"].to_numpy(dtype=float)
    recommended_excess = steps[
        "recommended_exceedance_c"].to_numpy(dtype=float)
    denominator = (
        CoolingConfig().t_inlet_allowable_max_c
        - CoolingConfig().t_inlet_recommended_max_c
    ) * len(steps)
    return {
        "recommended_compliance_pct": float(
            100.0 * steps["recommended_compliant"].mean()),
        "recommended_exceedance_hours": float(
            steps["recommended_exceedance_event"].sum()),
        "recommended_exceedance_degc_h": float(recommended_excess.sum()),
        "allowable_exceedance_hours": float(
            steps["allowable_exceedance_event"].sum()),
        "allowable_exceedance_degc_h": float(
            steps["allowable_exceedance_c"].sum()),
        "mean_t_inlet_c": float(np.mean(temperatures)),
        "p95_t_inlet_c": float(np.quantile(temperatures, 0.95)),
        "p99_t_inlet_c": float(np.quantile(temperatures, 0.99)),
        "max_t_inlet_c": float(np.max(temperatures)),
        "min_allowable_headroom_c": float(
            steps["allowable_headroom_c"].min()),
        "temporal_rci_hi_pct": float(
            100.0 * (1.0 - recommended_excess.sum() / denominator)),
    }


def episode_row(country: str, algorithm: str, episode: int,
                reset_seed: int, trajectory: dict) -> dict:
    steps = trajectory["steps"]
    requested = trajectory["requested_actions"]
    applied = trajectory["applied_actions"]
    return {
        "country": country,
        "algorithm": algorithm,
        "episode": episode,
        "reset_seed": reset_seed,
        "start_hour": int(trajectory["reset_info"]["data_hour"]),
        "start_timestamp": trajectory["start_timestamp"],
        "episode_return": float(steps["reward"].sum()),
        **{key: float(value) for key, value in trajectory["metrics"].items()},
        **aggregate_thermal_metrics(steps),
        "safety_override_fraction": float(
            np.mean(np.any(np.abs(requested - applied) > 1e-6, axis=1))),
        "mean_abs_action_override": float(np.abs(requested - applied).mean()),
    }


def weekly_rows(country: str, algorithm: str, episode: int,
                reset_seed: int, trajectory: dict) -> list[dict]:
    steps = trajectory["steps"].copy()
    steps["week"] = steps["step"] // 168
    reset_info = trajectory.get("reset_info", {})
    start_hour = int(reset_info.get("data_hour", -1))
    start_timestamp = trajectory.get("start_timestamp")
    rows = []
    sum_columns = [
        "reward", "e_grid_mwh", "co2_kg", "e_total_mwh", "e_cooling_mwh",
        "e_pump_mwh", "e_wind_used_mwh", "e_wind_unused_mwh",
        "sla_violation_mwh", "safety_intervention",
        "recommended_exceedance_event", "recommended_exceedance_c",
        "allowable_exceedance_event", "allowable_exceedance_c",
        "safety_infeasible", "workload_intervention", "thermal_safety_override",
        "rate_limit_override",
    ]
    for week, block in steps.groupby("week", sort=True):
        sums = block[sum_columns].sum()
        rows.append({
            "country": country,
            "algorithm": algorithm,
            "episode": episode,
            "reset_seed": reset_seed,
            "start_hour": start_hour,
            "start_timestamp": start_timestamp,
            "week": int(week),
            "start_step": int(block["step"].iloc[0]),
            "n_hours": int(len(block)),
            **{column: float(sums[column]) for column in sum_columns},
            **aggregate_thermal_metrics(block),
            "mean_action_override": float(block["action_override"].mean()),
            "final_backlog_mwh": float(block["backlog_mwh"].iloc[-1]),
        })
    return rows


def solver_rows(country: str, algorithm: str, episode: int,
                controller) -> list[dict]:
    history = getattr(controller, "solver_history", None)
    if history is None:
        return []
    return [
        {"country": country, "algorithm": algorithm, "episode": episode,
         "step": step, **diagnostics}
        for step, diagnostics in enumerate(history)
    ]


def summarize(episodes: pd.DataFrame, solver: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "episode_return", "e_grid_mwh", "co2_kg", "e_total_mwh",
        "e_cooling_mwh", "e_pump_mwh", "wind_utilization_pct",
        "sla_violation_mwh", "terminal_unserved_mwh",
        "thermal_violation_hours", "safety_interventions",
        "safety_infeasible_hours", "workload_interventions",
        "recommended_compliance_pct", "recommended_exceedance_hours",
        "recommended_exceedance_degc_h", "allowable_exceedance_hours",
        "allowable_exceedance_degc_h", "mean_t_inlet_c", "p95_t_inlet_c",
        "p99_t_inlet_c", "max_t_inlet_c", "min_allowable_headroom_c",
        "temporal_rci_hi_pct", "mean_abs_action_override",
    ]
    rows = []
    for (country, algorithm), group in episodes.groupby(["country", "algorithm"]):
        row = {"country": country, "algorithm": algorithm,
               "n_episodes": len(group)}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=0))
        if not solver.empty:
            ctrl_solver = solver[
                (solver["country"] == country) & (solver["algorithm"] == algorithm)]
            if not ctrl_solver.empty:
                fallback_counts = Counter(ctrl_solver["fallback"])
                row.update({
                    "accepted_plan_rate": float(ctrl_solver["accepted"].mean()),
                    "solver_convergence_rate": float(
                        ctrl_solver["solver_success"].mean()),
                    "solver_fallback_rate": float(
                        np.mean(ctrl_solver["fallback"] != "none")),
                    "solver_rule_fallbacks": int(fallback_counts["rule_based"]),
                    "solver_safe_recovery_fallbacks": int(
                        fallback_counts["safe_recovery"]),
                    "solver_shifted_fallbacks": int(fallback_counts["shifted_plan"]),
                    "solve_time_mean_s": float(ctrl_solver["solve_time_s"].mean()),
                    "solve_time_p95_s": float(
                        ctrl_solver["solve_time_s"].quantile(0.95)),
                    "min_constraint": float(ctrl_solver["min_constraint"].min()),
                })
        rows.append(row)
    return pd.DataFrame(rows)


def _file_record(path: str) -> dict:
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        return {"path": resolved, "exists": False}
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": resolved,
        "exists": True,
        "size_bytes": os.path.getsize(resolved),
        "sha256": digest.hexdigest(),
    }


def build_manifest(
    args,
    starts_by_country: dict[str, list[int | None]],
    outputs: dict[str, tuple[str, pd.DataFrame]],
) -> dict:
    input_files = []
    for country in args.countries:
        input_files.append(_file_record(os.path.join(
            args.data_dir, f"hourly_inputs_{country}.csv")))
        if any(name in args.controllers for name in (
                "nominal", "static", "eact", "eact_no_bias",
                "eact_no_intervention")):
            input_files.extend([
                _file_record(os.path.join(
                    args.forecast_dir, f"causal_ridge_{country}.npz")),
                _file_record(os.path.join(
                    args.forecast_dir,
                    f"calibration_residuals_{country}.npz")),
            ])
    return {
        "schema_version": 2,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "countries": list(args.countries),
            "controllers": list(args.controllers),
            "episode_hours": 8736 if args.continuous_year else args.episode_hours,
            "continuous_year": bool(args.continuous_year),
            "horizon": args.horizon,
            "maxiter": args.maxiter,
            "block_hours": args.block_hours,
            "gamma": args.gamma,
            "confidence": args.confidence,
            "constraint_tolerance": args.constraint_tolerance,
            "intervention_weight": args.intervention_weight,
            "weight_grid": args.weight_grid,
            "weight_co2": args.weight_co2,
            "weight_total": args.weight_total,
            "weight_smooth": args.weight_smooth,
            "adaptive_beta_floor": args.adaptive_beta_floor,
            "oracle_workload_projection": args.oracle_workload_projection,
            "thermal_safety_shield": args.thermal_safety_shield,
            "cooling_conductance_nominal_mw_per_k": (
                CoolingConfig().conductance_mw_per_k),
            "cooling_conductance_multiplier": (
                args.cooling_conductance_multiplier),
            "cooling_conductance_effective_mw_per_k": (
                CoolingConfig().conductance_mw_per_k
                * args.cooling_conductance_multiplier),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
            "thermal_state_semantics": THERMAL_STATE_SEMANTICS,
            "t_inlet_recommended_min_c": 18.0,
            "t_inlet_recommended_max_c": 27.0,
            "t_inlet_allowable_min_c": 15.0,
            "t_inlet_allowable_max_c": 32.0,
            "forecast_stress": args.forecast_stress,
            "forecast_stress_scale": args.forecast_stress_scale,
            "forecast_stress_start_step": args.forecast_stress_start_step,
            "forecast_stress_seed": args.forecast_stress_seed,
            "data_dir": os.path.abspath(args.data_dir),
            "forecast_dir": os.path.abspath(args.forecast_dir),
            "tag": args.tag,
        },
        "starts_by_country": starts_by_country,
        "inputs": input_files,
        "outputs": {
            name: {"path": os.path.abspath(path), "rows": int(len(frame))}
            for name, (path, frame) in outputs.items()
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.episodes <= 0 or args.episode_hours <= 0 or args.episode_offset < 0:
        raise ValueError(
            "Episodes and episode hours must be positive; offset must be nonnegative")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be nonnegative")
    if (not np.isfinite(args.constraint_tolerance)
            or args.constraint_tolerance <= 0.0):
        raise ValueError("--constraint-tolerance must be finite and positive")
    if args.forecast_stress_scale < 0.0:
        raise ValueError("--forecast-stress-scale must be nonnegative")
    cooling_config_from_args(args)
    if not 0.0 <= args.adaptive_beta_floor <= 1.0:
        raise ValueError("--adaptive-beta-floor must be in [0, 1]")
    reward_weights = objective_weights_from_args(args)
    if args.forecast_stress_start_step < 0 or args.forecast_stress_seed < 0:
        raise ValueError("Forecast stress start step and seed must be nonnegative")
    episode_hours = 8736 if args.continuous_year else args.episode_hours
    os.makedirs(args.out_dir, exist_ok=True)

    episode_records = []
    weekly_records = []
    solver_records = []
    hourly_records = []
    starts_by_country: dict[str, list[int | None]] = {}
    for country in args.countries:
        explicit_starts = (
            args.continuous_year
            or args.start_hours is not None
            or args.start_timestamps is not None
        )
        env = CoastalDCContinuousEnv(
            country=country,
            data_dir=args.data_dir,
            episode_hours=episode_hours,
            cooling_config=cooling_config_from_args(args),
            reward_weights=reward_weights,
            random_episode_start=not explicit_starts,
            use_oracle_workload_projection=args.oracle_workload_projection,
            use_oracle_forecast_observations=False,
            seed=0,
        )
        episode_starts = resolve_episode_starts(env, args)
        starts_by_country[country] = episode_starts
        for controller in build_controllers(env, args):
            for episode_index, start_hour in enumerate(episode_starts):
                episode = args.episode_offset + episode_index
                reset_seed = 5000 + episode
                trajectory = run_detailed_episode(
                    env,
                    controller,
                    seed=reset_seed,
                    start_hour=start_hour,
                    constraint_tolerance=args.constraint_tolerance,
                    progress_every=args.progress_every,
                    progress_label=(
                        f"{country} {controller.name} episode={episode}"),
                )
                record = episode_row(
                    country, controller.name, episode, reset_seed, trajectory)
                episode_records.append(record)
                weekly_records.extend(weekly_rows(
                    country, controller.name, episode, reset_seed, trajectory))
                solver_records.extend(solver_rows(
                    country, controller.name, episode, controller))
                hourly = trajectory["steps"].copy()
                hourly.insert(0, "reset_seed", reset_seed)
                hourly.insert(0, "episode", episode)
                hourly.insert(0, "algorithm", controller.name)
                hourly.insert(0, "country", country)
                hourly_records.extend(hourly.to_dict("records"))
                print(
                    f"{country} | {controller.name:22s} | "
                    f"return {record['episode_return']:9.2f} | "
                    f"grid {record['e_grid_mwh']:9.2f} MWh | "
                    f"safety {record['safety_interventions']:6.0f}",
                    flush=True)

    episode_df = pd.DataFrame(episode_records)
    weekly_df = pd.DataFrame(weekly_records)
    solver_df = pd.DataFrame(solver_records)
    hourly_df = pd.DataFrame(hourly_records)
    summary_df = summarize(episode_df, solver_df)
    frames = {
        "summary": summary_df,
        "episodes": episode_df,
        "weekly": weekly_df,
        "hourly": hourly_df,
        "solver": solver_df,
    }
    for frame in frames.values():
        if frame.empty:
            continue
        frame["mpc_horizon"] = args.horizon
        frame["confidence"] = args.confidence
        frame["constraint_tolerance"] = args.constraint_tolerance
        frame["intervention_weight"] = args.intervention_weight
        frame["weight_grid"] = args.weight_grid
        frame["weight_co2"] = args.weight_co2
        frame["weight_total"] = args.weight_total
        frame["weight_smooth"] = args.weight_smooth
        frame["adaptive_beta_floor"] = args.adaptive_beta_floor
        frame["thermal_safety_shield"] = args.thermal_safety_shield
        frame["cooling_conductance_nominal_mw_per_k"] = (
            CoolingConfig().conductance_mw_per_k)
        frame["cooling_conductance_multiplier"] = (
            args.cooling_conductance_multiplier)
        frame["cooling_conductance_effective_mw_per_k"] = (
            CoolingConfig().conductance_mw_per_k
            * args.cooling_conductance_multiplier)
        frame["thermal_metric_schema"] = THERMAL_METRIC_SCHEMA
        frame["thermal_state_semantics"] = THERMAL_STATE_SEMANTICS
        frame["t_inlet_recommended_min_c"] = 18.0
        frame["t_inlet_recommended_max_c"] = 27.0
        frame["t_inlet_allowable_min_c"] = 15.0
        frame["t_inlet_allowable_max_c"] = 32.0
        frame["forecast_stress"] = args.forecast_stress
        frame["forecast_stress_scale"] = args.forecast_stress_scale
        frame["forecast_stress_start_step"] = args.forecast_stress_start_step
        frame["forecast_stress_seed"] = args.forecast_stress_seed

    outputs: dict[str, tuple[str, pd.DataFrame]] = {}
    for name, frame in frames.items():
        path = os.path.join(args.out_dir, f"{name}_{args.tag}.csv")
        frame.to_csv(path, index=False)
        outputs[name] = (path, frame)
        print(f"saved -> {path}", flush=True)
    manifest_path = os.path.join(args.out_dir, f"manifest_{args.tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(
            build_manifest(args, starts_by_country, outputs),
            handle,
            indent=2,
            default=str,
        )
    print(f"saved -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
