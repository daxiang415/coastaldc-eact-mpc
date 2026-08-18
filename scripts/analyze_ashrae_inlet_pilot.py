"""Strict validation and paired analysis for the ASHRAE inlet pilot."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from scripts.analyze_eact_paper_evidence import (
    hierarchical_bootstrap_distribution,
    safe_wilcoxon,
)
from scripts.evaluate_eact_mpc import (
    THERMAL_METRIC_SCHEMA,
    THERMAL_STATE_SEMANTICS,
)


DEFAULT_ROOT = os.path.join(ROOT, "results", "ashrae_inlet_pilot_v1")
DEFAULT_OUT = os.path.join(DEFAULT_ROOT, "analysis")
COUNTRIES = {"CHN", "JPN", "NOR"}
ALGORITHMS = {
    "nominal_causal_mpc",
    "static_robust_mpc",
    "eact_mpc",
}
SCENARIOS = {("none", 0.0), ("adverse_bias", 1.0)}
START_HOURS = {336, 2496, 4680, 6888}
WEIGHTS = {
    "weight_grid": 1.0,
    "weight_co2": 2.0,
    "weight_total": 0.2,
    "weight_smooth": 0.5,
}
COST_METRICS = ("common_cost", "e_grid_mwh", "co2_kg", "e_total_mwh")
THERMAL_DIRECTIONS = {
    "recommended_exceedance_degc_h": "lower",
    "recommended_exceedance_hours": "lower",
    "recommended_compliance_pct": "higher",
    "p95_t_inlet_c": "lower",
    "p99_t_inlet_c": "lower",
    "max_t_inlet_c": "lower",
    "min_allowable_headroom_c": "higher",
    "allowable_exceedance_hours": "lower",
    "allowable_exceedance_degc_h": "lower",
    "temporal_rci_hi_pct": "higher",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser


def load_outputs(
    root: str,
    kind: str,
    *,
    expected_files: int = 6,
) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", f"{kind}_*.csv")))
    if len(paths) != expected_files:
        raise ValueError(
            f"Expected {expected_files} {kind} files, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.abspath(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_manifests(root: str, *, expected_files: int = 6) -> list[dict]:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "manifest_*.json")))
    if len(paths) != expected_files:
        raise ValueError(
            f"Expected {expected_files} manifests, found {len(paths)}")
    manifests = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["_path"] = os.path.abspath(path)
        manifests.append(manifest)
    return manifests


def _validate_identity(frame: pd.DataFrame) -> None:
    required = {
        "thermal_metric_schema",
        "thermal_state_semantics",
        "t_inlet_recommended_min_c",
        "t_inlet_recommended_max_c",
        "t_inlet_allowable_min_c",
        "t_inlet_allowable_max_c",
        "constraint_tolerance",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Thermal identity fields are missing: {missing}")
    if set(frame.thermal_metric_schema) != {THERMAL_METRIC_SCHEMA}:
        raise ValueError("Mixed or incorrect thermal metric schema")
    if set(frame.thermal_state_semantics) != {THERMAL_STATE_SEMANTICS}:
        raise ValueError("Mixed or incorrect thermal state semantics")
    expected = {
        "t_inlet_recommended_min_c": 18.0,
        "t_inlet_recommended_max_c": 27.0,
        "t_inlet_allowable_min_c": 15.0,
        "t_inlet_allowable_max_c": 32.0,
        "constraint_tolerance": 1e-4,
    }
    for column, value in expected.items():
        if not np.allclose(frame[column].astype(float), value):
            raise ValueError(f"Incorrect or mixed thermal identity: {column}")


def validate_episode_matrix(
    episodes: pd.DataFrame,
    *,
    expected_countries: set[str] = COUNTRIES,
) -> None:
    required = {
        "country", "algorithm", "start_hour", "episode_return",
        "forecast_stress", "forecast_stress_scale",
        "thermal_safety_shield", "adaptive_beta_floor",
        *WEIGHTS,
        *THERMAL_DIRECTIONS,
    }
    missing = sorted(required.difference(episodes.columns))
    if missing:
        raise ValueError(f"Pilot episode fields are missing: {missing}")
    _validate_identity(episodes)
    expected_weeks = (
        len(expected_countries)
        * len(ALGORITHMS)
        * len(SCENARIOS)
        * len(START_HOURS)
    )
    if len(episodes) != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} controller-week episodes, "
            f"found {len(episodes)}")
    if set(episodes.country) != expected_countries:
        raise ValueError("Country set is incomplete")
    if set(episodes.algorithm) != ALGORITHMS:
        raise ValueError("Pilot controller set is incomplete")
    conditions = set(zip(
        episodes.forecast_stress,
        episodes.forecast_stress_scale.astype(float),
    ))
    if conditions != SCENARIOS:
        raise ValueError("Pilot scenario set is incomplete or mixed")
    if set(episodes.start_hour.astype(int)) != START_HOURS:
        raise ValueError("Pilot seasonal start set is incorrect")
    if set(episodes.thermal_safety_shield.astype(bool)) != {False}:
        raise ValueError("Pilot must disable the external thermal safety shield")
    for column, value in WEIGHTS.items():
        if not np.allclose(episodes[column].astype(float), value):
            raise ValueError(f"Objective configuration mismatch: {column}")
    eact = episodes[episodes.algorithm == "eact_mpc"]
    if not np.allclose(eact.adaptive_beta_floor.astype(float), 0.10):
        raise ValueError("EACT beta-floor configuration mismatch")
    keys = ["country", "algorithm", "forecast_stress", "start_hour"]
    if episodes.duplicated(keys).any():
        raise ValueError("Duplicate pilot controller-week key")
    counts = episodes.groupby(
        ["country", "algorithm", "forecast_stress"]).size()
    expected_groups = (
        len(expected_countries) * len(ALGORITHMS) * len(SCENARIOS))
    if len(counts) != expected_groups or not np.all(
            counts.to_numpy() == len(START_HOURS)):
        raise ValueError("Each country-controller-scenario requires four weeks")
    numeric = [
        "episode_return", "e_grid_mwh", "co2_kg", "e_total_mwh",
        *THERMAL_DIRECTIONS,
    ]
    if not np.isfinite(episodes[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Pilot episode outputs contain NaN or infinite values")


def validate_hourly_matrix(
    hourly: pd.DataFrame,
    *,
    expected_controller_weeks: int = 72,
) -> None:
    required = {
        "country", "algorithm", "episode", "step", "forecast_stress",
        "t_inlet_c", "recommended_exceedance_c", "allowable_exceedance_c",
        "recommended_exceedance_event", "allowable_exceedance_event",
        "requested_workload", "requested_setpoint", "requested_pump",
        "applied_workload", "applied_setpoint", "applied_pump",
    }
    missing = sorted(required.difference(hourly.columns))
    if missing:
        raise ValueError(f"Pilot hourly fields are missing: {missing}")
    _validate_identity(hourly)
    expected_rows = expected_controller_weeks * 168
    if len(hourly) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} controller-hour rows, "
            f"found {len(hourly)}")
    keys = ["country", "algorithm", "forecast_stress", "episode", "step"]
    if hourly.duplicated(keys).any():
        raise ValueError("Duplicate pilot controller-hour key")
    numeric = [
        "t_inlet_c", "recommended_exceedance_c", "allowable_exceedance_c",
        "requested_workload", "requested_setpoint", "requested_pump",
        "applied_workload", "applied_setpoint", "applied_pump",
    ]
    if not np.isfinite(hourly[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Pilot hourly outputs contain NaN or infinite values")
    for prefix in ("requested", "applied"):
        if not hourly[f"{prefix}_workload"].between(-1.0, 1.0).all():
            raise ValueError(f"{prefix} workload actions are out of bounds")
        if not hourly[f"{prefix}_setpoint"].between(-1.0, 1.0).all():
            raise ValueError(f"{prefix} setpoint actions are out of bounds")
        if not hourly[f"{prefix}_pump"].between(0.0, 1.0).all():
            raise ValueError(f"{prefix} pump actions are out of bounds")


def validate_solver_matrix(
    solver: pd.DataFrame,
    *,
    expected_controller_weeks: int = 72,
) -> None:
    required = {
        "country", "algorithm", "episode", "step", "forecast_stress",
        "accepted", "solver_success", "fallback", "solve_time_s",
        "min_constraint",
    }
    missing = sorted(required.difference(solver.columns))
    if missing:
        raise ValueError(f"Pilot solver fields are missing: {missing}")
    _validate_identity(solver)
    expected_rows = expected_controller_weeks * 168
    if len(solver) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} solver rows, found {len(solver)}")
    keys = ["country", "algorithm", "forecast_stress", "episode", "step"]
    if solver.duplicated(keys).any():
        raise ValueError("Duplicate pilot solver key")
    if not np.isfinite(
            solver[["solve_time_s", "min_constraint"]].to_numpy(float)).all():
        raise ValueError("Pilot solver outputs contain nonfinite values")
    allowed_fallbacks = {"none", "safe_recovery", "shifted_plan", "rule_based"}
    if not set(solver.fallback).issubset(allowed_fallbacks):
        raise ValueError("Unknown solver fallback label")


def validate_manifests(manifests: list[dict]) -> None:
    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get("schema_version") != 2:
            raise ValueError(f"Incomplete manifest: {manifest['_path']}")
        cfg = manifest.get("configuration", {})
        expected = {
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
            "thermal_state_semantics": THERMAL_STATE_SEMANTICS,
            "t_inlet_recommended_min_c": 18.0,
            "t_inlet_recommended_max_c": 27.0,
            "t_inlet_allowable_min_c": 15.0,
            "t_inlet_allowable_max_c": 32.0,
            "constraint_tolerance": 1e-4,
            "thermal_safety_shield": False,
        }
        for key, value in expected.items():
            actual = cfg.get(key)
            if isinstance(value, float):
                valid = actual is not None and np.isclose(float(actual), value)
            else:
                valid = actual == value
            if not valid:
                raise ValueError(
                    f"Manifest thermal identity mismatch for {key}: "
                    f"{manifest['_path']}")
        outputs = manifest.get("outputs", {})
        if set(outputs) != {"summary", "episodes", "weekly", "hourly", "solver"}:
            raise ValueError(f"Manifest output set is incomplete: {manifest['_path']}")
        if outputs["episodes"]["rows"] != 12:
            raise ValueError(f"Manifest episode count is incorrect: {manifest['_path']}")
        if outputs["hourly"]["rows"] != 12 * 168:
            raise ValueError(f"Manifest hourly count is incorrect: {manifest['_path']}")


def _paired(
    frame: pd.DataFrame,
    treatment: str,
    baseline: str,
    metric: str,
) -> pd.DataFrame:
    keys = ["country", "start_hour"]
    treatment_rows = frame.loc[
        frame.algorithm == treatment, [*keys, metric]]
    baseline_rows = frame.loc[
        frame.algorithm == baseline, [*keys, metric]]
    paired = treatment_rows.merge(
        baseline_rows,
        on=keys,
        suffixes=("_treatment", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(treatment_rows) or len(paired) != len(baseline_rows):
        raise ValueError(f"Incomplete pairing for {metric}")
    paired["season"] = paired.start_hour
    return paired


def cost_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260725,
) -> pd.DataFrame:
    frame = episodes.copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    rows = []
    for scenario, scenario_rows in frame.groupby("forecast_stress"):
        for baseline in ("nominal_causal_mpc", "static_robust_mpc"):
            for offset, metric in enumerate(COST_METRICS):
                paired = _paired(scenario_rows, "eact_mpc", baseline, metric)
                baseline_values = paired[
                    f"{metric}_baseline"].to_numpy(dtype=float)
                if np.any(np.abs(baseline_values) <= 1e-12):
                    raise ValueError(f"Zero baseline value for {metric}")
                paired["effect"] = (
                    baseline_values
                    - paired[f"{metric}_treatment"].to_numpy(dtype=float)
                ) / np.abs(baseline_values)
                distribution = hierarchical_bootstrap_distribution(
                    paired, "effect", samples=samples, seed=seed + offset)
                low, high = np.quantile(distribution, [0.025, 0.975])
                rows.append({
                    "scenario": scenario,
                    "treatment": "eact_mpc",
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_relative_improvement_pct": float(
                        100.0 * paired.effect.mean()),
                    "ci95_low_pct": float(100.0 * low),
                    "ci95_high_pct": float(100.0 * high),
                    "wilcoxon_p_value": safe_wilcoxon(
                        paired.effect.to_numpy(dtype=float)),
                })
    return pd.DataFrame(rows)


def thermal_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260725,
) -> pd.DataFrame:
    rows = []
    for scenario, scenario_rows in episodes.groupby("forecast_stress"):
        for baseline in ("nominal_causal_mpc", "static_robust_mpc"):
            for offset, (metric, direction) in enumerate(
                    THERMAL_DIRECTIONS.items()):
                paired = _paired(scenario_rows, "eact_mpc", baseline, metric)
                treatment = paired[f"{metric}_treatment"].to_numpy(float)
                baseline_values = paired[f"{metric}_baseline"].to_numpy(float)
                paired["effect"] = (
                    baseline_values - treatment
                    if direction == "lower"
                    else treatment - baseline_values
                )
                distribution = hierarchical_bootstrap_distribution(
                    paired, "effect", samples=samples, seed=seed + offset)
                low, high = np.quantile(distribution, [0.025, 0.975])
                rows.append({
                    "scenario": scenario,
                    "treatment": "eact_mpc",
                    "baseline": baseline,
                    "metric": metric,
                    "favorable_direction": direction,
                    "n_pairs": len(paired),
                    "mean_improvement": float(paired.effect.mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "wilcoxon_p_value": safe_wilcoxon(
                        paired.effect.to_numpy(dtype=float)),
                })
    return pd.DataFrame(rows)


def solver_summary(solver: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, algorithm), group in solver.groupby(
            ["forecast_stress", "algorithm"]):
        accepted = group[group.accepted.astype(bool)]
        fallback_counts = group.fallback.value_counts()
        rows.append({
            "scenario": scenario,
            "algorithm": algorithm,
            "n_control_steps": len(group),
            "accepted_plan_rate": float(group.accepted.astype(bool).mean()),
            "rejected_plan_count": int(
                (~group.accepted.astype(bool)).sum()),
            "solver_convergence_rate": float(
                group.solver_success.astype(bool).mean()),
            "fallback_rate": float((group.fallback != "none").mean()),
            "shifted_plan_fallbacks": int(
                fallback_counts.get("shifted_plan", 0)),
            "safe_recovery_fallbacks": int(
                fallback_counts.get("safe_recovery", 0)),
            "rule_based_fallbacks": int(
                fallback_counts.get("rule_based", 0)),
            "mean_solve_time_s": float(group.solve_time_s.mean()),
            "p95_solve_time_s": float(group.solve_time_s.quantile(0.95)),
            "min_selected_candidate_constraint": float(
                group.min_constraint.min()),
            "min_accepted_constraint": float(
                accepted.min_constraint.min()) if len(accepted) else np.nan,
        })
    return pd.DataFrame(rows)


def absolute_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes.copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    rows = []
    for (scenario, algorithm), group in frame.groupby(
            ["forecast_stress", "algorithm"]):
        rows.append({
            "scenario": scenario,
            "algorithm": algorithm,
            "n_controller_weeks": len(group),
            "mean_common_cost": float(group.common_cost.mean()),
            "mean_e_grid_mwh": float(group.e_grid_mwh.mean()),
            "mean_co2_kg": float(group.co2_kg.mean()),
            "mean_e_total_mwh": float(group.e_total_mwh.mean()),
            "total_sla_violation_mwh": float(
                group.sla_violation_mwh.sum()),
            "total_recommended_exceedance_degc_h": float(
                group.recommended_exceedance_degc_h.sum()),
            "total_recommended_exceedance_hours": float(
                group.recommended_exceedance_hours.sum()),
            "mean_recommended_compliance_pct": float(
                group.recommended_compliance_pct.mean()),
            "mean_p95_t_inlet_c": float(group.p95_t_inlet_c.mean()),
            "mean_p99_t_inlet_c": float(group.p99_t_inlet_c.mean()),
            "mean_max_t_inlet_c": float(group.max_t_inlet_c.mean()),
            "overall_max_t_inlet_c": float(group.max_t_inlet_c.max()),
            "mean_p95_recommended_headroom_c": float(
                27.0 - group.p95_t_inlet_c.mean()),
            "total_allowable_exceedance_hours": float(
                group.allowable_exceedance_hours.sum()),
            "total_allowable_exceedance_degc_h": float(
                group.allowable_exceedance_degc_h.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    episodes = load_outputs(args.root, "episodes")
    hourly = load_outputs(args.root, "hourly")
    solver = load_outputs(args.root, "solver")
    manifests = load_manifests(args.root)

    validate_episode_matrix(episodes)
    validate_hourly_matrix(hourly)
    validate_solver_matrix(solver)
    validate_manifests(manifests)

    outputs = {
        "pilot_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_manifests": len(manifests),
            "n_controller_weeks": len(episodes),
            "n_controller_hours": len(hourly),
            "n_solver_rows": len(solver),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "pilot_absolute_summary.csv": absolute_summary(episodes),
        "pilot_cost_comparisons.csv": cost_comparisons(
            episodes, samples=args.bootstrap_samples),
        "pilot_thermal_comparisons.csv": thermal_comparisons(
            episodes, samples=args.bootstrap_samples),
        "pilot_solver_summary.csv": solver_summary(solver),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    output_records = {}
    for name, frame in outputs.items():
        path = os.path.join(args.out_dir, name)
        frame.to_csv(path, index=False)
        output_records[name] = {"path": os.path.abspath(path), "rows": len(frame)}
        print(f"saved -> {path}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_root": os.path.abspath(args.root),
        "bootstrap_samples": args.bootstrap_samples,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [manifest["_path"] for manifest in manifests],
        "outputs": output_records,
    }
    manifest_path = os.path.join(args.out_dir, "pilot_analysis_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {manifest_path}")


if __name__ == "__main__":
    main()
