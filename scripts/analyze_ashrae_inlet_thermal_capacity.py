"""Validate and analyze the near-capacity ASHRAE-inlet stress matrix."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from scripts.analyze_ashrae_inlet_pilot import (  # noqa: E402
    COST_METRICS,
    THERMAL_DIRECTIONS,
    WEIGHTS,
    _validate_identity,
)
from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    hierarchical_bootstrap_distribution,
    safe_wilcoxon,
)
from scripts.evaluate_eact_mpc import THERMAL_METRIC_SCHEMA  # noqa: E402


DEFAULT_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_thermal_capacity_stress_v1")
DEFAULT_OUT = os.path.join(DEFAULT_ROOT, "analysis")
COUNTRIES = {"JPN", "CHN", "NOR"}
ALGORITHMS = {
    "nominal_causal_mpc",
    "static_robust_mpc",
    "eact_mpc",
}
CONDITIONS = {
    ("none", 0.0),
    ("adverse_bias", 1.0),
}
AVAILABILITY_LEVELS = {0.50, 0.75, 1.00}
EXPECTED_MANIFESTS = (
    len(COUNTRIES) * len(CONDITIONS) * len(AVAILABILITY_LEVELS))
EXPECTED_CONTROLLER_WEEKS = (
    len(COUNTRIES)
    * 4
    * len(CONDITIONS)
    * len(AVAILABILITY_LEVELS)
    * len(ALGORITHMS)
)
OPERATIONAL_DIRECTIONS = {
    **{metric: "lower" for metric in COST_METRICS},
    "e_cooling_mwh": "lower",
    "e_pump_mwh": "lower",
    **THERMAL_DIRECTIONS,
}
RELATIVE_METRICS = {
    *COST_METRICS,
    "e_cooling_mwh",
    "e_pump_mwh",
}
DERATING_DIRECTIONS = {
    "recommended_exceedance_degc_h": "lower",
    "recommended_exceedance_hours": "lower",
    "recommended_compliance_pct": "higher",
    "p95_t_inlet_c": "lower",
    "p99_t_inlet_c": "lower",
    "max_t_inlet_c": "lower",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser


def _short_workspace_path(path: str) -> str:
    absolute = os.path.abspath(path)
    try:
        relative = os.path.relpath(absolute, os.getcwd())
    except ValueError:
        return absolute
    return relative if not relative.startswith("..") else absolute


def load_outputs(root: str, kind: str) -> pd.DataFrame:
    root = _short_workspace_path(root)
    paths = sorted(glob.glob(
        os.path.join(root, "**", f"{kind}_*.csv"), recursive=True))
    if len(paths) != EXPECTED_MANIFESTS:
        raise ValueError(
            f"Expected {EXPECTED_MANIFESTS} capacity {kind} files, "
            f"found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def load_manifests(root: str) -> list[dict]:
    root = _short_workspace_path(root)
    paths = sorted(glob.glob(
        os.path.join(root, "**", "manifest_*.json"), recursive=True))
    if len(paths) != EXPECTED_MANIFESTS:
        raise ValueError(
            f"Expected {EXPECTED_MANIFESTS} capacity manifests, "
            f"found {len(paths)}")
    manifests = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        item["_path"] = os.path.abspath(path)
        manifests.append(item)
    return manifests


def load_window_selection(root: str) -> dict:
    path = os.path.join(root, "high_load_window_selection.json")
    with open(path, encoding="utf-8") as handle:
        selection = json.load(handle)
    selection["_path"] = os.path.abspath(path)
    return selection


def _condition(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.forecast_stress.astype(str)
        + "_s"
        + frame.forecast_stress_scale.astype(float).map(
            lambda value: f"{value:.1f}"))


def _validate_window_selection(selection: dict) -> dict[str, set[int]]:
    if selection.get("status") != "complete":
        raise ValueError("High-load window selection is incomplete")
    if selection.get("episode_hours") != 168:
        raise ValueError("High-load selection must use 168-hour windows")
    windows = selection.get("windows", {})
    if set(windows) != COUNTRIES:
        raise ValueError("High-load selection country set is incomplete")
    starts = {}
    for country, items in windows.items():
        if len(items) != 4 or {item["quarter"] for item in items} != {
                "Q1", "Q2", "Q3", "Q4"}:
            raise ValueError(f"High-load quarter set is incomplete for {country}")
        country_starts = {int(item["start_hour"]) for item in items}
        if len(country_starts) != 4:
            raise ValueError(f"High-load starts are duplicated for {country}")
        starts[country] = country_starts
    return starts


def validate_matrix(
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
    solver: pd.DataFrame,
    manifests: list[dict],
    selection: dict,
) -> None:
    for frame in (episodes, hourly, solver):
        _validate_identity(frame)
        required = {
            "cooling_conductance_nominal_mw_per_k",
            "cooling_conductance_multiplier",
            "cooling_conductance_effective_mw_per_k",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Capacity identity fields are missing: {sorted(missing)}")
    if len(episodes) != EXPECTED_CONTROLLER_WEEKS:
        raise ValueError(
            f"Expected {EXPECTED_CONTROLLER_WEEKS} controller-weeks, "
            f"found {len(episodes)}")
    expected_rows = EXPECTED_CONTROLLER_WEEKS * 168
    if len(hourly) != expected_rows or len(solver) != expected_rows:
        raise ValueError("Capacity hourly or solver row count is incorrect")
    if set(episodes.country) != COUNTRIES:
        raise ValueError("Capacity country set is incomplete")
    if set(episodes.algorithm) != ALGORITHMS:
        raise ValueError("Capacity controller set is incomplete")
    conditions = set(zip(
        episodes.forecast_stress,
        episodes.forecast_stress_scale.astype(float),
    ))
    if conditions != CONDITIONS:
        raise ValueError("Capacity forecast condition set is incomplete")
    levels = set(np.round(
        episodes.cooling_conductance_multiplier.astype(float), 8))
    if levels != AVAILABILITY_LEVELS:
        raise ValueError("Capacity availability-level set is incomplete")
    if not np.allclose(
            episodes.cooling_conductance_nominal_mw_per_k.astype(float), 2.5):
        raise ValueError("Nominal cooling conductance is inconsistent")
    if not np.allclose(
            episodes.cooling_conductance_effective_mw_per_k.astype(float),
            2.5 * episodes.cooling_conductance_multiplier.astype(float)):
        raise ValueError("Effective cooling conductance is inconsistent")
    if not np.allclose(episodes.adaptive_beta_floor.astype(float), 0.10):
        raise ValueError("Capacity beta floor is incorrect")
    if set(episodes.thermal_safety_shield.astype(bool)) != {False}:
        raise ValueError("Capacity matrix must disable the safety shield")
    for column, value in WEIGHTS.items():
        if not np.allclose(episodes[column].astype(float), value):
            raise ValueError(f"Capacity objective mismatch: {column}")

    starts_by_country = _validate_window_selection(selection)
    for country, expected_starts in starts_by_country.items():
        actual = set(episodes.loc[
            episodes.country == country, "start_hour"].astype(int))
        if actual != expected_starts:
            raise ValueError(f"Capacity starts do not match selection for {country}")
    keyed = episodes.copy()
    keyed["condition"] = _condition(keyed)
    keys = [
        "condition",
        "cooling_conductance_multiplier",
        "country",
        "algorithm",
        "start_hour",
    ]
    if keyed.duplicated(keys).any():
        raise ValueError("Duplicate capacity controller-week key")
    counts = keyed.groupby([
        "condition", "cooling_conductance_multiplier"]).size()
    expected_per_cell = len(COUNTRIES) * 4 * len(ALGORITHMS)
    if len(counts) != len(CONDITIONS) * len(AVAILABILITY_LEVELS) or not np.all(
            counts.to_numpy() == expected_per_cell):
        raise ValueError("Capacity condition-level cells are incomplete")

    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get(
                "schema_version") != 2:
            raise ValueError(f"Incomplete capacity manifest: {manifest['_path']}")
        cfg = manifest.get("configuration", {})
        expected = {
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
            "adaptive_beta_floor": 0.10,
            "constraint_tolerance": 1e-4,
            "thermal_safety_shield": False,
            "cooling_conductance_nominal_mw_per_k": 2.5,
        }
        for key, value in expected.items():
            actual = cfg.get(key)
            valid = (
                np.isclose(float(actual), value)
                if isinstance(value, float) and actual is not None
                else actual == value
            )
            if not valid:
                raise ValueError(
                    f"Capacity manifest mismatch for {key}: "
                    f"{manifest['_path']}")
        multiplier = float(cfg.get("cooling_conductance_multiplier", np.nan))
        effective = float(
            cfg.get("cooling_conductance_effective_mw_per_k", np.nan))
        if (round(multiplier, 8) not in AVAILABILITY_LEVELS
                or not np.isclose(effective, 2.5 * multiplier)):
            raise ValueError(
                f"Capacity manifest conductance mismatch: {manifest['_path']}")
        expected_output_rows = {
            "episodes": 12,
            "weekly": 12,
            "hourly": 12 * 168,
            "solver": 12 * 168,
        }
        for name, count in expected_output_rows.items():
            if manifest["outputs"][name]["rows"] != count:
                raise ValueError(
                    f"Capacity manifest {name} count is incorrect")


def absolute_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes.copy()
    frame["condition"] = _condition(frame)
    frame["common_cost"] = -frame.episode_return.astype(float)
    rows = []
    for keys, group in frame.groupby([
            "condition", "cooling_conductance_multiplier", "algorithm"]):
        condition, availability, algorithm = keys
        rows.append({
            "condition": condition,
            "availability": float(availability),
            "algorithm": algorithm,
            "n_weeks": len(group),
            "common_cost_mean": float(group.common_cost.mean()),
            "e_total_mwh_mean": float(group.e_total_mwh.mean()),
            "e_grid_mwh_mean": float(group.e_grid_mwh.mean()),
            "co2_kg_mean": float(group.co2_kg.mean()),
            "e_cooling_mwh_mean": float(group.e_cooling_mwh.mean()),
            "e_pump_mwh_mean": float(group.e_pump_mwh.mean()),
            "recommended_exceedance_hours_total": float(
                group.recommended_exceedance_hours.sum()),
            "recommended_exceedance_degc_h_total": float(
                group.recommended_exceedance_degc_h.sum()),
            "recommended_compliance_pct_mean": float(
                group.recommended_compliance_pct.mean()),
            "p95_t_inlet_c_mean": float(group.p95_t_inlet_c.mean()),
            "p99_t_inlet_c_mean": float(group.p99_t_inlet_c.mean()),
            "max_t_inlet_c_max": float(group.max_t_inlet_c.max()),
            "allowable_exceedance_hours_total": float(
                group.allowable_exceedance_hours.sum()),
            "allowable_exceedance_degc_h_total": float(
                group.allowable_exceedance_degc_h.sum()),
            "sla_violation_mwh_total": float(group.sla_violation_mwh.sum()),
        })
    return pd.DataFrame(rows)


def controller_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260726,
) -> pd.DataFrame:
    frame = episodes.copy()
    if "condition" not in frame:
        frame["condition"] = _condition(frame)
    frame["common_cost"] = -frame.episode_return.astype(float)
    rows = []
    grouped = frame.groupby([
        "condition", "cooling_conductance_multiplier"])
    for group_index, ((condition, availability), group) in enumerate(grouped):
        for offset, (metric, direction) in enumerate(
                OPERATIONAL_DIRECTIONS.items()):
            keys = ["country", "start_hour"]
            eact = group.loc[
                group.algorithm == "eact_mpc", [*keys, metric]]
            static = group.loc[
                group.algorithm == "static_robust_mpc", [*keys, metric]]
            paired = eact.merge(
                static,
                on=keys,
                suffixes=("_eact", "_static"),
                validate="one_to_one",
            )
            if len(paired) != len(COUNTRIES) * 4:
                raise ValueError(
                    f"Incomplete capacity pairs for {condition}/"
                    f"{availability}/{metric}")
            eact_values = paired[f"{metric}_eact"].to_numpy(float)
            static_values = paired[f"{metric}_static"].to_numpy(float)
            paired["effect"] = (
                static_values - eact_values
                if direction == "lower"
                else eact_values - static_values
            )
            paired["season"] = paired.start_hour
            local_seed = seed + group_index * 100 + offset
            distribution = hierarchical_bootstrap_distribution(
                paired, "effect", samples=samples, seed=local_seed)
            low, high = np.quantile(distribution, [0.025, 0.975])
            sorted_effect = np.sort(paired.effect.to_numpy(float))
            worst_count = max(1, int(np.ceil(0.20 * len(sorted_effect))))
            row = {
                "condition": condition,
                "availability": float(availability),
                "metric": metric,
                "favorable_direction": direction,
                "n_pairs": len(paired),
                "mean_improvement": float(paired.effect.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "worst_case_improvement": float(sorted_effect[0]),
                "worst_20pct_mean_improvement": float(
                    sorted_effect[:worst_count].mean()),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.effect.to_numpy(float)),
            }
            if metric in RELATIVE_METRICS:
                if np.any(np.abs(static_values) <= 1e-12):
                    raise ValueError(f"Zero Static baseline for {metric}")
                relative = (
                    static_values - eact_values) / np.abs(static_values)
                paired["relative_effect"] = relative
                relative_distribution = hierarchical_bootstrap_distribution(
                    paired,
                    "relative_effect",
                    samples=samples,
                    seed=local_seed + 1000,
                )
                relative_low, relative_high = np.quantile(
                    relative_distribution, [0.025, 0.975])
                sorted_relative = np.sort(relative)
                row.update({
                    "mean_relative_improvement_pct": float(
                        100.0 * relative.mean()),
                    "relative_ci95_low_pct": float(100.0 * relative_low),
                    "relative_ci95_high_pct": float(100.0 * relative_high),
                    "worst_case_relative_improvement_pct": float(
                        100.0 * sorted_relative[0]),
                    "worst_20pct_mean_relative_improvement_pct": float(
                        100.0 * sorted_relative[:worst_count].mean()),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def derating_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260727,
) -> pd.DataFrame:
    frame = episodes.copy()
    if "condition" not in frame:
        frame["condition"] = _condition(frame)
    rows = []
    for group_index, ((condition, algorithm), group) in enumerate(
            frame.groupby(["condition", "algorithm"])):
        for offset, (metric, direction) in enumerate(
                DERATING_DIRECTIONS.items()):
            keys = ["country", "start_hour"]
            full = group.loc[
                np.isclose(group.cooling_conductance_multiplier, 1.0),
                [*keys, metric],
            ]
            near = group.loc[
                np.isclose(group.cooling_conductance_multiplier, 0.5),
                [*keys, metric],
            ]
            paired = near.merge(
                full,
                on=keys,
                suffixes=("_near_capacity", "_full_capacity"),
                validate="one_to_one",
            )
            if len(paired) != len(COUNTRIES) * 4:
                raise ValueError(
                    f"Incomplete derating pairs for {condition}/"
                    f"{algorithm}/{metric}")
            near_values = paired[
                f"{metric}_near_capacity"].to_numpy(float)
            full_values = paired[
                f"{metric}_full_capacity"].to_numpy(float)
            paired["effect"] = (
                near_values - full_values
                if direction == "lower"
                else full_values - near_values
            )
            paired["season"] = paired.start_hour
            distribution = hierarchical_bootstrap_distribution(
                paired,
                "effect",
                samples=samples,
                seed=seed + group_index * 100 + offset,
            )
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
                "condition": condition,
                "algorithm": algorithm,
                "metric": metric,
                "favorable_direction": direction,
                "comparison": "availability_0.50_minus_1.00",
                "n_pairs": len(paired),
                "mean_degradation": float(paired.effect.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.effect.to_numpy(float)),
            })
    return pd.DataFrame(rows)


def solver_summary(solver: pd.DataFrame) -> pd.DataFrame:
    frame = solver.copy()
    frame["condition"] = _condition(frame)
    rows = []
    for keys, group in frame.groupby([
            "condition", "cooling_conductance_multiplier", "algorithm"]):
        condition, availability, algorithm = keys
        fallbacks = Counter(group.fallback)
        accepted = group[group.accepted.astype(bool)]
        rows.append({
            "condition": condition,
            "availability": float(availability),
            "algorithm": algorithm,
            "n_candidate_plans": len(group),
            "n_accepted_plans": int(group.accepted.astype(bool).sum()),
            "n_rejected_plans": int((~group.accepted.astype(bool)).sum()),
            "accepted_plan_rate": float(group.accepted.astype(bool).mean()),
            "solver_success_rate": float(
                group.solver_success.astype(bool).mean()),
            "shifted_plan_fallbacks": int(fallbacks["shifted_plan"]),
            "safe_recovery_fallbacks": int(fallbacks["safe_recovery"]),
            "rule_based_fallbacks": int(fallbacks["rule_based"]),
            "accepted_min_constraint": (
                float(accepted.min_constraint.min())
                if len(accepted) else np.nan),
            "solve_time_mean_s": float(group.solve_time_s.mean()),
            "solve_time_p95_s": float(group.solve_time_s.quantile(0.95)),
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
    selection = load_window_selection(args.root)
    validate_matrix(episodes, hourly, solver, manifests, selection)

    comparisons = controller_comparisons(
        episodes, samples=args.bootstrap_samples)
    derating = derating_comparisons(
        episodes, samples=args.bootstrap_samples)
    outputs = {
        "capacity_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_manifests": len(manifests),
            "n_controller_weeks": len(episodes),
            "n_controller_hours": len(hourly),
            "n_solver_rows": len(solver),
            "n_capacity_levels": len(AVAILABILITY_LEVELS),
            "n_forecast_conditions": len(CONDITIONS),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "capacity_absolute_summary.csv": absolute_summary(episodes),
        "capacity_paired_comparisons.csv": comparisons,
        "capacity_core_summary.csv": comparisons[
            comparisons.metric.isin([
                "common_cost",
                "e_total_mwh",
                "recommended_exceedance_hours",
                "recommended_exceedance_degc_h",
                "p95_t_inlet_c",
                "p99_t_inlet_c",
                "allowable_exceedance_hours",
            ])
        ].copy(),
        "capacity_derating_comparisons.csv": derating,
        "capacity_solver_summary.csv": solver_summary(solver),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    output_records = {}
    for name, frame in outputs.items():
        path = os.path.join(args.out_dir, name)
        frame.to_csv(path, index=False)
        output_records[name] = {
            "path": os.path.abspath(path),
            "rows": len(frame),
        }
        print(f"saved -> {path}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": os.path.abspath(args.root),
        "window_selection": selection["_path"],
        "bootstrap_samples": args.bootstrap_samples,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [item["_path"] for item in manifests],
        "outputs": output_records,
    }
    path = os.path.join(args.out_dir, "capacity_analysis_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
