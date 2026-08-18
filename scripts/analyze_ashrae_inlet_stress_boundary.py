"""Validate and analyze the ASHRAE-inlet forecast-stress boundary."""

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

from scripts.analyze_ashrae_inlet_pilot import (  # noqa: E402
    COST_METRICS,
    START_HOURS,
    THERMAL_DIRECTIONS,
    WEIGHTS,
    _validate_identity,
)
from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    hierarchical_bootstrap_distribution,
    safe_wilcoxon,
)
from scripts.evaluate_eact_mpc import THERMAL_METRIC_SCHEMA  # noqa: E402


DEFAULT_STRESS_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_stress_boundary_v1")
DEFAULT_SEASONAL_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_seasonal_v1")
DEFAULT_OUT = os.path.join(DEFAULT_STRESS_ROOT, "analysis")
COUNTRIES = {"JPN", "CHN", "NOR"}
ALGORITHMS = {"static_robust_mpc", "eact_mpc"}
NEW_CONDITIONS = {
    ("adverse_bias", 0.5),
    ("adverse_bias", 2.0),
    ("noise", 1.0),
    ("combined", 1.0),
}
ALL_CONDITIONS = {
    ("none", 0.0),
    ("adverse_bias", 0.5),
    ("adverse_bias", 1.0),
    ("adverse_bias", 2.0),
    ("noise", 1.0),
    ("combined", 1.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-root", default=DEFAULT_STRESS_ROOT)
    parser.add_argument("--seasonal-root", default=DEFAULT_SEASONAL_ROOT)
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


def _load_new_outputs(root: str, kind: str) -> pd.DataFrame:
    root = _short_workspace_path(root)
    paths = sorted(glob.glob(
        os.path.join(root, "**", f"{kind}_*.csv"), recursive=True))
    if len(paths) != 12:
        raise ValueError(
            f"Expected 12 stress {kind} files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _load_manifests(root: str) -> list[dict]:
    root = _short_workspace_path(root)
    paths = sorted(glob.glob(
        os.path.join(root, "**", "manifest_*.json"), recursive=True))
    if len(paths) != 12:
        raise ValueError(f"Expected 12 stress manifests, found {len(paths)}")
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        item["_path"] = os.path.abspath(path)
        items.append(item)
    return items


def _load_primary(root: str) -> pd.DataFrame:
    root = _short_workspace_path(root)
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "episodes_*.csv")))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 seasonal episode files, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return frame[
        frame.country.isin(COUNTRIES)
        & frame.algorithm.isin(ALGORITHMS)
    ].copy()


def _condition(frame: pd.DataFrame) -> pd.Series:
    return (
        frame.forecast_stress.astype(str)
        + "_s"
        + frame.forecast_stress_scale.astype(float).map(
            lambda value: f"{value:.1f}"))


def _validate_new(
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
    solver: pd.DataFrame,
    manifests: list[dict],
) -> None:
    for frame in (episodes, hourly, solver):
        _validate_identity(frame)
    if len(episodes) != 96:
        raise ValueError(f"Expected 96 new stress weeks, found {len(episodes)}")
    if len(hourly) != 96 * 168 or len(solver) != 96 * 168:
        raise ValueError("Stress hourly or solver row count is incorrect")
    if set(episodes.country) != COUNTRIES:
        raise ValueError("Stress country set is incomplete")
    if set(episodes.algorithm) != ALGORITHMS:
        raise ValueError("Stress controller set is incomplete")
    conditions = set(zip(
        episodes.forecast_stress,
        episodes.forecast_stress_scale.astype(float),
    ))
    if conditions != NEW_CONDITIONS:
        raise ValueError("New stress condition set is incomplete")
    if set(episodes.start_hour.astype(int)) != START_HOURS:
        raise ValueError("Stress seasonal starts are incorrect")
    if not np.allclose(episodes.adaptive_beta_floor.astype(float), 0.10):
        raise ValueError("Stress beta floor is incorrect")
    if set(episodes.thermal_safety_shield.astype(bool)) != {False}:
        raise ValueError("Stress matrix must disable the safety shield")
    for column, value in WEIGHTS.items():
        if not np.allclose(episodes[column].astype(float), value):
            raise ValueError(f"Stress objective mismatch: {column}")
    episodes = episodes.copy()
    episodes["condition"] = _condition(episodes)
    keys = ["condition", "country", "algorithm", "start_hour"]
    if episodes.duplicated(keys).any():
        raise ValueError("Duplicate stress controller-week key")
    counts = episodes.groupby("condition").size()
    if len(counts) != len(NEW_CONDITIONS) or not np.all(
            counts.to_numpy() == len(COUNTRIES) * len(ALGORITHMS) * len(START_HOURS)):
        raise ValueError("Stress condition matrices are incomplete")
    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get(
                "schema_version") != 2:
            raise ValueError(f"Incomplete stress manifest: {manifest['_path']}")
        cfg = manifest.get("configuration", {})
        expected = {
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
            "adaptive_beta_floor": 0.10,
            "constraint_tolerance": 1e-4,
            "thermal_safety_shield": False,
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
                    f"Stress manifest mismatch for {key}: "
                    f"{manifest['_path']}")
        if manifest["outputs"]["episodes"]["rows"] != 8:
            raise ValueError("Stress manifest episode count is incorrect")


def _combine(new: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    _validate_identity(primary)
    if len(primary) != 48:
        raise ValueError("Primary stress-reuse matrix is incorrect")
    combined = pd.concat([new, primary], ignore_index=True)
    combined["condition"] = _condition(combined)
    conditions = set(zip(
        combined.forecast_stress,
        combined.forecast_stress_scale.astype(float),
    ))
    if conditions != ALL_CONDITIONS or len(combined) != 144:
        raise ValueError("Combined stress-boundary matrix is incomplete")
    keys = ["condition", "country", "algorithm", "start_hour"]
    if combined.duplicated(keys).any():
        raise ValueError("Duplicate combined stress key")
    return combined


def stress_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260725,
) -> pd.DataFrame:
    frame = episodes.copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    metrics = {
        **{metric: "lower" for metric in COST_METRICS},
        **THERMAL_DIRECTIONS,
    }
    rows = []
    for condition_index, (condition, group) in enumerate(
            frame.groupby("condition")):
        for offset, (metric, direction) in enumerate(metrics.items()):
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
            if len(paired) != len(COUNTRIES) * len(START_HOURS):
                raise ValueError(f"Incomplete stress pairs for {condition}/{metric}")
            eact_values = paired[f"{metric}_eact"].to_numpy(float)
            static_values = paired[f"{metric}_static"].to_numpy(float)
            paired["effect"] = (
                static_values - eact_values
                if direction == "lower"
                else eact_values - static_values
            )
            paired["season"] = paired.start_hour
            local_seed = seed + condition_index * 100 + offset
            distribution = hierarchical_bootstrap_distribution(
                paired, "effect", samples=samples, seed=local_seed)
            low, high = np.quantile(distribution, [0.025, 0.975])
            sorted_effect = np.sort(paired.effect.to_numpy(float))
            worst_count = max(1, int(np.ceil(0.20 * len(sorted_effect))))
            row = {
                "condition": condition,
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
                    paired.effect.to_numpy(dtype=float)),
            }
            if metric in COST_METRICS:
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


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    new = _load_new_outputs(args.stress_root, "episodes")
    hourly = _load_new_outputs(args.stress_root, "hourly")
    solver = _load_new_outputs(args.stress_root, "solver")
    manifests = _load_manifests(args.stress_root)
    _validate_new(new, hourly, solver, manifests)
    combined = _combine(new, _load_primary(args.seasonal_root))
    comparisons = stress_comparisons(
        combined, samples=args.bootstrap_samples)

    outputs = {
        "stress_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_new_manifests": len(manifests),
            "n_new_controller_weeks": len(new),
            "n_reused_primary_weeks": 48,
            "n_total_controller_weeks": len(combined),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "stress_paired_comparisons.csv": comparisons,
        "stress_core_summary.csv": comparisons[
            comparisons.metric.isin(
                ["common_cost", "p95_t_inlet_c", "p99_t_inlet_c"])
        ].copy(),
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
        "stress_root": os.path.abspath(args.stress_root),
        "seasonal_root": os.path.abspath(args.seasonal_root),
        "bootstrap_samples": args.bootstrap_samples,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [item["_path"] for item in manifests],
        "outputs": output_records,
    }
    path = os.path.join(args.out_dir, "stress_analysis_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
