"""Validate and analyze ASHRAE-inlet objective-weight sensitivity."""

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
    _validate_identity,
)
from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    hierarchical_bootstrap_distribution,
    safe_wilcoxon,
)
from scripts.evaluate_eact_mpc import THERMAL_METRIC_SCHEMA  # noqa: E402
from scripts.run_eact_weight_sensitivity import (  # noqa: E402
    COUNTRIES,
    WEIGHT_SETTINGS,
)


DEFAULT_WEIGHT_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_weight_sensitivity_v1")
DEFAULT_SEASONAL_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_seasonal_v1")
DEFAULT_OUT = os.path.join(DEFAULT_WEIGHT_ROOT, "analysis")
SCENARIOS = {"none", "adverse_bias"}
ALGORITHMS = {"static_robust_mpc", "eact_mpc"}
NEW_SETTINGS = {
    setting.name for setting in WEIGHT_SETTINGS if setting.name != "primary"}
SETTING_BY_WEIGHTS = {
    (setting.grid, setting.co2, setting.total, setting.smooth): setting.name
    for setting in WEIGHT_SETTINGS
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-root", default=DEFAULT_WEIGHT_ROOT)
    parser.add_argument("--seasonal-root", default=DEFAULT_SEASONAL_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser


def _load_new_outputs(root: str, kind: str) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "*", f"{kind}_*.csv")))
    expected = len(NEW_SETTINGS) * len(SCENARIOS) * len(COUNTRIES)
    if len(paths) != expected:
        raise ValueError(
            f"Expected {expected} weight {kind} files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _load_manifests(root: str) -> list[dict]:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "*", "manifest_*.json")))
    expected = len(NEW_SETTINGS) * len(SCENARIOS) * len(COUNTRIES)
    if len(paths) != expected:
        raise ValueError(
            f"Expected {expected} weight manifests, found {len(paths)}")
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        item["_path"] = os.path.abspath(path)
        items.append(item)
    return items


def _load_primary(root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "episodes_*.csv")))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 seasonal episode files, found {len(paths)}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    return frame[
        frame.country.isin(COUNTRIES)
        & frame.algorithm.isin(ALGORITHMS)
    ].copy()


def _setting_name(row: pd.Series) -> str:
    weights = tuple(round(float(row[column]), 12) for column in (
        "weight_grid", "weight_co2", "weight_total", "weight_smooth"))
    if weights not in SETTING_BY_WEIGHTS:
        raise ValueError(f"Unknown objective-weight tuple: {weights}")
    return SETTING_BY_WEIGHTS[weights]


def _add_setting(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["setting"] = result.apply(_setting_name, axis=1)
    return result


def _validate_new_matrix(
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
    solver: pd.DataFrame,
    manifests: list[dict],
) -> None:
    for frame in (episodes, hourly, solver):
        _validate_identity(frame)
    expected_weeks = (
        len(NEW_SETTINGS) * len(SCENARIOS) * len(COUNTRIES)
        * len(ALGORITHMS) * len(START_HOURS)
    )
    if len(episodes) != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} new weight weeks, found {len(episodes)}")
    if len(hourly) != expected_weeks * 168:
        raise ValueError("Weight hourly row count is incorrect")
    if len(solver) != expected_weeks * 168:
        raise ValueError("Weight solver row count is incorrect")
    episodes = _add_setting(episodes)
    if set(episodes.setting) != NEW_SETTINGS:
        raise ValueError("Weight setting set is incomplete")
    if set(episodes.country) != set(COUNTRIES):
        raise ValueError("Weight country set is incomplete")
    if set(episodes.algorithm) != ALGORITHMS:
        raise ValueError("Weight controller set is incomplete")
    if set(episodes.forecast_stress) != SCENARIOS:
        raise ValueError("Weight scenario set is incomplete")
    if set(episodes.start_hour.astype(int)) != START_HOURS:
        raise ValueError("Weight seasonal starts are incorrect")
    if not np.allclose(episodes.adaptive_beta_floor.astype(float), 0.10):
        raise ValueError("Weight matrix beta floor is incorrect")
    if set(episodes.thermal_safety_shield.astype(bool)) != {False}:
        raise ValueError("Weight matrix must disable the safety shield")
    keys = ["setting", "forecast_stress", "country", "algorithm", "start_hour"]
    if episodes.duplicated(keys).any():
        raise ValueError("Duplicate weight controller-week key")
    counts = episodes.groupby(["setting", "forecast_stress"]).size()
    expected_group = len(COUNTRIES) * len(ALGORITHMS) * len(START_HOURS)
    if not np.all(counts.to_numpy() == expected_group):
        raise ValueError("Weight setting-scenario matrix is incomplete")
    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get(
                "schema_version") != 2:
            raise ValueError(f"Incomplete weight manifest: {manifest['_path']}")
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
                    f"Weight manifest mismatch for {key}: "
                    f"{manifest['_path']}")
        if manifest["outputs"]["episodes"]["rows"] != (
                len(ALGORITHMS) * len(START_HOURS)):
            raise ValueError("Weight manifest episode count is incorrect")


def _combine_with_primary(
    new_episodes: pd.DataFrame,
    primary: pd.DataFrame,
) -> pd.DataFrame:
    _validate_identity(primary)
    primary = _add_setting(primary)
    if set(primary.setting) != {"primary"} or len(primary) != 48:
        raise ValueError("Primary-weight reuse matrix is incorrect")
    combined = pd.concat(
        [_add_setting(new_episodes), primary], ignore_index=True)
    keys = ["setting", "forecast_stress", "country", "algorithm", "start_hour"]
    if combined.duplicated(keys).any() or len(combined) != 240:
        raise ValueError("Combined five-setting matrix is incomplete")
    return combined


def weight_comparisons(
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
    group_columns = ["setting", "forecast_stress"]
    for group_index, ((setting, scenario), group) in enumerate(
            frame.groupby(group_columns)):
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
                raise ValueError(
                    f"Incomplete pairs for {setting}/{scenario}/{metric}")
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
            row = {
                "setting": setting,
                "scenario": scenario,
                "metric": metric,
                "favorable_direction": direction,
                "n_pairs": len(paired),
                "mean_improvement": float(paired.effect.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
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
                row.update({
                    "mean_relative_improvement_pct": float(
                        100.0 * relative.mean()),
                    "relative_ci95_low_pct": float(100.0 * relative_low),
                    "relative_ci95_high_pct": float(100.0 * relative_high),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def absolute_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = episodes.copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    metrics = [
        "common_cost", "e_grid_mwh", "co2_kg", "e_total_mwh",
        "recommended_exceedance_degc_h", "recommended_exceedance_hours",
        "recommended_compliance_pct", "p95_t_inlet_c", "p99_t_inlet_c",
        "max_t_inlet_c", "allowable_exceedance_hours",
    ]
    return frame.groupby(
        ["setting", "forecast_stress", "algorithm"],
        as_index=False,
    )[metrics].mean()


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    new_episodes = _load_new_outputs(args.weight_root, "episodes")
    hourly = _load_new_outputs(args.weight_root, "hourly")
    solver = _load_new_outputs(args.weight_root, "solver")
    manifests = _load_manifests(args.weight_root)
    _validate_new_matrix(new_episodes, hourly, solver, manifests)
    combined = _combine_with_primary(
        new_episodes, _load_primary(args.seasonal_root))

    outputs = {
        "weight_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_new_manifests": len(manifests),
            "n_new_controller_weeks": len(new_episodes),
            "n_reused_primary_weeks": 48,
            "n_total_controller_weeks": len(combined),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "weight_absolute_summary.csv": absolute_summary(combined),
        "weight_paired_comparisons.csv": weight_comparisons(
            combined, samples=args.bootstrap_samples),
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
        "weight_root": os.path.abspath(args.weight_root),
        "seasonal_root": os.path.abspath(args.seasonal_root),
        "bootstrap_samples": args.bootstrap_samples,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [item["_path"] for item in manifests],
        "outputs": output_records,
    }
    path = os.path.join(args.out_dir, "weight_analysis_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
