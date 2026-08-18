"""Validate and analyze continuous annual ASHRAE-inlet MPC trajectories."""

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
    THERMAL_DIRECTIONS,
    WEIGHTS,
    _validate_identity,
    solver_summary,
)
from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    annual_comparisons,
    annual_noninferiority,
    moving_block_bootstrap_distribution,
    safe_wilcoxon,
)
from scripts.evaluate_eact_mpc import THERMAL_METRIC_SCHEMA  # noqa: E402
from scripts.run_eact_final_annual import (  # noqa: E402
    CONTROLLERS,
    COUNTRIES,
)


DEFAULT_ROOT = os.path.join(ROOT, "results", "ashrae_inlet_annual_v1")
DEFAULT_OUT = os.path.join(DEFAULT_ROOT, "analysis")
ALGORITHMS = {
    "nominal_causal_mpc",
    "static_robust_mpc",
    "eact_mpc",
}
EXPECTED_HOURS = 8_736
EXPECTED_WEEKS = 52


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser


def _load_outputs(root: str, kind: str) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", f"{kind}_*.csv")))
    expected = len(COUNTRIES) * len(CONTROLLERS)
    if len(paths) != expected:
        raise ValueError(
            f"Expected {expected} annual {kind} files, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _load_manifests(root: str) -> list[dict]:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "manifest_*.json")))
    expected = len(COUNTRIES) * len(CONTROLLERS)
    if len(paths) != expected:
        raise ValueError(
            f"Expected {expected} annual manifests, found {len(paths)}")
    items = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        item["_path"] = os.path.abspath(path)
        items.append(item)
    return items


def validate_annual_matrix(
    episodes: pd.DataFrame,
    weekly: pd.DataFrame,
    hourly: pd.DataFrame,
    solver: pd.DataFrame,
    manifests: list[dict],
) -> None:
    for frame in (episodes, weekly, hourly, solver):
        _validate_identity(frame)
    expected_series = len(COUNTRIES) * len(CONTROLLERS)
    if len(episodes) != expected_series:
        raise ValueError("Annual episode count is incorrect")
    if len(weekly) != expected_series * EXPECTED_WEEKS:
        raise ValueError("Annual weekly row count is incorrect")
    if len(hourly) != expected_series * EXPECTED_HOURS:
        raise ValueError("Annual hourly row count is incorrect")
    if len(solver) != expected_series * EXPECTED_HOURS:
        raise ValueError("Annual solver row count is incorrect")
    for frame in (episodes, weekly, hourly, solver):
        if set(frame.country) != set(COUNTRIES):
            raise ValueError("Annual country set is incomplete")
        if set(frame.algorithm) != ALGORITHMS:
            raise ValueError("Annual controller set is incomplete")
        if set(frame.forecast_stress) != {"none"}:
            raise ValueError("Annual trajectories must use no forecast stress")
        if set(frame.thermal_safety_shield.astype(bool)) != {True}:
            raise ValueError("Annual external 32 degC safety layer must be enabled")
        for column, value in WEIGHTS.items():
            if not np.allclose(frame[column].astype(float), value):
                raise ValueError(f"Annual objective mismatch: {column}")
    eact = episodes[episodes.algorithm == "eact_mpc"]
    if not np.allclose(eact.adaptive_beta_floor.astype(float), 0.10):
        raise ValueError("Annual EACT beta floor is incorrect")
    if episodes.duplicated(["country", "algorithm"]).any():
        raise ValueError("Duplicate annual trajectory")
    if weekly.duplicated(["country", "algorithm", "week"]).any():
        raise ValueError("Duplicate annual weekly key")
    counts = weekly.groupby(["country", "algorithm"]).week.nunique()
    if not np.all(counts.to_numpy() == EXPECTED_WEEKS):
        raise ValueError("Annual trajectories require 52 complete weeks")
    if hourly.duplicated(["country", "algorithm", "episode", "step"]).any():
        raise ValueError("Duplicate annual hourly key")
    if solver.duplicated(["country", "algorithm", "episode", "step"]).any():
        raise ValueError("Duplicate annual solver key")
    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get(
                "schema_version") != 2:
            raise ValueError(f"Incomplete annual manifest: {manifest['_path']}")
        cfg = manifest.get("configuration", {})
        expected = {
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
            "constraint_tolerance": 1e-4,
            "thermal_safety_shield": True,
            "continuous_year": True,
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
                    f"Annual manifest mismatch for {key}: "
                    f"{manifest['_path']}")
        outputs = manifest.get("outputs", {})
        if outputs["episodes"]["rows"] != 1:
            raise ValueError("Annual manifest episode count is incorrect")
        if outputs["weekly"]["rows"] != EXPECTED_WEEKS:
            raise ValueError("Annual manifest weekly count is incorrect")
        if outputs["hourly"]["rows"] != EXPECTED_HOURS:
            raise ValueError("Annual manifest hourly count is incorrect")


def annual_thermal_comparisons(
    weekly: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260725,
) -> pd.DataFrame:
    rows = []
    keys = ["country", "week", "start_step"]
    for baseline in ("nominal_causal_mpc", "static_robust_mpc"):
        for offset, (metric, direction) in enumerate(
                THERMAL_DIRECTIONS.items()):
            eact = weekly.loc[
                weekly.algorithm == "eact_mpc", [*keys, metric]]
            base = weekly.loc[
                weekly.algorithm == baseline, [*keys, metric]]
            paired = eact.merge(
                base,
                on=keys,
                suffixes=("_eact", "_baseline"),
                validate="one_to_one",
            )
            if len(paired) != len(COUNTRIES) * EXPECTED_WEEKS:
                raise ValueError(
                    f"Incomplete annual thermal pairs for {baseline}/{metric}")
            eact_values = paired[f"{metric}_eact"].to_numpy(float)
            baseline_values = paired[f"{metric}_baseline"].to_numpy(float)
            paired["effect"] = (
                baseline_values - eact_values
                if direction == "lower"
                else eact_values - baseline_values
            )
            distribution = moving_block_bootstrap_distribution(
                paired,
                "effect",
                samples=samples,
                seed=seed + offset,
            )
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
                "treatment": "eact_mpc",
                "baseline": baseline,
                "metric": metric,
                "favorable_direction": direction,
                "n_weeks": len(paired),
                "mean_improvement": float(paired.effect.mean()),
                "moving_block_ci95_low": float(low),
                "moving_block_ci95_high": float(high),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.effect.to_numpy(dtype=float)),
            })
    return pd.DataFrame(rows)


def annual_absolute_summary(
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for algorithm, episode_group in episodes.groupby("algorithm"):
        hour_group = hourly[hourly.algorithm == algorithm]
        rows.append({
            "algorithm": algorithm,
            "n_countries": len(episode_group),
            "mean_annual_common_cost": float(
                -episode_group.episode_return.mean()),
            "mean_annual_e_grid_mwh": float(
                episode_group.e_grid_mwh.mean()),
            "mean_annual_co2_kg": float(episode_group.co2_kg.mean()),
            "mean_annual_e_total_mwh": float(
                episode_group.e_total_mwh.mean()),
            "total_recommended_exceedance_hours": float(
                hour_group.recommended_exceedance_event.sum()),
            "total_recommended_exceedance_degc_h": float(
                hour_group.recommended_exceedance_c.sum()),
            "overall_max_t_inlet_c": float(hour_group.t_inlet_c.max()),
            "total_allowable_exceedance_hours": float(
                hour_group.allowable_exceedance_event.sum()),
            "total_sla_violation_mwh": float(
                hour_group.sla_violation_mwh.sum()),
            "thermal_safety_override_steps": int(
                (hour_group.thermal_safety_override > 1e-12).sum()),
            "mean_thermal_safety_override": float(
                hour_group.thermal_safety_override.mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    episodes = _load_outputs(args.root, "episodes")
    weekly = _load_outputs(args.root, "weekly")
    hourly = _load_outputs(args.root, "hourly")
    solver = _load_outputs(args.root, "solver")
    manifests = _load_manifests(args.root)
    validate_annual_matrix(episodes, weekly, hourly, solver, manifests)

    outputs = {
        "annual_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_manifests": len(manifests),
            "n_trajectories": len(episodes),
            "n_weekly_rows": len(weekly),
            "n_hourly_rows": len(hourly),
            "n_solver_rows": len(solver),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "annual_absolute_summary.csv": annual_absolute_summary(
            episodes, hourly),
        "annual_cost_comparisons.csv": annual_comparisons(
            weekly, samples=args.bootstrap_samples),
        "annual_noninferiority.csv": annual_noninferiority(
            weekly, samples=args.bootstrap_samples),
        "annual_thermal_comparisons.csv": annual_thermal_comparisons(
            weekly, samples=args.bootstrap_samples),
        "annual_solver_summary.csv": solver_summary(solver),
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
        "annual_root": os.path.abspath(args.root),
        "bootstrap_samples": args.bootstrap_samples,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [item["_path"] for item in manifests],
        "outputs": output_records,
    }
    path = os.path.join(args.out_dir, "annual_analysis_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
