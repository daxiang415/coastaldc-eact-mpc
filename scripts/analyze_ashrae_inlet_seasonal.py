"""Validate and analyze the 15-country ASHRAE inlet seasonal matrices."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from coastaldc_env import COUNTRIES  # noqa: E402
from scripts.analyze_ashrae_inlet_pilot import (  # noqa: E402
    ALGORITHMS,
    SCENARIOS,
    START_HOURS,
    absolute_summary,
    cost_comparisons,
    load_manifests,
    load_outputs,
    solver_summary,
    thermal_comparisons,
    validate_episode_matrix,
    validate_hourly_matrix,
    validate_manifests,
    validate_solver_matrix,
)
from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    hierarchical_bootstrap_distribution,
)
from scripts.evaluate_eact_mpc import THERMAL_METRIC_SCHEMA  # noqa: E402


DEFAULT_ROOT = os.path.join(ROOT, "results", "ashrae_inlet_seasonal_v1")
DEFAULT_OUT = os.path.join(DEFAULT_ROOT, "analysis")
EXPECTED_COUNTRIES = set(COUNTRIES)
EXPECTED_MANIFESTS = len(EXPECTED_COUNTRIES) * len(SCENARIOS)
EXPECTED_CONTROLLER_WEEKS = (
    len(EXPECTED_COUNTRIES)
    * len(ALGORITHMS)
    * len(SCENARIOS)
    * len(START_HOURS)
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--noninferiority-margin-pct", type=float, default=1.0)
    return parser


def no_shift_noninferiority(
    episodes: pd.DataFrame,
    *,
    samples: int,
    margin_pct: float,
    seed: int = 20260725,
) -> pd.DataFrame:
    frame = episodes[episodes.forecast_stress == "none"].copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    keys = ["country", "start_hour"]
    eact = frame.loc[
        frame.algorithm == "eact_mpc", [*keys, "common_cost"]]
    static = frame.loc[
        frame.algorithm == "static_robust_mpc", [*keys, "common_cost"]]
    paired = eact.merge(
        static,
        on=keys,
        suffixes=("_eact", "_static"),
        validate="one_to_one",
    )
    expected_pairs = len(EXPECTED_COUNTRIES) * len(START_HOURS)
    if len(paired) != expected_pairs:
        raise ValueError(
            f"Expected {expected_pairs} no-shift pairs, found {len(paired)}")
    denominator = paired.common_cost_static.to_numpy(dtype=float)
    if np.any(np.abs(denominator) <= 1e-12):
        raise ValueError("Static common cost contains zero values")
    paired["cost_increase_fraction"] = (
        paired.common_cost_eact.to_numpy(dtype=float) - denominator
    ) / np.abs(denominator)
    paired["season"] = paired.start_hour
    distribution = hierarchical_bootstrap_distribution(
        paired,
        "cost_increase_fraction",
        samples=samples,
        seed=seed,
    )
    upper = float(np.quantile(distribution, 0.95))
    margin = margin_pct / 100.0
    return pd.DataFrame([{
        "comparison": "eact_mpc_vs_static_robust_mpc",
        "metric": "common_cost",
        "n_pairs": len(paired),
        "mean_cost_increase_pct": float(
            100.0 * paired.cost_increase_fraction.mean()),
        "one_sided_95_upper_pct": 100.0 * upper,
        "noninferiority_margin_pct": margin_pct,
        "noninferiority_passed": bool(upper < margin),
    }])


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.noninferiority_margin_pct <= 0.0:
        raise ValueError("--noninferiority-margin-pct must be positive")

    episodes = load_outputs(
        args.root, "episodes", expected_files=EXPECTED_MANIFESTS)
    hourly = load_outputs(
        args.root, "hourly", expected_files=EXPECTED_MANIFESTS)
    solver = load_outputs(
        args.root, "solver", expected_files=EXPECTED_MANIFESTS)
    manifests = load_manifests(
        args.root, expected_files=EXPECTED_MANIFESTS)

    validate_episode_matrix(
        episodes, expected_countries=EXPECTED_COUNTRIES)
    validate_hourly_matrix(
        hourly, expected_controller_weeks=EXPECTED_CONTROLLER_WEEKS)
    validate_solver_matrix(
        solver, expected_controller_weeks=EXPECTED_CONTROLLER_WEEKS)
    validate_manifests(manifests)

    outputs = {
        "seasonal_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_countries": len(EXPECTED_COUNTRIES),
            "n_manifests": len(manifests),
            "n_controller_weeks": len(episodes),
            "n_controller_hours": len(hourly),
            "n_solver_rows": len(solver),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "seasonal_absolute_summary.csv": absolute_summary(episodes),
        "seasonal_cost_comparisons.csv": cost_comparisons(
            episodes, samples=args.bootstrap_samples),
        "seasonal_thermal_comparisons.csv": thermal_comparisons(
            episodes, samples=args.bootstrap_samples),
        "seasonal_solver_summary.csv": solver_summary(solver),
        "seasonal_no_shift_noninferiority.csv": no_shift_noninferiority(
            episodes,
            samples=args.bootstrap_samples,
            margin_pct=args.noninferiority_margin_pct,
        ),
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

    analysis_manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seasonal_root": os.path.abspath(args.root),
        "bootstrap_samples": args.bootstrap_samples,
        "noninferiority_margin_pct": args.noninferiority_margin_pct,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [item["_path"] for item in manifests],
        "outputs": output_records,
    }
    manifest_path = os.path.join(
        args.out_dir, "seasonal_analysis_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(analysis_manifest, handle, indent=2)
    print(f"saved -> {manifest_path}")


if __name__ == "__main__":
    main()
