"""Validate and analyze the ASHRAE-inlet beta-floor ablation."""

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

from coastaldc_env import COUNTRIES  # noqa: E402
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


DEFAULT_BETA_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_beta_ablation_v1")
DEFAULT_SEASONAL_ROOT = os.path.join(
    ROOT, "results", "ashrae_inlet_seasonal_v1")
DEFAULT_OUT = os.path.join(DEFAULT_BETA_ROOT, "analysis")
EXPECTED_COUNTRIES = set(COUNTRIES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta-root", default=DEFAULT_BETA_ROOT)
    parser.add_argument("--seasonal-root", default=DEFAULT_SEASONAL_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser


def _load_country_outputs(root: str, kind: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(root, "*", f"{kind}_*.csv")))
    if len(paths) != len(EXPECTED_COUNTRIES):
        raise ValueError(
            f"Expected {len(EXPECTED_COUNTRIES)} beta {kind} files, "
            f"found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _load_beta_manifests(root: str) -> list[dict]:
    paths = sorted(glob.glob(os.path.join(root, "*", "manifest_*.json")))
    if len(paths) != len(EXPECTED_COUNTRIES):
        raise ValueError(
            f"Expected {len(EXPECTED_COUNTRIES)} beta manifests, "
            f"found {len(paths)}")
    manifests = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            item = json.load(handle)
        item["_path"] = os.path.abspath(path)
        manifests.append(item)
    return manifests


def _load_seasonal_episodes(root: str) -> pd.DataFrame:
    paths = sorted(glob.glob(
        os.path.join(root, "*", "*", "episodes_*.csv")))
    if len(paths) != 2 * len(EXPECTED_COUNTRIES):
        raise ValueError(
            f"Expected {2 * len(EXPECTED_COUNTRIES)} seasonal episode files, "
            f"found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _validate_beta_rows(
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
    solver: pd.DataFrame,
    manifests: list[dict],
) -> None:
    _validate_identity(episodes)
    _validate_identity(hourly)
    _validate_identity(solver)
    if len(episodes) != len(EXPECTED_COUNTRIES) * len(START_HOURS):
        raise ValueError("Beta episode count is incorrect")
    if len(hourly) != len(episodes) * 168 or len(solver) != len(episodes) * 168:
        raise ValueError("Beta hourly or solver row count is incorrect")
    if set(episodes.country) != EXPECTED_COUNTRIES:
        raise ValueError("Beta country set is incomplete")
    if set(episodes.algorithm) != {"eact_mpc"}:
        raise ValueError("Beta ablation must contain only EACT-MPC")
    if set(episodes.start_hour.astype(int)) != START_HOURS:
        raise ValueError("Beta seasonal starts are incorrect")
    if set(episodes.forecast_stress) != {"adverse_bias"}:
        raise ValueError("Beta forecast stress is incorrect")
    if not np.allclose(episodes.forecast_stress_scale.astype(float), 1.0):
        raise ValueError("Beta forecast-stress scale is incorrect")
    if not np.allclose(episodes.adaptive_beta_floor.astype(float), 0.0):
        raise ValueError("Beta-zero rows have an incorrect beta floor")
    if set(episodes.thermal_safety_shield.astype(bool)) != {False}:
        raise ValueError("Beta ablation must disable the safety shield")
    for column, value in WEIGHTS.items():
        if not np.allclose(episodes[column].astype(float), value):
            raise ValueError(f"Beta objective mismatch: {column}")
    keys = ["country", "start_hour"]
    if episodes.duplicated(keys).any():
        raise ValueError("Duplicate beta controller-week key")
    hourly_keys = ["country", "episode", "step"]
    solver_keys = ["country", "episode", "step"]
    if hourly.duplicated(hourly_keys).any() or solver.duplicated(solver_keys).any():
        raise ValueError("Duplicate beta hourly or solver key")
    for manifest in manifests:
        if manifest.get("status") != "complete" or manifest.get(
                "schema_version") != 2:
            raise ValueError(f"Incomplete beta manifest: {manifest['_path']}")
        cfg = manifest.get("configuration", {})
        expected = {
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
            "adaptive_beta_floor": 0.0,
            "forecast_stress": "adverse_bias",
            "forecast_stress_scale": 1.0,
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
                    f"Beta manifest mismatch for {key}: {manifest['_path']}")
        if manifest["outputs"]["episodes"]["rows"] != len(START_HOURS):
            raise ValueError("Beta manifest episode count is incorrect")


def _matched_variants(
    beta: pd.DataFrame,
    seasonal: pd.DataFrame,
) -> pd.DataFrame:
    final = seasonal[
        (seasonal.algorithm == "eact_mpc")
        & (seasonal.forecast_stress == "adverse_bias")
        & np.isclose(seasonal.forecast_stress_scale.astype(float), 1.0)
    ].copy()
    if len(final) != len(beta):
        raise ValueError("Final EACT and beta-zero row counts do not match")
    _validate_identity(final)
    if not np.allclose(final.adaptive_beta_floor.astype(float), 0.10):
        raise ValueError("Final EACT rows have an incorrect beta floor")
    identity = [
        *WEIGHTS,
        "forecast_stress_scale",
        "constraint_tolerance",
        "t_inlet_recommended_min_c",
        "t_inlet_recommended_max_c",
        "t_inlet_allowable_min_c",
        "t_inlet_allowable_max_c",
    ]
    keys = ["country", "start_hour"]
    beta_check = beta[[*keys, *identity]]
    final_check = final[[*keys, *identity]]
    checked = beta_check.merge(
        final_check,
        on=keys,
        suffixes=("_beta0", "_final"),
        validate="one_to_one",
    )
    for column in identity:
        if not np.allclose(
                checked[f"{column}_beta0"].astype(float),
                checked[f"{column}_final"].astype(float)):
            raise ValueError(f"Beta comparison identity mismatch: {column}")
    beta = beta.copy()
    final = final.copy()
    beta["variant"] = "beta0"
    final["variant"] = "beta0p10"
    return pd.concat([beta, final], ignore_index=True)


def paired_comparisons(
    variants: pd.DataFrame,
    *,
    samples: int,
    seed: int = 20260725,
) -> pd.DataFrame:
    frame = variants.copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    rows = []
    metrics = {
        **{metric: "lower" for metric in COST_METRICS},
        **THERMAL_DIRECTIONS,
    }
    for offset, (metric, direction) in enumerate(metrics.items()):
        keys = ["country", "start_hour"]
        final = frame.loc[
            frame.variant == "beta0p10", [*keys, metric]]
        beta0 = frame.loc[frame.variant == "beta0", [*keys, metric]]
        paired = final.merge(
            beta0,
            on=keys,
            suffixes=("_final", "_beta0"),
            validate="one_to_one",
        )
        final_values = paired[f"{metric}_final"].to_numpy(float)
        beta_values = paired[f"{metric}_beta0"].to_numpy(float)
        paired["effect"] = (
            beta_values - final_values
            if direction == "lower"
            else final_values - beta_values
        )
        paired["season"] = paired.start_hour
        distribution = hierarchical_bootstrap_distribution(
            paired, "effect", samples=samples, seed=seed + offset)
        low, high = np.quantile(distribution, [0.025, 0.975])
        row = {
            "treatment": "beta0p10",
            "baseline": "beta0",
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
            denominator = beta_values
            if np.any(np.abs(denominator) <= 1e-12):
                raise ValueError(f"Zero beta-zero baseline for {metric}")
            relative = (beta_values - final_values) / np.abs(denominator)
            paired["relative_effect"] = relative
            relative_distribution = hierarchical_bootstrap_distribution(
                paired,
                "relative_effect",
                samples=samples,
                seed=seed + 100 + offset,
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


def variant_summary(variants: pd.DataFrame) -> pd.DataFrame:
    frame = variants.copy()
    frame["common_cost"] = -frame.episode_return.astype(float)
    metrics = [
        "common_cost",
        "e_grid_mwh",
        "co2_kg",
        "e_total_mwh",
        "recommended_exceedance_degc_h",
        "recommended_exceedance_hours",
        "recommended_compliance_pct",
        "p95_t_inlet_c",
        "p99_t_inlet_c",
        "max_t_inlet_c",
        "allowable_exceedance_hours",
    ]
    return frame.groupby("variant", as_index=False)[metrics].mean()


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    beta = _load_country_outputs(args.beta_root, "episodes")
    hourly = _load_country_outputs(args.beta_root, "hourly")
    solver = _load_country_outputs(args.beta_root, "solver")
    manifests = _load_beta_manifests(args.beta_root)
    _validate_beta_rows(beta, hourly, solver, manifests)
    seasonal = _load_seasonal_episodes(args.seasonal_root)
    variants = _matched_variants(beta, seasonal)

    outputs = {
        "beta_validation_summary.csv": pd.DataFrame([{
            "status": "passed",
            "n_manifests": len(manifests),
            "n_beta0_weeks": len(beta),
            "n_matched_pairs": len(beta),
            "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        }]),
        "beta_variant_summary.csv": variant_summary(variants),
        "beta_paired_comparisons.csv": paired_comparisons(
            variants, samples=args.bootstrap_samples),
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
        "beta_root": os.path.abspath(args.beta_root),
        "seasonal_root": os.path.abspath(args.seasonal_root),
        "bootstrap_samples": args.bootstrap_samples,
        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
        "inputs": [item["_path"] for item in manifests],
        "outputs": output_records,
    }
    path = os.path.join(args.out_dir, "beta_analysis_manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
