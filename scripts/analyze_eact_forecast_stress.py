"""Analyze paired MPC performance under forecast-distribution stress."""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    hierarchical_bootstrap_distribution,
    safe_wilcoxon,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_ROOT = os.path.join(ROOT, "results", "eact_forecast_stress_v1")
ALGORITHMS = ("nominal_causal_mpc", "static_robust_mpc", "eact_mpc")
OBJECTIVE_WEIGHT_DEFAULTS = {
    "weight_grid": 1.0,
    "weight_co2": 2.0,
    "weight_total": 0.2,
    "weight_smooth": 0.5,
}
OBJECTIVE_WEIGHT_COLUMNS = tuple(OBJECTIVE_WEIGHT_DEFAULTS)
EACT_GROUP_COLUMNS = (
    "forecast_stress", "forecast_stress_scale", "thermal_safety_shield",
    "adaptive_beta_floor",
    *OBJECTIVE_WEIGHT_COLUMNS,
)
BASELINE_MATCH_COLUMNS = (
    "forecast_stress", "forecast_stress_scale", "thermal_safety_shield",
    *OBJECTIVE_WEIGHT_COLUMNS,
)
SAFETY_METRICS = (
    "safety_infeasible_hours",
    "thermal_margin_violation_hours",
    "thermal_margin_exceedance_kh",
    "sla_violation_mwh",
    "terminal_unserved_mwh",
    "thermal_violation_hours",
)


def load_episodes(root: str) -> pd.DataFrame:
    paths = [
        path for path in glob.glob(
            os.path.join(root, "**", "episodes_*.csv"), recursive=True)
        if f"{os.sep}_smoke{os.sep}" not in path
    ]
    if not paths:
        raise FileNotFoundError(f"No forecast-stress episode files under {root}")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = {
        "country", "algorithm", "start_hour", "episode_return",
        "forecast_stress", "forecast_stress_scale", "thermal_safety_shield",
        "safety_infeasible_hours", "thermal_margin_violation_hours",
        "thermal_margin_exceedance_kh", "max_t_room_c", "sla_violation_mwh",
        "terminal_unserved_mwh", "thermal_violation_hours",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast-stress results missing columns: {sorted(missing)}")
    if "adaptive_beta_floor" not in frame.columns:
        frame["adaptive_beta_floor"] = 0.0
    for column, default in OBJECTIVE_WEIGHT_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default
    frame = frame[frame.algorithm.isin(ALGORITHMS)].copy()
    key = [
        "country", "algorithm", "start_hour", "forecast_stress",
        "forecast_stress_scale", "thermal_safety_shield",
        "adaptive_beta_floor",
        *OBJECTIVE_WEIGHT_COLUMNS,
    ]
    if frame.duplicated(key).any():
        raise ValueError("Forecast-stress results contain duplicate paired rows")
    return frame


def _matching_baseline(
    frame: pd.DataFrame,
    config: tuple,
    baseline: str,
) -> pd.DataFrame:
    config_values = dict(zip(EACT_GROUP_COLUMNS, config))
    mask = frame.algorithm == baseline
    for column in BASELINE_MATCH_COLUMNS:
        mask &= frame[column] == config_values[column]
    matching = frame[mask].copy()
    baseline_key = ["country", "start_hour"]
    if matching.duplicated(baseline_key).any():
        value_columns = ["episode_return", *[
            column for column in (*SAFETY_METRICS, "max_t_room_c")
            if column in matching.columns
        ]]
        inconsistent = matching.groupby(baseline_key)[value_columns].nunique()
        if (inconsistent > 1).any().any():
            raise ValueError(
                f"Conflicting duplicate {baseline} rows for {config[:3]}")
        matching = matching.drop_duplicates(baseline_key)
    if matching.empty:
        raise ValueError(f"Missing {baseline} rows for {config[:3]}")
    return matching


def paired_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int = 10_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    frame = episodes.copy()
    frame["common_cost"] = -frame.episode_return
    rows = []
    eact_frame = frame[frame.algorithm == "eact_mpc"]
    for config, eact in eact_frame.groupby(list(EACT_GROUP_COLUMNS), sort=True):
        for baseline in ("nominal_causal_mpc", "static_robust_mpc"):
            other = _matching_baseline(frame, config, baseline)
            paired = eact.merge(
                other,
                on=["country", "start_hour"],
                suffixes=("_eact", "_baseline"),
                validate="one_to_one",
            )
            paired["relative_improvement"] = (
                (paired.common_cost_baseline - paired.common_cost_eact)
                / paired.common_cost_baseline)
            paired["season"] = paired["start_hour"]
            distribution = hierarchical_bootstrap_distribution(
                paired,
                "relative_improvement",
                samples=samples,
                seed=seed,
            )
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
                "forecast_stress": config[0],
                "forecast_stress_scale": float(config[1]),
                "thermal_safety_shield": bool(config[2]),
                "adaptive_beta_floor": float(config[3]),
                **{
                    column: float(config_values)
                    for column, config_values in zip(
                        OBJECTIVE_WEIGHT_COLUMNS, config[4:])
                },
                "baseline": baseline,
                "n_pairs": len(paired),
                "mean_relative_improvement_pct": float(
                    100.0 * paired.relative_improvement.mean()),
                "ci95_low_pct": float(100.0 * low),
                "ci95_high_pct": float(100.0 * high),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.relative_improvement.to_numpy()),
            })
    return pd.DataFrame(rows)


def paired_safety_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int = 10_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    rows = []
    eact_frame = episodes[episodes.algorithm == "eact_mpc"]
    for config, eact in eact_frame.groupby(list(EACT_GROUP_COLUMNS), sort=True):
        for baseline in ("nominal_causal_mpc", "static_robust_mpc"):
            other = _matching_baseline(episodes, config, baseline)
            paired = eact.merge(
                other,
                on=["country", "start_hour"],
                suffixes=("_eact", "_baseline"),
                validate="one_to_one",
            )
            paired["season"] = paired["start_hour"]
            for metric in SAFETY_METRICS:
                reduction = f"{metric}_reduction"
                paired[reduction] = (
                    paired[f"{metric}_baseline"]
                    - paired[f"{metric}_eact"]
                )
                distribution = hierarchical_bootstrap_distribution(
                    paired, reduction, samples=samples, seed=seed)
                low, high = np.quantile(distribution, [0.025, 0.975])
                rows.append({
                    "forecast_stress": config[0],
                    "forecast_stress_scale": float(config[1]),
                    "thermal_safety_shield": bool(config[2]),
                    "adaptive_beta_floor": float(config[3]),
                    **{
                        column: float(config_values)
                        for column, config_values in zip(
                            OBJECTIVE_WEIGHT_COLUMNS, config[4:])
                    },
                    "baseline": baseline,
                    "metric": metric,
                    "n_pairs": len(paired),
                    "mean_reduction": float(paired[reduction].mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "wilcoxon_p_value": safe_wilcoxon(
                        paired[reduction].to_numpy()),
                })
    return pd.DataFrame(rows)


def safety_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    metrics = [*SAFETY_METRICS, "max_t_room_c"]
    grouped = episodes.groupby([
        "forecast_stress", "forecast_stress_scale",
        "thermal_safety_shield", "adaptive_beta_floor",
        *OBJECTIVE_WEIGHT_COLUMNS, "algorithm"], sort=True)
    rows = []
    for config, group in grouped:
        row = {
            "forecast_stress": config[0],
            "forecast_stress_scale": float(config[1]),
            "thermal_safety_shield": bool(config[2]),
            "adaptive_beta_floor": float(config[3]),
            **{
                column: float(config_values)
                for column, config_values in zip(
                    OBJECTIVE_WEIGHT_COLUMNS, config[4:8])
            },
            "algorithm": config[8],
            "n_episodes": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_total"] = float(group[metric].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", default=[DEFAULT_ROOT])
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.roots[0], "analysis")
    os.makedirs(out_dir, exist_ok=True)
    episodes = pd.concat(
        [load_episodes(root) for root in args.roots], ignore_index=True)
    combined_key = [
        "country", "algorithm", "start_hour", "forecast_stress",
        "forecast_stress_scale", "thermal_safety_shield",
        "adaptive_beta_floor",
        *OBJECTIVE_WEIGHT_COLUMNS,
    ]
    if episodes.duplicated(combined_key).any():
        raise ValueError("Combined forecast-stress roots contain duplicate rows")
    outputs = {
        "stress_comparisons.csv": paired_comparisons(
            episodes, samples=args.bootstrap_samples),
        "stress_safety.csv": safety_summary(episodes),
        "stress_safety_comparisons.csv": paired_safety_comparisons(
            episodes, samples=args.bootstrap_samples),
    }
    for name, frame in outputs.items():
        path = os.path.join(out_dir, name)
        frame.to_csv(path, index=False)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
