"""Generate the final EACT-MPC paper evidence tables."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.analyze_eact_paper_evidence import (  # noqa: E402
    analyze_forecast_calibration,
    annual_comparisons,
    annual_noninferiority,
    hierarchical_bootstrap_distribution,
    holm_adjust,
    safe_wilcoxon,
)
from coastaldc_env import COUNTRIES  # noqa: E402
from scripts.analyze_eact_forecast_stress import (  # noqa: E402
    load_episodes as load_stress_episodes,
    paired_comparisons as stress_cost_comparisons,
    paired_safety_comparisons as stress_constraint_comparisons,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_SEASONAL_ROOT = os.path.join(
    ROOT, "results", "eact_final_seasonal_v1")
DEFAULT_BETA_ROOT = os.path.join(ROOT, "results", "eact_beta_ablation_v1")
DEFAULT_WEIGHT_ROOT = os.path.join(
    ROOT, "results", "eact_weight_sensitivity_v1")
DEFAULT_ANNUAL_ROOT = os.path.join(ROOT, "results", "eact_final_annual_v2")
DEFAULT_FORECAST_EVAL_ROOT = os.path.join(
    ROOT, "results", "causal_forecast_evaluation_final_v1")
DEFAULT_E6_EACT_ROOT = os.path.join(
    ROOT, "results", "eact_forecast_stress_beta_v1")
DEFAULT_E6_BASELINE_ROOTS = (
    os.path.join(ROOT, "results", "eact_forecast_stress_v1"),
    os.path.join(ROOT, "results", "eact_forecast_stress_dose_v1"),
)
DEFAULT_OUT = os.path.join(ROOT, "results", "eact_final_paper_v1", "analysis")
WEIGHT_COLUMNS = (
    "weight_grid", "weight_co2", "weight_total", "weight_smooth")
PRIMARY_WEIGHTS = (1.0, 2.0, 0.2, 0.5)
COST_METRICS = ("common_cost", "e_grid_mwh", "co2_kg", "e_total_mwh")
CONSTRAINT_METRICS = (
    "thermal_margin_violation_hours",
    "thermal_margin_exceedance_kh",
    "safety_infeasible_hours",
    "thermal_violation_hours",
    "sla_violation_mwh",
    "terminal_unserved_mwh",
)
E6_CONDITIONS = {
    ("none", 0.0),
    ("adverse_bias", 0.5),
    ("adverse_bias", 1.0),
    ("adverse_bias", 2.0),
    ("noise", 1.0),
    ("combined", 1.0),
}
FORECAST_COLUMNS = {
    "fixed_load_mw",
    "flexible_arrival_mw",
    "sst_c",
    "wind_mw",
    "carbon_kg_per_mwh",
}


def load_outputs(root: str, kind: str, *, stage: str | None = None) -> pd.DataFrame:
    base = os.path.join(root, stage) if stage else root
    paths = sorted(glob.glob(
        os.path.join(base, "**", f"{kind}_*.csv"), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No {kind} files under {base}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.abspath(path)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    defaults = dict(zip(WEIGHT_COLUMNS, PRIMARY_WEIGHTS))
    for column, value in defaults.items():
        if column not in result.columns:
            result[column] = value
    if "adaptive_beta_floor" not in result.columns:
        result["adaptive_beta_floor"] = 0.0
    return result


def validate_configuration(
    frame: pd.DataFrame,
    *,
    algorithms: tuple[str, ...],
    stress: str,
    stress_scale: float,
    beta_floor: float | None,
    shield: bool,
    weights: tuple[float, float, float, float] = PRIMARY_WEIGHTS,
) -> None:
    required = {
        "country", "algorithm", "start_hour", "episode_return",
        "forecast_stress", "forecast_stress_scale", "thermal_safety_shield",
        "adaptive_beta_floor", *WEIGHT_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Results missing configuration columns: {sorted(missing)}")
    if set(frame.algorithm.unique()) != set(algorithms):
        raise ValueError("Controller set does not match the requested experiment")
    if set(frame.forecast_stress.unique()) != {stress}:
        raise ValueError("Mixed or incorrect forecast-stress modes")
    if not np.allclose(frame.forecast_stress_scale, stress_scale):
        raise ValueError("Mixed or incorrect forecast-stress scales")
    if beta_floor is not None:
        eact = frame[frame.algorithm.str.startswith("eact")]
        if eact.empty or not np.allclose(eact.adaptive_beta_floor, beta_floor):
            raise ValueError("EACT beta-floor configuration mismatch")
    if set(frame.thermal_safety_shield.astype(bool).unique()) != {shield}:
        raise ValueError("Mixed or incorrect thermal-safety-shield states")
    for column, expected in zip(WEIGHT_COLUMNS, weights):
        if not np.allclose(frame[column], expected):
            raise ValueError(f"Mixed or incorrect objective weight: {column}")
    key = ["country", "algorithm", "start_hour"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate country-controller-season rows")
    counts = frame.groupby(["country", "algorithm"]).size()
    if counts.empty or not np.all(counts.to_numpy() == 4):
        raise ValueError("Every seasonal country-controller pair needs four starts")


def validate_annual_configuration(weekly: pd.DataFrame) -> None:
    required = {
        "country", "algorithm", "week", "start_step", "reward",
        "mpc_horizon", "confidence", "intervention_weight",
        "forecast_stress", "forecast_stress_scale", "thermal_safety_shield",
        "adaptive_beta_floor", *WEIGHT_COLUMNS,
    }
    missing = required - set(weekly.columns)
    if missing:
        raise ValueError(
            f"Annual results missing configuration columns: {sorted(missing)}")
    expected_algorithms = {
        "nominal_causal_mpc", "static_robust_mpc", "eact_mpc"}
    if set(weekly.algorithm.unique()) != expected_algorithms:
        raise ValueError("Annual controller set does not match the final experiment")
    if set(weekly.country.unique()) != {"CHN", "JPN", "NOR"}:
        raise ValueError("Annual experiment requires CHN, JPN, and NOR")
    if set(weekly.forecast_stress.unique()) != {"none"}:
        raise ValueError("Annual experiment must use the no-stress forecast")
    if not np.allclose(weekly.forecast_stress_scale, 0.0):
        raise ValueError("Annual forecast-stress scale must be zero")
    if set(weekly.thermal_safety_shield.astype(bool).unique()) != {True}:
        raise ValueError("Annual experiment must enable the thermal safety shield")
    if not np.allclose(weekly.mpc_horizon, 24):
        raise ValueError("Annual MPC horizon must be 24 hours")
    if not np.allclose(weekly.confidence, 0.90):
        raise ValueError("Annual confidence must be 0.90")
    if not np.allclose(weekly.intervention_weight, 0.0):
        raise ValueError("Annual intervention weight must be zero")
    for column, expected in zip(WEIGHT_COLUMNS, PRIMARY_WEIGHTS):
        if not np.allclose(weekly[column], expected):
            raise ValueError(f"Annual objective-weight mismatch: {column}")
    eact = weekly[weekly.algorithm == "eact_mpc"]
    if not np.allclose(eact.adaptive_beta_floor, 0.10):
        raise ValueError("Annual EACT beta floor must be 0.10")
    key = ["country", "algorithm", "week", "start_step"]
    if weekly.duplicated(key).any():
        raise ValueError("Duplicate annual country-controller-week rows")
    counts = weekly.groupby(["country", "algorithm"]).week.nunique()
    if len(counts) != 9 or not np.all(counts.to_numpy() == 52):
        raise ValueError("Annual comparison requires 52 complete weeks per series")


def validate_forecast_evaluation(summary: pd.DataFrame) -> None:
    required = {
        "country", "mode", "column", "n_forecasts", "coverage",
        "adaptive_beta_floor",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            f"Forecast evaluation missing columns: {sorted(missing)}")
    if set(summary.country.unique()) != set(COUNTRIES):
        raise ValueError("Forecast evaluation requires all 15 countries")
    if set(summary["mode"].unique()) != {"nominal", "static", "adaptive"}:
        raise ValueError("Forecast evaluation requires all three forecast modes")
    if set(summary["column"].unique()) != FORECAST_COLUMNS:
        raise ValueError("Forecast evaluation requires all five variables")
    if not np.allclose(summary.adaptive_beta_floor, 0.10):
        raise ValueError("Forecast evaluation must use beta floor 0.10")
    key = ["country", "mode", "column"]
    if summary.duplicated(key).any():
        raise ValueError("Duplicate forecast-evaluation summary rows")
    if len(summary) != len(COUNTRIES) * 3 * len(FORECAST_COLUMNS):
        raise ValueError("Forecast-evaluation matrix is incomplete")
    if np.any(summary.n_forecasts.astype(float) <= 0):
        raise ValueError("Forecast-evaluation counts must be positive")
    bounded = summary[summary["mode"].isin(("static", "adaptive"))]
    if bounded.coverage.isna().any():
        raise ValueError("Bounded forecast modes require finite coverage")


def validate_e6_boundary_configuration(episodes: pd.DataFrame) -> None:
    required_algorithms = {
        "nominal_causal_mpc", "static_robust_mpc", "eact_mpc"}
    if set(episodes.algorithm.unique()) != required_algorithms:
        raise ValueError("E6 requires all three MPC controllers")
    if set(episodes.country.unique()) != {"CHN", "JPN", "NOR"}:
        raise ValueError("E6 requires CHN, JPN, and NOR")
    conditions = set(zip(
        episodes.forecast_stress,
        episodes.forecast_stress_scale.astype(float),
    ))
    if conditions != E6_CONDITIONS:
        raise ValueError("E6 forecast-stress conditions are incomplete or mixed")
    if set(episodes.thermal_safety_shield.astype(bool).unique()) != {False}:
        raise ValueError("E6 must disable the external thermal safety shield")
    for column, expected in zip(WEIGHT_COLUMNS, PRIMARY_WEIGHTS):
        if not np.allclose(episodes[column], expected):
            raise ValueError(f"E6 objective-weight mismatch: {column}")
    eact = episodes[episodes.algorithm == "eact_mpc"]
    if not np.allclose(eact.adaptive_beta_floor, 0.10):
        raise ValueError("E6 must use final EACT with beta floor 0.10")
    key = [
        "country", "algorithm", "start_hour",
        "forecast_stress", "forecast_stress_scale",
    ]
    if episodes.duplicated(key).any():
        raise ValueError("Duplicate E6 country-controller-condition rows")
    counts = episodes.groupby(
        ["algorithm", "forecast_stress", "forecast_stress_scale"]).size()
    if len(counts) != 18 or not np.all(counts.to_numpy() == 12):
        raise ValueError("E6 requires 12 paired country-season rows per condition")


def load_e6_boundary_episodes(
    eact_root: str,
    baseline_roots: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    eact = load_stress_episodes(eact_root)
    eact = eact[eact.algorithm == "eact_mpc"].copy()
    baselines = pd.concat(
        [load_stress_episodes(root) for root in baseline_roots],
        ignore_index=True,
    )
    baselines = baselines[
        baselines.algorithm.isin(
            ("nominal_causal_mpc", "static_robust_mpc"))
    ].copy()
    episodes = pd.concat([eact, baselines], ignore_index=True)
    validate_e6_boundary_configuration(episodes)
    return episodes


def _boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise ValueError(f"Invalid Boolean values in {series.name}")
    return normalized.isin(("true", "1"))


def computational_performance(
    experiments: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    required = {
        "algorithm", "accepted", "solver_success", "fallback",
        "min_constraint", "solve_time_s",
    }
    rows = []
    for experiment, solver in experiments.items():
        missing = required - set(solver.columns)
        if missing:
            raise ValueError(
                f"Solver results missing columns for {experiment}: "
                f"{sorted(missing)}")
        for algorithm, group in solver.groupby("algorithm", sort=True):
            accepted = _boolean_series(group.accepted)
            converged = _boolean_series(group.solver_success)
            fallback = group.fallback.astype(str)
            rows.append({
                "experiment": experiment,
                "algorithm": algorithm,
                "n_control_steps": int(len(group)),
                "accepted_plan_rate": float(accepted.mean()),
                "solver_convergence_rate": float(converged.mean()),
                "fallback_rate": float((fallback != "none").mean()),
                "rule_based_fallbacks": int((fallback == "rule_based").sum()),
                "safe_recovery_fallbacks": int(
                    (fallback == "safe_recovery").sum()),
                "shifted_plan_fallbacks": int(
                    (fallback == "shifted_plan").sum()),
                "min_constraint": float(group.min_constraint.min()),
                "solve_time_mean_s": float(group.solve_time_s.mean()),
                "solve_time_p95_s": float(
                    group.solve_time_s.quantile(0.95)),
            })
    return pd.DataFrame(rows)


def annual_diagnostics(
    episodes: pd.DataFrame,
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    episode_metrics = (
        "thermal_margin_violation_hours",
        "thermal_margin_exceedance_kh",
        "thermal_violation_hours",
        "sla_violation_mwh",
        "terminal_unserved_mwh",
        "final_backlog_mwh",
        "safety_interventions",
        "safety_infeasible_hours",
        "workload_infeasible_hours",
    )
    episode_required = {"country", "algorithm", *episode_metrics}
    hourly_required = {
        "country", "algorithm", "safety_intervention",
        "thermal_safety_override",
    }
    missing_episode = episode_required - set(episodes.columns)
    missing_hourly = hourly_required - set(hourly.columns)
    if missing_episode or missing_hourly:
        raise ValueError(
            "Annual diagnostics missing columns: "
            f"episodes={sorted(missing_episode)}, "
            f"hourly={sorted(missing_hourly)}")

    rows = []
    for algorithm, episode_group in episodes.groupby("algorithm", sort=True):
        hourly_group = hourly[hourly.algorithm == algorithm]
        if hourly_group.empty:
            raise ValueError(
                f"Annual diagnostics missing hourly rows for {algorithm}")
        intervention_mask = _boolean_series(
            hourly_group.safety_intervention)
        intervened = hourly_group.loc[intervention_mask]
        row = {
            "algorithm": algorithm,
            "n_countries": int(episode_group.country.nunique()),
            "n_control_hours": int(len(hourly_group)),
        }
        row.update({
            metric: float(episode_group[metric].sum())
            for metric in episode_metrics
        })
        row["conditional_mean_thermal_safety_override"] = (
            float(intervened.thermal_safety_override.mean())
            if not intervened.empty else 0.0
        )
        row["max_thermal_safety_override"] = (
            float(intervened.thermal_safety_override.max())
            if not intervened.empty else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _with_common_cost(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["common_cost"] = -result.episode_return.astype(float)
    result["season"] = result.start_hour.astype(int)
    return result


def _paired(
    frame: pd.DataFrame,
    treatment: str,
    baseline: str,
    metric: str,
) -> pd.DataFrame:
    keys = ["country", "start_hour"]
    columns = [*keys, metric]
    treatment_rows = frame.loc[frame.algorithm == treatment, columns]
    baseline_rows = frame.loc[frame.algorithm == baseline, columns]
    if treatment_rows.duplicated(keys).any() or baseline_rows.duplicated(keys).any():
        raise ValueError("Duplicate paired rows")
    paired = treatment_rows.merge(
        baseline_rows,
        on=keys,
        suffixes=("_treatment", "_baseline"),
        validate="one_to_one",
    )
    if len(paired) != len(treatment_rows) or len(paired) != len(baseline_rows):
        raise ValueError(f"Incomplete pairing: {treatment} versus {baseline}")
    paired["season"] = paired.start_hour
    return paired


def seasonal_cost_comparisons(
    episodes: pd.DataFrame,
    *,
    treatment: str = "eact_mpc",
    baselines: tuple[str, ...] = ("nominal_causal_mpc", "static_robust_mpc"),
    samples: int = 20_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    frame = _with_common_cost(episodes)
    rows = []
    for baseline in baselines:
        for metric in COST_METRICS:
            paired = _paired(frame, treatment, baseline, metric)
            baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
            if np.any(np.abs(baseline_values) <= 1e-12):
                raise ValueError(f"Zero baseline values for relative metric {metric}")
            paired["relative_improvement"] = (
                baseline_values
                - paired[f"{metric}_treatment"].to_numpy(dtype=float)
            ) / np.abs(baseline_values)
            distribution = hierarchical_bootstrap_distribution(
                paired, "relative_improvement", samples=samples, seed=seed)
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
                "treatment": treatment,
                "baseline": baseline,
                "metric": metric,
                "n_pairs": len(paired),
                "mean_relative_improvement_pct": float(
                    100.0 * paired.relative_improvement.mean()),
                "ci95_low_pct": float(100.0 * low),
                "ci95_high_pct": float(100.0 * high),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.relative_improvement.to_numpy()),
            })
    result = pd.DataFrame(rows)
    result["holm_p_value"] = result.wilcoxon_p_value
    for baseline in baselines:
        mask = (
            (result.baseline == baseline)
            & result.metric.isin(("e_grid_mwh", "co2_kg", "e_total_mwh"))
        )
        result.loc[mask, "holm_p_value"] = holm_adjust(
            result.loc[mask, "wilcoxon_p_value"])
    return result


def seasonal_constraint_comparisons(
    episodes: pd.DataFrame,
    *,
    treatment: str = "eact_mpc",
    baseline: str = "static_robust_mpc",
    samples: int = 20_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    rows = []
    for metric in CONSTRAINT_METRICS:
        paired = _paired(episodes, treatment, baseline, metric)
        paired["reduction"] = (
            paired[f"{metric}_baseline"] - paired[f"{metric}_treatment"])
        distribution = hierarchical_bootstrap_distribution(
            paired, "reduction", samples=samples, seed=seed)
        low, high = np.quantile(distribution, [0.025, 0.975])
        rows.append({
            "treatment": treatment,
            "baseline": baseline,
            "metric": metric,
            "n_pairs": len(paired),
            "mean_reduction": float(paired.reduction.mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "wilcoxon_p_value": safe_wilcoxon(
                paired.reduction.to_numpy()),
        })
    return pd.DataFrame(rows)


def seasonal_noninferiority(
    episodes: pd.DataFrame,
    *,
    margin: float = 0.01,
    samples: int = 20_000,
    seed: int = 20260718,
) -> pd.DataFrame:
    frame = _with_common_cost(episodes)
    paired = _paired(
        frame, "eact_mpc", "static_robust_mpc", "common_cost")
    baseline = paired.common_cost_baseline.to_numpy(dtype=float)
    paired["relative_difference"] = (
        paired.common_cost_treatment.to_numpy(dtype=float) - baseline
    ) / np.abs(baseline)
    distribution = hierarchical_bootstrap_distribution(
        paired, "relative_difference", samples=samples, seed=seed)
    upper = float(np.quantile(distribution, 0.95))
    return pd.DataFrame([{
        "margin_pct": 100.0 * margin,
        "mean_relative_difference_pct": float(
            100.0 * paired.relative_difference.mean()),
        "one_sided_upper_95_pct": 100.0 * upper,
        "noninferior": bool(upper < margin),
        "n_pairs": len(paired),
    }])


def beta_ablation_comparisons(
    shifted: pd.DataFrame,
    beta_zero: pd.DataFrame,
    *,
    samples: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    beta_zero = beta_zero.copy()
    beta_zero["algorithm"] = "eact_beta0"
    final = shifted[shifted.algorithm == "eact_mpc"].copy()
    final["algorithm"] = "eact_beta010"
    static = shifted[shifted.algorithm == "static_robust_mpc"].copy()
    combined = pd.concat([beta_zero, final, static], ignore_index=True)
    cost = seasonal_cost_comparisons(
        combined,
        treatment="eact_beta010",
        baselines=("eact_beta0", "static_robust_mpc"),
        samples=samples,
    )
    constraints = seasonal_constraint_comparisons(
        combined,
        treatment="eact_beta010",
        baseline="eact_beta0",
        samples=samples,
    )
    return cost, constraints


def weight_robustness(
    episodes: pd.DataFrame,
    *,
    samples: int = 20_000,
) -> pd.DataFrame:
    rows = []
    group_columns = ["forecast_stress", "forecast_stress_scale", *WEIGHT_COLUMNS]
    for config, group in episodes.groupby(group_columns, sort=True):
        if set(group.algorithm.unique()) != {"static_robust_mpc", "eact_mpc"}:
            raise ValueError(f"Incomplete weight-sensitivity controllers: {config}")
        cost = seasonal_cost_comparisons(
            group, baselines=("static_robust_mpc",), samples=samples)
        constraints = seasonal_constraint_comparisons(group, samples=samples)
        common = cost[cost.metric == "common_cost"].iloc[0]
        margin = constraints[
            constraints.metric == "thermal_margin_violation_hours"].iloc[0]
        row = dict(zip(group_columns, config))
        row.update({
            "n_pairs": int(common.n_pairs),
            "common_cost_improvement_pct": float(
                common.mean_relative_improvement_pct),
            "common_cost_ci95_low_pct": float(common.ci95_low_pct),
            "common_cost_ci95_high_pct": float(common.ci95_high_pct),
            "thermal_margin_hours_reduction": float(margin.mean_reduction),
            "thermal_margin_ci95_low": float(margin.ci95_low),
            "thermal_margin_ci95_high": float(margin.ci95_high),
        })
        for metric in ("e_grid_mwh", "co2_kg", "e_total_mwh"):
            metric_row = cost[cost.metric == metric].iloc[0]
            row[f"{metric}_improvement_pct"] = float(
                metric_row.mean_relative_improvement_pct)
        rows.append(row)
    return pd.DataFrame(rows)


def assemble_weight_sensitivity(
    nonprimary: pd.DataFrame,
    no_shift: pd.DataFrame,
    shifted: pd.DataFrame,
) -> pd.DataFrame:
    primary_mask = np.logical_and.reduce([
        np.isclose(nonprimary[column], expected)
        for column, expected in zip(WEIGHT_COLUMNS, PRIMARY_WEIGHTS)
    ])
    nonprimary = nonprimary.loc[~primary_mask].copy()
    primary = pd.concat([no_shift, shifted], ignore_index=True)
    primary = primary[
        primary.country.isin(("CHN", "JPN", "NOR"))
        & primary.algorithm.isin(("static_robust_mpc", "eact_mpc"))
    ].copy()
    combined = pd.concat([primary, nonprimary], ignore_index=True)
    key = [
        "country", "algorithm", "start_hour",
        "forecast_stress", "forecast_stress_scale", *WEIGHT_COLUMNS,
    ]
    if combined.duplicated(key).any():
        raise ValueError("Duplicate objective-weight sensitivity rows")
    groups = combined.groupby([
        "forecast_stress", "forecast_stress_scale", *WEIGHT_COLUMNS,
        "algorithm",
    ]).size()
    if len(groups) != 20 or not np.all(groups.to_numpy() == 12):
        raise ValueError(
            "Weight sensitivity requires five settings, two scenarios, "
            "two controllers, and 12 rows per treatment")
    return combined


def tail_risk_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = _with_common_cost(episodes)
    cost = _paired(frame, "eact_mpc", "static_robust_mpc", "common_cost")
    baseline = cost.common_cost_baseline.to_numpy(dtype=float)
    cost["effect"] = (
        baseline - cost.common_cost_treatment.to_numpy(dtype=float)
    ) / np.abs(baseline)
    margin = _paired(
        episodes, "eact_mpc", "static_robust_mpc",
        "thermal_margin_violation_hours")
    margin["effect"] = (
        margin.thermal_margin_violation_hours_baseline
        - margin.thermal_margin_violation_hours_treatment)
    rows = []
    for metric, paired, scale in (
        ("common_cost_relative_improvement", cost, 100.0),
        ("thermal_margin_hours_reduction", margin, 1.0),
    ):
        values = np.sort(paired.effect.to_numpy(dtype=float))
        tail_count = max(1, int(np.ceil(0.20 * len(values))))
        rows.append({
            "metric": metric,
            "n_pairs": len(values),
            "worst_case": float(scale * values[0]),
            "worst_20pct_cvar": float(scale * values[:tail_count].mean()),
            "median": float(scale * np.median(values)),
            "best_case": float(scale * values[-1]),
        })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasonal-root", default=DEFAULT_SEASONAL_ROOT)
    parser.add_argument("--beta-root", default=DEFAULT_BETA_ROOT)
    parser.add_argument("--weight-root", default=DEFAULT_WEIGHT_ROOT)
    parser.add_argument("--annual-root", default=DEFAULT_ANNUAL_ROOT)
    parser.add_argument(
        "--forecast-eval-root", default=DEFAULT_FORECAST_EVAL_ROOT)
    parser.add_argument("--e6-eact-root", default=DEFAULT_E6_EACT_ROOT)
    parser.add_argument(
        "--e6-baseline-roots", nargs="+",
        default=list(DEFAULT_E6_BASELINE_ROOTS))
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    e1 = load_outputs(args.seasonal_root, "episodes", stage="e1_no_shift")
    e2 = load_outputs(args.seasonal_root, "episodes", stage="e2_shift")
    e1_solver = load_outputs(
        args.seasonal_root, "solver", stage="e1_no_shift")
    e2_solver = load_outputs(
        args.seasonal_root, "solver", stage="e2_shift")
    validate_configuration(
        e1,
        algorithms=("nominal_causal_mpc", "static_robust_mpc", "eact_mpc"),
        stress="none", stress_scale=0.0, beta_floor=0.10, shield=False)
    validate_configuration(
        e2,
        algorithms=("nominal_causal_mpc", "static_robust_mpc", "eact_mpc"),
        stress="adverse_bias", stress_scale=1.0,
        beta_floor=0.10, shield=False)
    forecast_summary = load_outputs(
        args.forecast_eval_root, "summary")
    validate_forecast_evaluation(forecast_summary)

    beta_zero = load_outputs(args.beta_root, "episodes")
    validate_configuration(
        beta_zero,
        algorithms=("eact_mpc",),
        stress="adverse_bias", stress_scale=1.0,
        beta_floor=0.0, shield=False)
    nonprimary_weights = load_outputs(args.weight_root, "episodes")
    weights = assemble_weight_sensitivity(nonprimary_weights, e1, e2)
    beta_cost, beta_constraints = beta_ablation_comparisons(
        e2, beta_zero, samples=args.bootstrap_samples)

    annual = load_outputs(args.annual_root, "weekly")
    annual_solver = load_outputs(args.annual_root, "solver")
    annual_episodes = load_outputs(args.annual_root, "episodes")
    annual_hourly = load_outputs(args.annual_root, "hourly")
    validate_annual_configuration(annual)
    e6 = load_e6_boundary_episodes(
        args.e6_eact_root, args.e6_baseline_roots)

    outputs = {
        "forecast_calibration.csv": analyze_forecast_calibration(
            forecast_summary, samples=args.bootstrap_samples),
        "e1_no_shift_comparisons.csv": seasonal_cost_comparisons(
            e1, samples=args.bootstrap_samples),
        "e1_no_shift_constraint_comparisons.csv":
            seasonal_constraint_comparisons(
                e1, samples=args.bootstrap_samples),
        "e1_noninferiority.csv": seasonal_noninferiority(
            e1, samples=args.bootstrap_samples),
        "e2_shift_cost_comparisons.csv": seasonal_cost_comparisons(
            e2, samples=args.bootstrap_samples),
        "e2_shift_constraint_comparisons.csv":
            seasonal_constraint_comparisons(
                e2, samples=args.bootstrap_samples),
        "e3_annual_comparisons.csv": annual_comparisons(
            annual, samples=args.bootstrap_samples),
        "e3_annual_noninferiority.csv": annual_noninferiority(
            annual, samples=args.bootstrap_samples),
        "e3_annual_diagnostics.csv": annual_diagnostics(
            annual_episodes, annual_hourly),
        "e4_beta_cost_ablation.csv": beta_cost,
        "e4_beta_constraint_ablation.csv": beta_constraints,
        "e5_weight_robustness.csv": weight_robustness(
            weights, samples=args.bootstrap_samples),
        "e6_boundary_cost_comparisons.csv": stress_cost_comparisons(
            e6, samples=args.bootstrap_samples),
        "e6_boundary_constraint_comparisons.csv":
            stress_constraint_comparisons(
                e6, samples=args.bootstrap_samples),
        "e6_tail_risk.csv": tail_risk_summary(e2),
        "computational_performance.csv": computational_performance({
            "e1_no_shift": e1_solver,
            "e2_adverse_bias_1sigma": e2_solver,
            "e3_annual_no_shift": annual_solver,
        }),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    output_records = {}
    for name, frame in outputs.items():
        path = os.path.join(args.out_dir, name)
        frame.to_csv(path, index=False)
        output_records[name] = {
            "path": os.path.abspath(path),
            "rows": int(len(frame)),
        }
        print(f"saved -> {path}")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_samples": args.bootstrap_samples,
        "roots": {
            "seasonal": os.path.abspath(args.seasonal_root),
            "beta": os.path.abspath(args.beta_root),
            "weight": os.path.abspath(args.weight_root),
            "annual": os.path.abspath(args.annual_root),
            "forecast_evaluation": os.path.abspath(
                args.forecast_eval_root),
            "e6_eact": os.path.abspath(args.e6_eact_root),
            "e6_baselines": [
                os.path.abspath(root) for root in args.e6_baseline_roots],
        },
        "outputs": output_records,
    }
    manifest_path = os.path.join(
        args.out_dir, "final_evidence_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {manifest_path}")


if __name__ == "__main__":
    main()
