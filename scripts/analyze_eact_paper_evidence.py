"""Analyze the preregistered EACT experiment matrix for paper reporting."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import linregress, wilcoxon


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_EXPERIMENT_ROOT = os.path.join(ROOT, "results", "eact_paper_v1")
METRICS = ("common_cost", "e_grid_mwh", "co2_kg", "e_total_mwh")
BASELINES = (
    "no_control", "rule_based", "nominal_causal_mpc", "static_robust_mpc")


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Holm correction requires finite one-dimensional p-values")
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_bootstrap_mean_ci(
    values: np.ndarray,
    *,
    samples: int = 20_000,
    seed: int = 20260717,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be a nonempty finite vector")
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, quantiles)
    return float(low), float(high)


def hierarchical_bootstrap_distribution(
    frame: pd.DataFrame,
    value_col: str,
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> np.ndarray:
    required = {"country", "season", value_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"Hierarchical bootstrap missing columns: {required - set(frame)}")
    countries = sorted(frame["country"].unique())
    if not countries:
        raise ValueError("Hierarchical bootstrap received no countries")
    groups = {
        country: frame.loc[frame.country == country, value_col].to_numpy(dtype=float)
        for country in countries
    }
    if any(values.size == 0 or not np.isfinite(values).all()
           for values in groups.values()):
        raise ValueError("Hierarchical bootstrap groups must be finite and nonempty")
    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(countries, size=len(countries), replace=True)
        country_means = []
        for country in selected:
            values = groups[country]
            country_means.append(float(np.mean(
                rng.choice(values, size=len(values), replace=True))))
        distribution[sample] = float(np.mean(country_means))
    return distribution


def moving_block_bootstrap_distribution(
    frame: pd.DataFrame,
    value_col: str,
    *,
    block_length: int = 4,
    samples: int = 20_000,
    seed: int = 20260717,
) -> np.ndarray:
    required = {"country", "week", value_col}
    if not required.issubset(frame.columns):
        raise ValueError(f"Moving-block bootstrap missing columns: {required - set(frame)}")
    countries = sorted(frame["country"].unique())
    groups = {}
    for country in countries:
        group = frame[frame.country == country].sort_values("week")
        weeks = group["week"].to_numpy(dtype=int)
        if not np.array_equal(weeks, np.arange(len(weeks))):
            raise ValueError(f"Incomplete or noncontiguous weeks for {country}")
        groups[country] = group[value_col].to_numpy(dtype=float)
    if not groups or block_length <= 0:
        raise ValueError("Moving-block bootstrap requires data and a positive block")
    rng = np.random.default_rng(seed)
    distribution = np.empty(samples, dtype=float)
    for sample in range(samples):
        selected = rng.choice(countries, size=len(countries), replace=True)
        country_means = []
        for country in selected:
            values = groups[country]
            sampled = []
            while len(sampled) < len(values):
                start = int(rng.integers(0, len(values)))
                sampled.extend(
                    values[(start + offset) % len(values)]
                    for offset in range(block_length))
            country_means.append(float(np.mean(sampled[:len(values)])))
        distribution[sample] = float(np.mean(country_means))
    return distribution


def safe_wilcoxon(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Wilcoxon differences must be finite and nonempty")
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, alternative=alternative).pvalue)


def _load_stage(root: str, stage: str, kind: str) -> pd.DataFrame:
    pattern = os.path.join(root, stage, "**", f"{kind}_*.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"No {kind} outputs found for stage {stage}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = os.path.abspath(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _with_common_cost(frame: pd.DataFrame, reward_col: str) -> pd.DataFrame:
    result = frame.copy()
    result["common_cost"] = -result[reward_col].astype(float)
    return result


def _paired_metric(
    frame: pd.DataFrame,
    treatment: str,
    baseline: str,
    metric: str,
    keys: list[str],
) -> pd.DataFrame:
    columns = [*keys, metric]
    treatment_df = frame.loc[frame.algorithm == treatment, columns]
    baseline_df = frame.loc[frame.algorithm == baseline, columns]
    if treatment_df.duplicated(keys).any() or baseline_df.duplicated(keys).any():
        raise ValueError(f"Duplicate pairing rows: {treatment} vs {baseline}")
    paired = treatment_df.merge(
        baseline_df, on=keys, suffixes=("_treatment", "_baseline"),
        validate="one_to_one")
    if len(paired) != len(treatment_df) or len(paired) != len(baseline_df):
        raise ValueError(f"Incomplete pairing: {treatment} vs {baseline}")
    treatment_values = paired[f"{metric}_treatment"].to_numpy(dtype=float)
    baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
    if np.any(np.abs(baseline_values) <= 1e-12):
        raise ValueError(f"Relative comparison has zero baseline for {metric}")
    paired["relative_improvement"] = (
        baseline_values - treatment_values) / np.abs(baseline_values)
    paired["relative_difference"] = (
        treatment_values - baseline_values) / np.abs(baseline_values)
    return paired


def analyze_forecast_calibration(
    summary: pd.DataFrame,
    *,
    target: float = 0.90,
    samples: int = 20_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    bounded = summary[summary["mode"].isin(["static", "adaptive"])].copy()
    bounded["coverage_error"] = np.abs(bounded["coverage"] - target)
    keys = ["country", "column"]
    static = bounded[bounded["mode"] == "static"][
        keys + ["coverage", "coverage_error"]]
    adaptive = bounded[bounded["mode"] == "adaptive"][
        keys + ["coverage", "coverage_error"]]
    paired = adaptive.merge(
        static, on=keys, suffixes=("_adaptive", "_static"),
        validate="one_to_one")
    if len(paired) != len(adaptive) or len(paired) != len(static):
        raise ValueError("Forecast calibration rows are not fully paired")
    differences = (
        paired.coverage_error_adaptive - paired.coverage_error_static).to_numpy()
    ci_low, ci_high = paired_bootstrap_mean_ci(
        differences, samples=samples, seed=seed)
    return pd.DataFrame([{
        "n_country_variables": len(paired),
        "target_coverage": target,
        "adaptive_mean_abs_coverage_error": float(
            paired.coverage_error_adaptive.mean()),
        "static_mean_abs_coverage_error": float(
            paired.coverage_error_static.mean()),
        "adaptive_minus_static_error": float(differences.mean()),
        "difference_ci95_low": ci_low,
        "difference_ci95_high": ci_high,
        "wilcoxon_p_value": safe_wilcoxon(differences, alternative="less"),
        "adaptive_closer_count": int(np.sum(differences < 0.0)),
        "adaptive_within_88_92_count": int(np.sum(
            paired.coverage_adaptive.between(0.88, 0.92))),
    }])


def select_intervention_weight(
    episodes: pd.DataFrame,
    solver: pd.DataFrame,
    *,
    expected_weights: Iterable[float] | None = None,
) -> pd.DataFrame:
    episodes = _with_common_cost(episodes, "episode_return")
    observed_weights = sorted(episodes.intervention_weight.unique())
    if expected_weights is not None:
        expected = sorted(float(value) for value in expected_weights)
        if len(observed_weights) != len(expected) or not np.allclose(
                observed_weights, expected):
            raise ValueError(
                f"Incomplete intervention-weight candidates: "
                f"observed={observed_weights}, expected={expected}")
    pairing_keys = [
        key for key in ("country", "start_timestamp") if key in episodes.columns]
    if pairing_keys:
        signatures = {
            float(weight): set(map(tuple, group[pairing_keys].to_numpy()))
            for weight, group in episodes.groupby("intervention_weight")
        }
        reference = next(iter(signatures.values()))
        if any(signature != reference for signature in signatures.values()):
            raise ValueError("Intervention weights are not paired on country and time")
    rows = []
    for weight, group in episodes.groupby("intervention_weight", sort=True):
        solver_group = solver[np.isclose(solver.intervention_weight, weight)]
        safe = bool(
            np.all(group.sla_violation_mwh <= 1e-9)
            and np.all(group.terminal_unserved_mwh <= 1e-9)
            and np.all(group.thermal_violation_hours <= 1e-9)
            and not solver_group.empty
            and np.all(solver_group.accepted.astype(bool))
            and np.all(solver_group.min_constraint >= -1e-4)
        )
        rows.append({
            "intervention_weight": float(weight),
            "n_episodes": len(group),
            "feasible_and_safe": safe,
            "mean_common_cost": float(group.common_cost.mean()),
            "accepted_plan_rate": float(solver_group.accepted.mean())
            if not solver_group.empty else np.nan,
        })
    result = pd.DataFrame(rows)
    feasible = result[result.feasible_and_safe]
    if feasible.empty:
        raise ValueError("No intervention-weight candidate is feasible and safe")
    best_cost = float(feasible.mean_common_cost.min())
    tied = feasible[feasible.mean_common_cost <= best_cost * 1.001 + 1e-12]
    selected = float(tied.intervention_weight.min())
    result["selected"] = np.isclose(result.intervention_weight, selected)
    return result


def seasonal_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _with_common_cost(episodes, "episode_return")
    frame["season"] = pd.to_datetime(frame.start_timestamp).dt.month.map(
        {1: "winter", 4: "spring", 7: "summer", 10: "autumn"})
    if frame.season.isna().any():
        raise ValueError("Seasonal results contain an unregistered start month")
    expected = frame.groupby(["algorithm", "country"]).size()
    if not np.all(expected.to_numpy() == 4):
        raise ValueError("Every seasonal country-controller pair must have four starts")
    rows = []
    for baseline in BASELINES:
        for metric in METRICS:
            paired = _paired_metric(
                frame, "eact_mpc", baseline, metric,
                ["country", "season", "start_hour"])
            distribution = hierarchical_bootstrap_distribution(
                paired, "relative_improvement", samples=samples, seed=seed)
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
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
    comparisons = pd.DataFrame(rows)
    mask = (
        (comparisons.baseline == "nominal_causal_mpc")
        & comparisons.metric.isin(["e_grid_mwh", "co2_kg", "e_total_mwh"]))
    comparisons["holm_p_value"] = comparisons.wilcoxon_p_value
    comparisons.loc[mask, "holm_p_value"] = holm_adjust(
        comparisons.loc[mask, "wilcoxon_p_value"])

    paired = _paired_metric(
        frame, "eact_mpc", "static_robust_mpc", "common_cost",
        ["country", "season", "start_hour"])
    distribution = hierarchical_bootstrap_distribution(
        paired, "relative_difference", samples=samples, seed=seed + 1)
    upper = float(np.quantile(distribution, 0.95))
    noninferiority = pd.DataFrame([{
        "comparison": "eact_mpc_vs_static_robust_mpc",
        "margin_pct": 1.0,
        "mean_relative_difference_pct": float(
            100.0 * paired.relative_difference.mean()),
        "one_sided_upper_95_pct": 100.0 * upper,
        "noninferior": bool(upper < 0.01),
        "n_pairs": len(paired),
    }])
    return comparisons, noninferiority


def annual_comparisons(
    weekly: pd.DataFrame,
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    frame = _with_common_cost(weekly, "reward")
    frame = frame[frame.algorithm.isin(("eact_mpc", *BASELINES))].copy()
    complete = frame.groupby(["country", "algorithm"])["week"].nunique()
    if not np.all(complete.to_numpy() == 52):
        raise ValueError("Annual comparison requires 52 complete weeks per series")
    rows = []
    for baseline in BASELINES:
        baseline_countries = set(frame.loc[frame.algorithm == baseline, "country"])
        if not baseline_countries:
            continue
        subset = frame[frame.country.isin(baseline_countries)]
        for metric in METRICS:
            paired = _paired_metric(
                subset, "eact_mpc", baseline, metric,
                ["country", "week", "start_step"])
            distribution = moving_block_bootstrap_distribution(
                paired, "relative_improvement", samples=samples, seed=seed)
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
                "baseline": baseline,
                "metric": metric,
                "countries": ",".join(sorted(baseline_countries)),
                "n_weeks": len(paired),
                "mean_relative_improvement_pct": float(
                    100.0 * paired.relative_improvement.mean()),
                "moving_block_ci95_low_pct": float(100.0 * low),
                "moving_block_ci95_high_pct": float(100.0 * high),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.relative_improvement.to_numpy()),
            })
    result = pd.DataFrame(rows)
    result["holm_p_value"] = result.wilcoxon_p_value
    mask = (
        (result.baseline == "nominal_causal_mpc")
        & result.metric.isin(["e_grid_mwh", "co2_kg", "e_total_mwh"]))
    result.loc[mask, "holm_p_value"] = holm_adjust(
        result.loc[mask, "wilcoxon_p_value"])
    return result


def annual_noninferiority(
    weekly: pd.DataFrame,
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    frame = _with_common_cost(weekly, "reward")
    paired = _paired_metric(
        frame, "eact_mpc", "static_robust_mpc", "common_cost",
        ["country", "week", "start_step"])
    distribution = moving_block_bootstrap_distribution(
        paired, "relative_difference", samples=samples, seed=seed)
    upper = float(np.quantile(distribution, 0.95))
    return pd.DataFrame([{
        "comparison": "eact_mpc_vs_static_robust_mpc",
        "margin_pct": 1.0,
        "mean_relative_difference_pct": float(
            100.0 * paired.relative_difference.mean()),
        "one_sided_upper_95_pct": 100.0 * upper,
        "noninferior": bool(upper < 0.01),
        "n_weeks": len(paired),
        "block_length_weeks": 4,
    }])


def ablation_comparisons(
    episodes: pd.DataFrame,
    *,
    samples: int = 20_000,
    seed: int = 20260717,
) -> pd.DataFrame:
    frame = _with_common_cost(episodes, "episode_return")
    frame["season"] = pd.to_datetime(frame.start_timestamp).dt.month.astype(str)
    rows = []
    baselines = ["static_robust_mpc", "eact_no_bias"]
    if "eact_no_intervention" in set(frame.algorithm):
        baselines.append("eact_no_intervention")
    for baseline in baselines:
        for metric in METRICS:
            paired = _paired_metric(
                frame, "eact_mpc", baseline, metric,
                ["country", "season", "start_hour"])
            distribution = hierarchical_bootstrap_distribution(
                paired, "relative_improvement", samples=samples, seed=seed)
            low, high = np.quantile(distribution, [0.025, 0.975])
            rows.append({
                "baseline": baseline,
                "metric": metric,
                "mean_relative_improvement_pct": float(
                    100.0 * paired.relative_improvement.mean()),
                "ci95_low_pct": float(100.0 * low),
                "ci95_high_pct": float(100.0 * high),
                "wilcoxon_p_value": safe_wilcoxon(
                    paired.relative_improvement.to_numpy()),
                "comparison_status": "estimated",
            })
    if ("eact_no_intervention" not in set(frame.algorithm)
            and np.allclose(frame.intervention_weight, 0.0)):
        rows.extend({
            "baseline": "eact_no_intervention",
            "metric": metric,
            "mean_relative_improvement_pct": 0.0,
            "ci95_low_pct": 0.0,
            "ci95_high_pct": 0.0,
            "wilcoxon_p_value": 1.0,
            "comparison_status": "structurally_identical_at_zero_weight",
        } for metric in METRICS)
    return pd.DataFrame(rows)


def sensitivity_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    frame = _with_common_cost(episodes, "episode_return")
    return frame.groupby(
        ["mpc_horizon", "confidence", "algorithm"], as_index=False
    )[[*METRICS, "sla_violation_mwh", "thermal_violation_hours"]].mean()


def high_error_analysis(
    annual_weekly: pd.DataFrame,
    weekly_errors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _with_common_cost(annual_weekly, "reward")
    paired = _paired_metric(
        frame, "eact_mpc", "static_robust_mpc", "common_cost",
        ["country", "week", "start_step"])
    merged = paired.merge(
        weekly_errors[["country", "week", "normalized_mae"]],
        on=["country", "week"], validate="one_to_one")
    if len(merged) != len(paired):
        raise ValueError("Weekly forecast errors do not fully pair with annual results")
    labels = ["low", "medium", "high"]
    merged["error_tertile"] = merged.groupby("country")["normalized_mae"].transform(
        lambda values: pd.qcut(values.rank(method="first"), 3, labels=labels))
    tertiles = merged.groupby("error_tertile", observed=True).agg(
        n_weeks=("relative_improvement", "size"),
        mean_normalized_mae=("normalized_mae", "mean"),
        mean_eact_improvement_pct=("relative_improvement", lambda x: 100.0 * x.mean()),
    ).reset_index()
    regression = linregress(
        merged.normalized_mae.to_numpy(),
        100.0 * merged.relative_improvement.to_numpy())
    slope = pd.DataFrame([{
        "n_weeks": len(merged),
        "slope_pct_per_normalized_mae": float(regression.slope),
        "intercept_pct": float(regression.intercept),
        "r_value": float(regression.rvalue),
        "p_value": float(regression.pvalue),
        "stderr": float(regression.stderr),
    }])
    return tertiles, slope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--forecast-summary", default=os.path.join(
        ROOT, "results", "causal_forecast_evaluation",
        "summary_all15_2025_v3.csv"))
    parser.add_argument("--weekly-forecast-errors", default=os.path.join(
        DEFAULT_EXPERIMENT_ROOT, "forecast_error",
        "weekly_forecast_error_annual_2025.csv"))
    parser.add_argument("--sections", nargs="+", choices=[
        "all", "forecast", "calibration", "seasonal", "annual", "ablation",
        "sensitivity", "mechanism"], default=["all"])
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--out-dir", default=os.path.join(
        DEFAULT_EXPERIMENT_ROOT, "analysis"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    sections = set(args.sections)
    if "all" in sections:
        sections = {
            "forecast", "calibration", "seasonal", "annual", "ablation",
            "sensitivity", "mechanism"}
    os.makedirs(args.out_dir, exist_ok=True)
    outputs: dict[str, pd.DataFrame] = {}

    if "forecast" in sections:
        outputs["forecast_calibration"] = analyze_forecast_calibration(
            pd.read_csv(args.forecast_summary),
            samples=args.bootstrap_samples, seed=args.seed)
    if "calibration" in sections:
        outputs["intervention_weight_selection"] = select_intervention_weight(
            _load_stage(args.experiment_root, "calibration", "episodes"),
            _load_stage(args.experiment_root, "calibration", "solver"),
            expected_weights=[0.0, 0.05, 0.10, 0.20])
    if "seasonal" in sections:
        comparisons, noninferiority = seasonal_comparisons(
            _load_stage(args.experiment_root, "seasonal", "episodes"),
            samples=args.bootstrap_samples, seed=args.seed)
        outputs["seasonal_comparisons"] = comparisons
        outputs["seasonal_noninferiority"] = noninferiority
    if "annual" in sections:
        annual_weekly = _load_stage(args.experiment_root, "annual", "weekly")
        outputs["annual_comparisons"] = annual_comparisons(
            annual_weekly,
            samples=args.bootstrap_samples, seed=args.seed)
        outputs["annual_noninferiority"] = annual_noninferiority(
            annual_weekly,
            samples=args.bootstrap_samples, seed=args.seed + 1)
    if "ablation" in sections:
        outputs["ablation_comparisons"] = ablation_comparisons(
            _load_stage(args.experiment_root, "ablation", "episodes"),
            samples=args.bootstrap_samples, seed=args.seed)
    if "sensitivity" in sections:
        outputs["sensitivity_summary"] = sensitivity_summary(
            _load_stage(args.experiment_root, "sensitivity", "episodes"))
    if "mechanism" in sections:
        tertiles, slope = high_error_analysis(
            _load_stage(args.experiment_root, "annual", "weekly"),
            pd.read_csv(args.weekly_forecast_errors))
        outputs["high_error_tertiles"] = tertiles
        outputs["high_error_slope"] = slope

    for name, frame in outputs.items():
        path = os.path.join(args.out_dir, f"{name}.csv")
        frame.to_csv(path, index=False)
        print(f"saved -> {path}")


if __name__ == "__main__":
    main()
