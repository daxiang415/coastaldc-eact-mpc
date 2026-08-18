import pandas as pd

from scripts.analyze_eact_forecast_stress import (
    paired_comparisons,
    paired_safety_comparisons,
)


def _episodes() -> pd.DataFrame:
    rows = []
    for country in ("CHN", "JPN", "NOR"):
        for start_hour in (0, 168):
            for algorithm, cost, infeasible in (
                ("nominal_causal_mpc", 12.0, 4.0),
                ("static_robust_mpc", 10.0, 3.0),
                ("eact_mpc", 9.0, 1.0),
            ):
                rows.append({
                    "country": country,
                    "algorithm": algorithm,
                    "start_hour": start_hour,
                    "episode_return": -cost,
                    "forecast_stress": "adverse_bias",
                    "forecast_stress_scale": 1.0,
                    "thermal_safety_shield": False,
                    "adaptive_beta_floor": (
                        0.1 if algorithm == "eact_mpc" else 0.0),
                    "weight_grid": 1.0,
                    "weight_co2": 2.0,
                    "weight_total": 0.2,
                    "weight_smooth": 0.5,
                    "safety_infeasible_hours": infeasible,
                    "thermal_margin_violation_hours": infeasible,
                    "thermal_margin_exceedance_kh": infeasible * 1e-4,
                    "sla_violation_mwh": 0.0,
                    "terminal_unserved_mwh": 0.0,
                    "thermal_violation_hours": 0.0,
                    "max_t_room_c": 29.0,
                })
    return pd.DataFrame(rows)


def test_eact_beta_group_reuses_fixed_baselines():
    result = paired_comparisons(_episodes(), samples=100)
    static = result[result.baseline == "static_robust_mpc"].iloc[0]

    assert static.adaptive_beta_floor == 0.1
    assert static.n_pairs == 6
    assert static.mean_relative_improvement_pct == 10.0


def test_paired_safety_comparison_reports_positive_reduction():
    result = paired_safety_comparisons(_episodes(), samples=100)
    row = result[
        (result.baseline == "static_robust_mpc")
        & (result.metric == "safety_infeasible_hours")
    ].iloc[0]

    assert row.n_pairs == 6
    assert row.mean_reduction == 2.0
    assert row.ci95_low == 2.0
    assert row.ci95_high == 2.0
