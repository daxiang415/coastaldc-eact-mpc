import numpy as np
import pandas as pd
import pytest

from scripts.analyze_eact_paper_evidence import (
    analyze_forecast_calibration,
    annual_comparisons,
    annual_noninferiority,
    holm_adjust,
    moving_block_bootstrap_distribution,
    seasonal_comparisons,
    select_intervention_weight,
)


def test_holm_adjust_matches_step_down_definition():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_forecast_calibration_pairs_mode_column_without_attribute_collision():
    rows = []
    for country in ("JPN", "CHN"):
        for column in ("wind_mw", "sst_c"):
            rows.extend([
                {"country": country, "column": column, "mode": "static",
                 "coverage": 0.84},
                {"country": country, "column": column, "mode": "adaptive",
                 "coverage": 0.90},
            ])

    result = analyze_forecast_calibration(
        pd.DataFrame(rows), samples=100, seed=1).iloc[0]

    assert result.n_country_variables == 4
    assert result.adaptive_closer_count == 4
    assert result.adaptive_within_88_92_count == 4


def _seasonal_frame() -> pd.DataFrame:
    rows = []
    algorithms = {
        "no_control": 110.0,
        "rule_based": 107.0,
        "nominal_causal_mpc": 103.0,
        "static_robust_mpc": 100.0,
        "eact_mpc": 100.5,
    }
    starts = [
        (336, "2025-01-15"), (2496, "2025-04-15"),
        (4680, "2025-07-15"), (6888, "2025-10-15"),
    ]
    for country in ("JPN", "CHN"):
        for algorithm, cost in algorithms.items():
            for episode, (start, timestamp) in enumerate(starts):
                rows.append({
                    "country": country,
                    "algorithm": algorithm,
                    "episode": episode,
                    "reset_seed": 5000 + episode,
                    "start_hour": start,
                    "start_timestamp": timestamp,
                    "episode_return": -cost,
                    "e_grid_mwh": cost,
                    "co2_kg": cost,
                    "e_total_mwh": cost,
                })
    return pd.DataFrame(rows)


def test_seasonal_noninferiority_uses_one_percent_margin():
    _, noninferiority = seasonal_comparisons(
        _seasonal_frame(), samples=100, seed=1)

    row = noninferiority.iloc[0]
    assert row.noninferior
    assert row.one_sided_upper_95_pct == pytest.approx(0.5)


def test_weight_selection_prefers_lower_weight_inside_point_one_percent_tie():
    episodes = pd.DataFrame([
        {
            "intervention_weight": weight,
            "episode_return": -cost,
            "sla_violation_mwh": 0.0,
            "terminal_unserved_mwh": 0.0,
            "thermal_violation_hours": 0.0,
        }
        for weight, cost in [(0.0, 100.05), (0.1, 100.0), (0.2, 101.0)]
    ])
    solver = pd.DataFrame([
        {"intervention_weight": weight, "accepted": True, "min_constraint": 0.0}
        for weight in (0.0, 0.1, 0.2)
    ])

    result = select_intervention_weight(episodes, solver)

    assert result.loc[result.selected, "intervention_weight"].item() == 0.0


def test_weight_selection_rejects_incomplete_candidate_set():
    episodes = pd.DataFrame([{
        "intervention_weight": 0.0,
        "episode_return": -100.0,
        "sla_violation_mwh": 0.0,
        "terminal_unserved_mwh": 0.0,
        "thermal_violation_hours": 0.0,
    }])
    solver = pd.DataFrame([{
        "intervention_weight": 0.0, "accepted": True, "min_constraint": 0.0,
    }])

    with pytest.raises(ValueError, match="Incomplete"):
        select_intervention_weight(
            episodes, solver, expected_weights=[0.0, 0.1])


def test_moving_block_bootstrap_rejects_missing_week():
    frame = pd.DataFrame({
        "country": ["JPN", "JPN"],
        "week": [0, 2],
        "value": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="Incomplete"):
        moving_block_bootstrap_distribution(
            frame, "value", samples=10, seed=1)


def test_annual_comparison_rejects_incomplete_series():
    rows = []
    for algorithm in ("eact_mpc", "static_robust_mpc"):
        for week in range(51):
            rows.append({
                "country": "JPN", "algorithm": algorithm,
                "episode": 0, "reset_seed": 5000, "week": week,
                "start_step": week * 168, "n_hours": 168,
                "reward": -100.0, "e_grid_mwh": 100.0,
                "co2_kg": 100.0, "e_total_mwh": 100.0,
            })
    with pytest.raises(ValueError, match="52 complete weeks"):
        annual_comparisons(pd.DataFrame(rows), samples=10, seed=1)


def test_annual_noninferiority_uses_moving_block_upper_limit():
    rows = []
    for country in ("JPN", "CHN"):
        for algorithm, cost in (
                ("static_robust_mpc", 100.0), ("eact_mpc", 100.5)):
            for week in range(52):
                rows.append({
                    "country": country, "algorithm": algorithm,
                    "week": week, "start_step": week * 168,
                    "reward": -cost,
                })

    result = annual_noninferiority(
        pd.DataFrame(rows), samples=100, seed=1).iloc[0]

    assert result.noninferior
    assert result.one_sided_upper_95_pct == pytest.approx(0.5)
