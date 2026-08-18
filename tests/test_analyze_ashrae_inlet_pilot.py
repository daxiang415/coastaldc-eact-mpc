import numpy as np
import pandas as pd

from scripts.analyze_ashrae_inlet_pilot import (
    ALGORITHMS,
    COUNTRIES,
    START_HOURS,
    cost_comparisons,
    solver_summary,
    thermal_comparisons,
    validate_episode_matrix,
)
from scripts.evaluate_eact_mpc import (
    THERMAL_METRIC_SCHEMA,
    THERMAL_STATE_SEMANTICS,
)


def _episodes() -> pd.DataFrame:
    rows = []
    for scenario, scale in (("none", 0.0), ("adverse_bias", 1.0)):
        for country in sorted(COUNTRIES):
            for start in sorted(START_HOURS):
                for algorithm in sorted(ALGORITHMS):
                    treatment = algorithm == "eact_mpc"
                    rows.append({
                        "country": country,
                        "algorithm": algorithm,
                        "start_hour": start,
                        "episode_return": -100.0 + (1.0 if treatment else 0.0),
                        "e_grid_mwh": 90.0 if treatment else 100.0,
                        "co2_kg": 900.0 if treatment else 1000.0,
                        "e_total_mwh": 190.0 if treatment else 200.0,
                        "forecast_stress": scenario,
                        "forecast_stress_scale": scale,
                        "thermal_safety_shield": False,
                        "adaptive_beta_floor": 0.10,
                        "weight_grid": 1.0,
                        "weight_co2": 2.0,
                        "weight_total": 0.2,
                        "weight_smooth": 0.5,
                        "recommended_exceedance_degc_h": (
                            1.0 if treatment else 2.0),
                        "recommended_exceedance_hours": (
                            1.0 if treatment else 2.0),
                        "recommended_compliance_pct": (
                            99.0 if treatment else 98.0),
                        "p95_t_inlet_c": 26.0 if treatment else 26.5,
                        "p99_t_inlet_c": 26.5 if treatment else 27.0,
                        "max_t_inlet_c": 27.0 if treatment else 27.5,
                        "min_allowable_headroom_c": (
                            5.0 if treatment else 4.5),
                        "allowable_exceedance_hours": 0.0,
                        "allowable_exceedance_degc_h": 0.0,
                        "temporal_rci_hi_pct": 99.0 if treatment else 98.0,
                        "thermal_metric_schema": THERMAL_METRIC_SCHEMA,
                        "thermal_state_semantics": THERMAL_STATE_SEMANTICS,
                        "t_inlet_recommended_min_c": 18.0,
                        "t_inlet_recommended_max_c": 27.0,
                        "t_inlet_allowable_min_c": 15.0,
                        "t_inlet_allowable_max_c": 32.0,
                        "constraint_tolerance": 1e-4,
                    })
    return pd.DataFrame(rows)


def test_fixed_episode_matrix_validation_accepts_complete_schema():
    episodes = _episodes()

    validate_episode_matrix(episodes)

    assert len(episodes) == 72


def test_paired_pilot_comparisons_use_declared_favorable_directions():
    episodes = _episodes()

    costs = cost_comparisons(episodes, samples=100, seed=1)
    thermal = thermal_comparisons(episodes, samples=100, seed=1)

    common = costs[
        (costs.scenario == "adverse_bias")
        & (costs.baseline == "static_robust_mpc")
        & (costs.metric == "common_cost")
    ].iloc[0]
    excess = thermal[
        (thermal.scenario == "adverse_bias")
        & (thermal.baseline == "static_robust_mpc")
        & (thermal.metric == "recommended_exceedance_degc_h")
    ].iloc[0]
    compliance = thermal[
        (thermal.scenario == "adverse_bias")
        & (thermal.baseline == "static_robust_mpc")
        & (thermal.metric == "recommended_compliance_pct")
    ].iloc[0]

    assert np.isclose(common.mean_relative_improvement_pct, 1.0)
    assert excess.mean_improvement == 1.0
    assert compliance.mean_improvement == 1.0
    assert np.isfinite(excess.ci95_low)


def test_solver_summary_separates_rejected_candidate_constraints():
    solver = pd.DataFrame([
        {
            "forecast_stress": "none",
            "algorithm": "eact_mpc",
            "accepted": True,
            "solver_success": True,
            "fallback": "none",
            "solve_time_s": 0.1,
            "min_constraint": -9e-5,
        },
        {
            "forecast_stress": "none",
            "algorithm": "eact_mpc",
            "accepted": False,
            "solver_success": False,
            "fallback": "shifted_plan",
            "solve_time_s": 0.2,
            "min_constraint": -0.5,
        },
    ])

    row = solver_summary(solver).iloc[0]

    assert row.rejected_plan_count == 1
    assert row.shifted_plan_fallbacks == 1
    assert np.isclose(row.min_selected_candidate_constraint, -0.5)
    assert np.isclose(row.min_accepted_constraint, -9e-5)
