import numpy as np
import pandas as pd

from coastaldc_env import COUNTRIES
from scripts.analyze_eact_final_paper import (
    annual_diagnostics,
    assemble_weight_sensitivity,
    computational_performance,
    seasonal_constraint_comparisons,
    seasonal_cost_comparisons,
    seasonal_noninferiority,
    tail_risk_summary,
    validate_annual_configuration,
    validate_configuration,
    validate_e6_boundary_configuration,
    validate_forecast_evaluation,
    weight_robustness,
)


def _episodes(
    *,
    eact_cost: float = 9.0,
    static_cost: float = 10.0,
    nominal_cost: float = 12.0,
    stress: str = "adverse_bias",
    scale: float = 1.0,
    weights=(1.0, 2.0, 0.2, 0.5),
) -> pd.DataFrame:
    rows = []
    costs = {
        "nominal_causal_mpc": nominal_cost,
        "static_robust_mpc": static_cost,
        "eact_mpc": eact_cost,
    }
    for country in ("CHN", "JPN", "NOR"):
        for start in (336, 2496, 4680, 6888):
            for algorithm, cost in costs.items():
                margin = 1.0 if algorithm == "eact_mpc" else 3.0
                rows.append({
                    "country": country,
                    "algorithm": algorithm,
                    "start_hour": start,
                    "episode_return": -cost,
                    "e_grid_mwh": cost,
                    "co2_kg": 100.0 * cost,
                    "e_total_mwh": 2.0 * cost,
                    "thermal_margin_violation_hours": margin,
                    "thermal_margin_exceedance_kh": margin * 1e-4,
                    "safety_infeasible_hours": margin,
                    "thermal_violation_hours": 0.0,
                    "sla_violation_mwh": 0.0,
                    "terminal_unserved_mwh": 0.0,
                    "forecast_stress": stress,
                    "forecast_stress_scale": scale,
                    "thermal_safety_shield": False,
                    "adaptive_beta_floor": (
                        0.1 if algorithm == "eact_mpc" else 0.0),
                    "weight_grid": weights[0],
                    "weight_co2": weights[1],
                    "weight_total": weights[2],
                    "weight_smooth": weights[3],
                })
    return pd.DataFrame(rows)


def test_final_seasonal_cost_constraint_and_noninferiority_statistics():
    episodes = _episodes()

    cost = seasonal_cost_comparisons(episodes, samples=100)
    constraints = seasonal_constraint_comparisons(episodes, samples=100)
    noninferiority = seasonal_noninferiority(episodes, samples=100)

    static_cost = cost[
        (cost.baseline == "static_robust_mpc")
        & (cost.metric == "common_cost")
    ].iloc[0]
    margin = constraints[
        constraints.metric == "thermal_margin_violation_hours"].iloc[0]
    assert np.isclose(static_cost.mean_relative_improvement_pct, 10.0)
    assert margin.mean_reduction == 2.0
    assert bool(noninferiority.iloc[0].noninferior)


def test_configuration_validation_rejects_mixed_weights():
    episodes = _episodes()
    episodes.loc[0, "weight_co2"] = 4.0

    try:
        validate_configuration(
            episodes,
            algorithms=(
                "nominal_causal_mpc", "static_robust_mpc", "eact_mpc"),
            stress="adverse_bias",
            stress_scale=1.0,
            beta_floor=0.1,
            shield=False,
        )
    except ValueError as error:
        assert "weight_co2" in str(error)
    else:
        raise AssertionError("Mixed objective weights must be rejected")


def test_annual_validation_rejects_mixed_shield_states():
    rows = []
    for country in ("CHN", "JPN", "NOR"):
        for algorithm in (
            "nominal_causal_mpc", "static_robust_mpc", "eact_mpc"
        ):
            for week in range(52):
                rows.append({
                    "country": country,
                    "algorithm": algorithm,
                    "week": week,
                    "start_step": week * 168,
                    "reward": -1.0,
                    "mpc_horizon": 24,
                    "confidence": 0.90,
                    "intervention_weight": 0.0,
                    "forecast_stress": "none",
                    "forecast_stress_scale": 0.0,
                    "thermal_safety_shield": True,
                    "adaptive_beta_floor": (
                        0.10 if algorithm == "eact_mpc" else 0.0),
                    "weight_grid": 1.0,
                    "weight_co2": 2.0,
                    "weight_total": 0.2,
                    "weight_smooth": 0.5,
                })
    weekly = pd.DataFrame(rows)
    validate_annual_configuration(weekly)
    weekly.loc[0, "thermal_safety_shield"] = False

    try:
        validate_annual_configuration(weekly)
    except ValueError as error:
        assert "safety shield" in str(error)
    else:
        raise AssertionError("Mixed annual shield states must be rejected")


def test_annual_diagnostics_separates_hard_events_and_safety_projections():
    episodes = pd.DataFrame([
        {
            "country": "JPN",
            "algorithm": algorithm,
            "thermal_margin_violation_hours": 0.0,
            "thermal_margin_exceedance_kh": 0.0,
            "thermal_violation_hours": 0.0,
            "sla_violation_mwh": 0.0,
            "terminal_unserved_mwh": 0.0,
            "final_backlog_mwh": 0.0,
            "safety_interventions": events,
            "safety_infeasible_hours": 0.0,
            "workload_infeasible_hours": 0.0,
        }
        for algorithm, events in (
            ("eact_mpc", 1.0),
            ("static_robust_mpc", 2.0),
        )
    ])
    hourly = pd.DataFrame([
        {
            "country": "JPN",
            "algorithm": algorithm,
            "safety_intervention": intervention,
            "thermal_safety_override": override,
        }
        for algorithm, intervention, override in (
            ("eact_mpc", True, 1e-4),
            ("eact_mpc", False, 0.0),
            ("static_robust_mpc", True, 2e-4),
            ("static_robust_mpc", True, 4e-4),
        )
    ])

    result = annual_diagnostics(episodes, hourly).set_index("algorithm")

    assert result.loc["eact_mpc", "thermal_violation_hours"] == 0.0
    assert result.loc["eact_mpc", "safety_interventions"] == 1.0
    assert np.isclose(
        result.loc[
            "static_robust_mpc",
            "conditional_mean_thermal_safety_override",
        ],
        3e-4,
    )


def test_forecast_validation_requires_final_beta_and_complete_matrix():
    rows = []
    for country in COUNTRIES:
        for mode in ("nominal", "static", "adaptive"):
            for column in (
                "fixed_load_mw", "flexible_arrival_mw", "sst_c",
                "wind_mw", "carbon_kg_per_mwh",
            ):
                rows.append({
                    "country": country,
                    "mode": mode,
                    "column": column,
                    "n_forecasts": 100,
                    "coverage": np.nan if mode == "nominal" else 0.90,
                    "adaptive_beta_floor": 0.10,
                })
    summary = pd.DataFrame(rows)
    validate_forecast_evaluation(summary)
    summary.loc[
        summary["mode"] == "adaptive", "adaptive_beta_floor"] = 0.0

    try:
        validate_forecast_evaluation(summary)
    except ValueError as error:
        assert "beta floor" in str(error)
    else:
        raise AssertionError("Old-beta forecast evaluation must be rejected")


def test_weight_robustness_keeps_configs_separate():
    primary = _episodes()[lambda x: x.algorithm != "nominal_causal_mpc"]
    high_carbon = _episodes(weights=(1.0, 4.0, 0.2, 0.5))
    high_carbon = high_carbon[
        high_carbon.algorithm != "nominal_causal_mpc"]

    result = weight_robustness(
        pd.concat([primary, high_carbon], ignore_index=True), samples=100)

    assert len(result) == 2
    assert set(result.weight_co2) == {2.0, 4.0}
    assert np.allclose(result.common_cost_improvement_pct, 10.0)


def test_weight_sensitivity_reuses_primary_rows_without_duplicates():
    e1 = _episodes(stress="none", scale=0.0)
    e2 = _episodes(stress="adverse_bias", scale=1.0)
    nonprimary = []
    for weights in (
        (1.0, 1.0, 0.2, 0.5),
        (1.0, 4.0, 0.2, 0.5),
        (1.0, 2.0, 0.1, 0.5),
        (1.0, 2.0, 0.5, 0.5),
    ):
        for stress, scale in (("none", 0.0), ("adverse_bias", 1.0)):
            frame = _episodes(stress=stress, scale=scale, weights=weights)
            nonprimary.append(
                frame[frame.algorithm != "nominal_causal_mpc"])

    combined = assemble_weight_sensitivity(
        pd.concat(nonprimary, ignore_index=True), e1, e2)
    result = weight_robustness(combined, samples=100)

    assert len(combined) == 240
    assert len(result) == 10
    assert set(result.weight_co2) == {1.0, 2.0, 4.0}


def test_e6_validation_requires_all_six_final_beta_conditions():
    frames = []
    for stress, scale in (
        ("none", 0.0),
        ("adverse_bias", 0.5),
        ("adverse_bias", 1.0),
        ("adverse_bias", 2.0),
        ("noise", 1.0),
        ("combined", 1.0),
    ):
        frames.append(_episodes(stress=stress, scale=scale))
    episodes = pd.concat(frames, ignore_index=True)
    validate_e6_boundary_configuration(episodes)
    incomplete = episodes[
        ~(
            (episodes.forecast_stress == "combined")
            & np.isclose(episodes.forecast_stress_scale, 1.0)
        )
    ]

    try:
        validate_e6_boundary_configuration(incomplete)
    except ValueError as error:
        assert "conditions" in str(error)
    else:
        raise AssertionError("Incomplete E6 conditions must be rejected")


def test_computational_performance_separates_acceptance_and_convergence():
    solver = pd.DataFrame({
        "algorithm": ["eact_mpc"] * 3,
        "accepted": [True, True, False],
        "solver_success": [True, False, False],
        "fallback": ["none", "none", "shifted_plan"],
        "min_constraint": [0.0, -1e-5, -2e-4],
        "solve_time_s": [0.2, 0.4, 0.6],
    })

    result = computational_performance({"fixture": solver}).iloc[0]

    assert np.isclose(result.accepted_plan_rate, 2.0 / 3.0)
    assert np.isclose(result.solver_convergence_rate, 1.0 / 3.0)
    assert np.isclose(result.fallback_rate, 1.0 / 3.0)
    assert result.shifted_plan_fallbacks == 1
    assert result.min_constraint == -2e-4


def test_tail_risk_uses_worst_twenty_percent():
    episodes = _episodes()
    starts = sorted(episodes.start_hour.unique())
    eact_mask = episodes.algorithm == "eact_mpc"
    for index, start in enumerate(starts):
        episodes.loc[
            eact_mask & (episodes.start_hour == start),
            "episode_return",
        ] = -(8.0 + index)

    result = tail_risk_summary(episodes)

    common = result[
        result.metric == "common_cost_relative_improvement"].iloc[0]
    assert common["worst_case"] < common["median"]
    assert common["worst_20pct_cvar"] <= common["median"]
