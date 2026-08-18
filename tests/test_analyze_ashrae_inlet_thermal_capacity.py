import numpy as np
import pandas as pd

from scripts.analyze_ashrae_inlet_thermal_capacity import (
    COUNTRIES,
    absolute_summary,
    controller_comparisons,
    derating_comparisons,
)


def _episodes():
    rows = []
    for country in COUNTRIES:
        for start_hour in (100, 2200, 4400, 6600):
            for algorithm, common_cost, p95, events in (
                ("static_robust_mpc", 100.0, 28.0, 2.0),
                ("eact_mpc", 101.0, 27.5, 1.0),
            ):
                rows.append({
                    "condition": "adverse_bias_s1.0",
                    "forecast_stress": "adverse_bias",
                    "forecast_stress_scale": 1.0,
                    "cooling_conductance_multiplier": 0.5,
                    "country": country,
                    "start_hour": start_hour,
                    "algorithm": algorithm,
                    "episode_return": -common_cost,
                    "e_grid_mwh": common_cost,
                    "co2_kg": common_cost,
                    "e_total_mwh": common_cost,
                    "e_cooling_mwh": common_cost / 10.0,
                    "e_pump_mwh": common_cost / 20.0,
                    "recommended_exceedance_degc_h": events,
                    "recommended_exceedance_hours": events,
                    "recommended_compliance_pct": 100.0 - events,
                    "p95_t_inlet_c": p95,
                    "p99_t_inlet_c": p95 + 0.1,
                    "max_t_inlet_c": p95 + 0.2,
                    "min_allowable_headroom_c": 32.0 - p95,
                    "allowable_exceedance_hours": 0.0,
                    "allowable_exceedance_degc_h": 0.0,
                    "temporal_rci_hi_pct": 100.0 - events,
                    "sla_violation_mwh": 0.0,
                })
    return pd.DataFrame(rows)


def test_capacity_comparison_preserves_cost_and_thermal_directions():
    result = controller_comparisons(
        _episodes(), samples=100, seed=1)
    common = result[result.metric == "common_cost"].iloc[0]
    p95 = result[result.metric == "p95_t_inlet_c"].iloc[0]
    events = result[
        result.metric == "recommended_exceedance_hours"].iloc[0]

    assert np.isclose(common.mean_relative_improvement_pct, -1.0)
    assert np.isclose(p95.mean_improvement, 0.5)
    assert np.isclose(events.mean_improvement, 1.0)
    assert np.isclose(p95.worst_case_improvement, 0.5)


def test_capacity_absolute_summary_keeps_event_totals():
    result = absolute_summary(_episodes())
    static = result[result.algorithm == "static_robust_mpc"].iloc[0]
    eact = result[result.algorithm == "eact_mpc"].iloc[0]

    assert static.n_weeks == 12
    assert static.recommended_exceedance_hours_total == 24.0
    assert eact.recommended_exceedance_hours_total == 12.0
    assert eact.allowable_exceedance_hours_total == 0.0


def test_derating_comparison_reports_increased_thermal_exposure():
    near = _episodes()
    full = near.copy()
    full["cooling_conductance_multiplier"] = 1.0
    full["recommended_exceedance_degc_h"] = 0.0
    full["recommended_exceedance_hours"] = 0.0
    full["recommended_compliance_pct"] = 100.0
    full["p95_t_inlet_c"] = 26.5
    full["p99_t_inlet_c"] = 26.6
    full["max_t_inlet_c"] = 26.7

    result = derating_comparisons(
        pd.concat([near, full], ignore_index=True),
        samples=100,
        seed=1,
    )
    events = result[
        result.metric == "recommended_exceedance_hours"].iloc[0]
    p95 = result[result.metric == "p95_t_inlet_c"].iloc[0]

    assert events.mean_degradation > 0.0
    assert p95.mean_degradation > 0.0
    assert p95.ci95_low > 0.0
