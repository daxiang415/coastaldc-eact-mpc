import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_eact_mpc import (
    aggregate_thermal_metrics,
    build_parser,
    cooling_config_from_args,
    objective_weights_from_args,
    resolve_start_timestamps,
    summarize,
    thermal_step_metrics,
    validate_start_hours,
    weekly_rows,
)


def test_episode_offset_defaults_to_zero_and_is_configurable():
    parser = build_parser()

    assert parser.parse_args([]).episode_offset == 0
    assert parser.parse_args([]).forecast_dir.endswith(
        "causal_forecasts_v3_gated_bias")
    assert parser.parse_args(["--episode-offset", "4"]).episode_offset == 4
    assert "sac" not in parser.parse_args([]).controllers
    assert parser.parse_args([]).forecast_stress == "none"
    assert parser.parse_args([]).thermal_safety_shield is True
    assert parser.parse_args([]).adaptive_beta_floor == 0.10
    assert parser.parse_args([]).weight_grid == 1.0
    assert parser.parse_args([]).weight_co2 == 2.0
    assert parser.parse_args([]).weight_total == 0.2
    assert parser.parse_args([]).weight_smooth == 0.5
    assert parser.parse_args([]).constraint_tolerance == 1e-4
    assert parser.parse_args([]).cooling_conductance_multiplier == 1.0


def test_objective_weights_are_explicit_and_other_penalties_remain_zero():
    args = build_parser().parse_args([
        "--weight-grid", "0.8",
        "--weight-co2", "3.0",
        "--weight-total", "0.4",
        "--weight-smooth", "0.25",
    ])

    weights = objective_weights_from_args(args)

    assert weights.w_grid == 0.8
    assert weights.w_co2 == 3.0
    assert weights.w_total == 0.4
    assert weights.w_smooth == 0.25
    assert weights.w_sla == 0.0
    assert weights.w_thermal == 0.0
    assert weights.w_pump == 0.0


def test_cooling_conductance_multiplier_changes_available_capacity():
    args = build_parser().parse_args([
        "--cooling-conductance-multiplier", "0.5",
        "--no-thermal-safety-shield",
    ])

    config = cooling_config_from_args(args)

    assert config.conductance_mw_per_k == pytest.approx(1.25)
    assert config.enforce_thermal_safety is False


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf"])
def test_cooling_conductance_multiplier_rejects_nonpositive_or_nonfinite(value):
    args = build_parser().parse_args([
        "--cooling-conductance-multiplier", value])

    with pytest.raises(ValueError, match="finite and positive"):
        cooling_config_from_args(args)


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf"])
def test_objective_weights_reject_negative_or_nonfinite_values(value):
    args = build_parser().parse_args(["--weight-co2", value])

    with pytest.raises(ValueError, match="finite and nonnegative"):
        objective_weights_from_args(args)


def test_fixed_season_timestamps_resolve_to_expected_2025_hours():
    data = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=8760, freq="h")})

    starts = resolve_start_timestamps(
        data,
        ["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15"],
        episode_hours=168,
    )

    assert starts == [336, 2496, 4680, 6888]


def test_explicit_starts_reject_duplicates_and_out_of_range():
    with pytest.raises(ValueError, match="unique"):
        validate_start_hours([10, 10], data_hours=100, episode_hours=24)
    with pytest.raises(ValueError, match="outside"):
        validate_start_hours([77], data_hours=100, episode_hours=24)


def test_weekly_rows_keep_contiguous_blocks_without_reset():
    steps = pd.DataFrame({
        "step": np.arange(336),
        "reward": np.ones(336),
        "e_grid_mwh": np.ones(336),
        "co2_kg": np.ones(336),
        "e_total_mwh": np.ones(336),
        "e_cooling_mwh": np.ones(336),
        "e_pump_mwh": np.ones(336),
        "e_wind_used_mwh": np.ones(336),
        "e_wind_unused_mwh": np.ones(336),
        "sla_violation_mwh": np.zeros(336),
        "t_inlet_c": np.full(336, 24.0),
        "recommended_compliant": np.ones(336),
        "recommended_exceedance_event": np.zeros(336),
        "recommended_exceedance_c": np.zeros(336),
        "allowable_exceedance_event": np.zeros(336),
        "allowable_exceedance_c": np.zeros(336),
        "allowable_headroom_c": np.full(336, 8.0),
        "safety_intervention": np.zeros(336),
        "safety_infeasible": np.zeros(336),
        "workload_intervention": np.zeros(336),
        "thermal_safety_override": np.zeros(336),
        "rate_limit_override": np.zeros(336),
        "action_override": np.zeros(336),
        "backlog_mwh": np.arange(336),
    })
    rows = weekly_rows(
        "JPN", "eact_mpc", 0, 5000,
        {"steps": steps})

    assert len(rows) == 2
    assert rows[0]["n_hours"] == 168
    assert rows[1]["start_step"] == 168
    assert rows[1]["final_backlog_mwh"] == 335


def test_summary_adds_solver_reliability_metrics():
    episodes = pd.DataFrame([{
        "country": "JPN", "algorithm": "eact_mpc", "episode_return": -1.0,
        "e_grid_mwh": 1.0, "co2_kg": 1.0, "e_total_mwh": 1.0,
        "e_cooling_mwh": 1.0, "e_pump_mwh": 1.0,
        "wind_utilization_pct": 50.0, "sla_violation_mwh": 0.0,
        "terminal_unserved_mwh": 0.0, "thermal_violation_hours": 0.0,
        "safety_interventions": 0.0, "safety_infeasible_hours": 0.0,
        "workload_interventions": 0.0,
        "recommended_compliance_pct": 100.0,
        "recommended_exceedance_hours": 0.0,
        "recommended_exceedance_degc_h": 0.0,
        "allowable_exceedance_hours": 0.0,
        "allowable_exceedance_degc_h": 0.0,
        "mean_t_inlet_c": 24.0, "p95_t_inlet_c": 24.0,
        "p99_t_inlet_c": 24.0, "max_t_inlet_c": 24.0,
        "min_allowable_headroom_c": 8.0,
        "temporal_rci_hi_pct": 100.0,
        "mean_abs_action_override": 0.0,
    }])
    solver = pd.DataFrame([
        {"country": "JPN", "algorithm": "eact_mpc", "accepted": True,
         "solver_success": True, "fallback": "none", "solve_time_s": 0.1,
         "min_constraint": 0.0},
        {"country": "JPN", "algorithm": "eact_mpc", "accepted": False,
         "solver_success": False, "fallback": "rule_based", "solve_time_s": 0.2,
         "min_constraint": -1.0},
    ])

    result = summarize(episodes, solver).iloc[0]

    assert result.accepted_plan_rate == 0.5
    assert result.solver_convergence_rate == 0.5
    assert result.solver_fallback_rate == 0.5
    assert result.solver_rule_fallbacks == 1
    assert result.solver_safe_recovery_fallbacks == 0


def test_ashrae_step_metrics_separate_recommended_and_allowable_limits():
    from coastaldc_env.swhp_cooling import CoolingConfig

    cfg = CoolingConfig()
    recommended = thermal_step_metrics(27.5, cfg, 1e-4)
    allowable = thermal_step_metrics(32.25, cfg, 1e-4)

    assert recommended["recommended_compliant"] == 0
    assert recommended["recommended_exceedance_event"] == 1
    assert recommended["recommended_exceedance_c"] == pytest.approx(0.5)
    assert recommended["allowable_exceedance_event"] == 0
    assert recommended["allowable_exceedance_c"] == 0.0
    assert allowable["allowable_exceedance_event"] == 1
    assert allowable["allowable_exceedance_c"] == pytest.approx(0.25)


def test_ashrae_step_metrics_exclude_solver_residue_from_event_count():
    from coastaldc_env.swhp_cooling import CoolingConfig

    metrics = thermal_step_metrics(27.00005, CoolingConfig(), 1e-4)

    assert metrics["recommended_exceedance_event"] == 0
    assert metrics["recommended_exceedance_c"] == pytest.approx(5e-5)


def test_aggregate_thermal_metrics_degree_hours_and_temporal_rci():
    from coastaldc_env.swhp_cooling import CoolingConfig

    cfg = CoolingConfig()
    steps = pd.DataFrame([
        thermal_step_metrics(value, cfg, 1e-4)
        for value in [24.0, 27.5, 28.0, 32.5]
    ])

    metrics = aggregate_thermal_metrics(steps)

    assert metrics["recommended_compliance_pct"] == 25.0
    assert metrics["recommended_exceedance_hours"] == 3.0
    assert metrics["recommended_exceedance_degc_h"] == pytest.approx(7.0)
    assert metrics["allowable_exceedance_hours"] == 1.0
    assert metrics["allowable_exceedance_degc_h"] == pytest.approx(0.5)
    assert metrics["max_t_inlet_c"] == 32.5
    assert metrics["min_allowable_headroom_c"] == -0.5
    assert metrics["temporal_rci_hi_pct"] == pytest.approx(65.0)
