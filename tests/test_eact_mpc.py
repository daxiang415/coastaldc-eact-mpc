import numpy as np
import pandas as pd

from coastaldc_env import CoastalDCContinuousEnv
from coastaldc_env.forecasting import CausalRidgeForecaster, RidgeForecastConfig
from controllers.eact_mpc import (
    EACTMPCController,
    EACTNoBiasMPCController,
    EACTNoInterventionMPCController,
    NominalCausalMPCController,
    OracleConstrainedMPCController,
    StaticRobustMPCController,
)


def _data(hours: int = 700) -> pd.DataFrame:
    t = np.arange(hours, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=hours, freq="h"),
        "fixed_load_mw": 6.0 + 0.4 * np.sin(2 * np.pi * t / 24),
        "flexible_arrival_mw": 1.5 + 0.2 * np.cos(2 * np.pi * t / 24),
        "sst_c": 17.0 + np.sin(2 * np.pi * t / (24 * 30)),
        "wind_mw": np.clip(6.0 + 3.0 * np.sin(2 * np.pi * t / 12), 0.0, 12.0),
        "carbon_kg_per_mwh": 400.0 + 40.0 * np.cos(2 * np.pi * t / 24),
        "price_usd_per_mwh": np.full(hours, 100.0),
    })


def _setup(controller_cls, horizon=4, maxiter=3, **controller_kwargs):
    data = _data()
    config = RidgeForecastConfig(horizon=30, alpha=0.1)
    forecaster = CausalRidgeForecaster(config).fit(data.iloc[:500])
    test = data.iloc[500:620].reset_index(drop=True)
    env = CoastalDCContinuousEnv(
        country="JPN", data=test, episode_hours=48,
        random_episode_start=False, seed=0)
    history_tail = data.iloc[332:500][
        ["timestamp", *forecaster.columns]].reset_index(drop=True)
    residuals = np.zeros((20, len(forecaster.columns), config.horizon))
    residuals[:, :, :] = np.linspace(-0.2, 0.2, 20)[:, None, None]
    controller = controller_cls(
        env,
        forecaster=forecaster,
        calibration_residuals=residuals,
        calibration_beta={column: 0.1 for column in forecaster.columns},
        history_tail=history_tail,
        horizon=horizon,
        control_block_hours=2,
        maxiter=maxiter,
        **controller_kwargs,
    )
    env.reset(seed=0, options={"start_hour": 0})
    return env, controller


def test_causal_mpc_prediction_does_not_access_future_window(monkeypatch):
    env, controller = _setup(NominalCausalMPCController)

    def reject_future_access(*args, **kwargs):
        raise AssertionError("Causal MPC must not access exogenous_window")

    monkeypatch.setattr(env, "exogenous_window", reject_future_access)
    inputs = controller._prediction_inputs()

    assert inputs["H"] == 4
    assert inputs["fixed"][0] == env.data.iloc[0].fixed_load_mw
    assert len(inputs["all_fixed"]) == 4 + env.wl_cfg.max_delay_hours


def test_causal_mpc_objective_is_independent_of_electricity_price():
    env, controller = _setup(NominalCausalMPCController)
    inputs = controller._prediction_inputs()
    safe_blocks = np.tile([1.0, -1.0, 1.0], 2)
    original = controller._simulate(safe_blocks, inputs)

    env.data.loc[:, "price_usd_per_mwh"] = 1000.0
    changed_inputs = controller._prediction_inputs()
    changed = controller._simulate(safe_blocks, changed_inputs)

    assert "price" not in inputs
    assert "price" not in changed_inputs
    assert changed.cost == original.cost
    np.testing.assert_allclose(changed.constraints, original.constraints)
    np.testing.assert_allclose(changed.actions, original.actions)


def test_static_and_adaptive_modes_apply_calibrated_bounds():
    _, nominal = _setup(NominalCausalMPCController)
    _, static = _setup(StaticRobustMPCController)
    _, adaptive = _setup(EACTMPCController)

    nominal_inputs = nominal._prediction_inputs()
    static_inputs = static._prediction_inputs()
    adaptive_inputs = adaptive._prediction_inputs()

    np.testing.assert_allclose(
        static_inputs["fixed"][1:], adaptive_inputs["fixed"][1:])
    assert np.all(static_inputs["wind"][1:] <= nominal_inputs["wind"][1:] + 1e-12)
    assert np.all(static_inputs["sst"][1:] >= nominal_inputs["sst"][1:] - 1e-12)


def test_adaptive_controller_registers_stressed_forecasts_causally():
    _, controller = _setup(
        EACTMPCController,
        forecast_stress_mode="adverse_bias",
        forecast_stress_scale=1.0,
        forecast_stress_start_step=0,
        forecast_stress_seed=11,
    )

    controller._prediction_inputs()

    assert controller._last_forecast_stress["forecast_stress_active"] is True
    assert controller._last_forecast_stress["forecast_stress_mean_abs"] > 0.0
    assert controller._adaptor.pending_targets


def test_adaptive_joint_load_bound_respects_it_capacity():
    env, controller = _setup(EACTMPCController)
    fixed_index = controller.forecaster.columns.index("fixed_load_mw")
    flex_index = controller.forecaster.columns.index("flexible_arrival_mw")
    controller._adaptor._bias[[fixed_index, flex_index], :] = 5.0

    inputs = controller._prediction_inputs()

    joint = inputs["fixed"] + inputs["flex"]
    assert np.all(joint <= env.wl_cfg.it_capacity_mw + 1e-12)


def test_eact_ablation_parameters_are_isolated():
    _, no_bias = _setup(EACTNoBiasMPCController)
    _, no_intervention = _setup(EACTNoInterventionMPCController)

    np.testing.assert_allclose(no_bias._adaptor.bias, 0.0)
    assert no_bias.adaptive_bias_correction is False
    assert no_intervention.adaptive_bias_correction is True
    assert no_intervention.intervention_weight == 0.0


def test_adaptive_beta_floor_only_changes_adaptive_bias_update():
    _, adaptive = _setup(EACTMPCController, adaptive_beta_floor=0.2)
    _, static = _setup(StaticRobustMPCController, adaptive_beta_floor=0.2)

    assert set(adaptive._adaptor.beta.values()) == {0.2}
    assert set(static._adaptor.beta.values()) == {0.1}


def test_rollout_constraints_evaluate_unprojected_actions():
    env, controller = _setup(NominalCausalMPCController, horizon=6)
    inputs = controller._prediction_inputs()
    initial_room = env.cooling.t_room
    initial_setpoint = env.cooling.t_set
    initial_queue = env.workload.queue.copy()
    # Warm setpoint and minimum pumping should expose thermal infeasibility.
    unsafe_blocks = np.tile([0.0, 1.0, 0.0], 3)
    rollout = controller._simulate(unsafe_blocks, inputs)

    assert rollout.constraints.ndim == 1
    assert np.min(rollout.constraints) < 0.0
    assert env.cooling.cfg.enforce_thermal_safety is True
    assert env.cooling.t_room == initial_room
    assert env.cooling.t_set == initial_setpoint
    np.testing.assert_array_equal(env.workload.queue, initial_queue)


def test_conservative_plan_satisfies_recommended_inlet_constraint():
    env, controller = _setup(NominalCausalMPCController, horizon=4)
    inputs = controller._prediction_inputs()
    safe_blocks = np.tile([1.0, -1.0, 1.0], 2)
    rollout = controller._simulate(safe_blocks, inputs)

    assert env.cool_cfg.t_inlet_recommended_max_c == 27.0
    assert np.min(rollout.constraints) >= -controller.constraint_tolerance


def test_mpc_rollout_does_not_invoke_shared_thermal_projector(monkeypatch):
    env, controller = _setup(NominalCausalMPCController, horizon=4)
    inputs = controller._prediction_inputs()
    safe_blocks = np.tile([1.0, -1.0, 1.0], 2)

    def reject_projection(*args, **kwargs):
        raise AssertionError("MPC candidate rollout must not use safety projection")

    monkeypatch.setattr(
        type(env.cooling), "_project_safe_controls", reject_projection)

    rollout = controller._simulate(safe_blocks, inputs)

    assert np.min(rollout.constraints) >= -controller.constraint_tolerance


def test_eact_action_is_bounded_and_solver_diagnostics_are_recorded():
    env, controller = _setup(EACTMPCController, horizon=3, maxiter=2)
    obs, info = env.reset(seed=0, options={"start_hour": 0})
    action = controller.act(obs, info)

    assert env.action_space.contains(action)
    assert len(controller.solver_history) == 1
    assert controller.solver_history[0]["fallback"] in {
        "none", "safe_recovery", "shifted_plan", "rule_based"}


def test_terminal_solver_failure_uses_safe_recovery(monkeypatch):
    env, controller = _setup(EACTMPCController, horizon=3, maxiter=2)
    obs, info = env.reset(seed=0, options={"start_hour": 0})
    env._step_idx = env.episode_hours - env.wl_cfg.max_delay_hours

    def reject_plan(inputs):
        plan = np.zeros((int(inputs["H"]), 3), dtype=float)
        diagnostics = {
            "accepted": False, "solver_success": False, "status": 9,
            "message": "test failure", "iterations": 0, "objective": 0.0,
            "min_constraint": -1.0, "plan_source": "solver",
        }
        return plan, False, diagnostics

    monkeypatch.setattr(controller, "_optimize", reject_plan)
    action = controller.act(obs, info)

    np.testing.assert_allclose(action, [1.0, -1.0, 1.0])
    assert controller.solver_history[0]["fallback"] == "safe_recovery"


def test_terminal_window_expands_to_episode_end():
    env, controller = _setup(NominalCausalMPCController, horizon=4)
    env._step_idx = env.episode_hours - env.wl_cfg.max_delay_hours

    inputs = controller._prediction_inputs()

    assert inputs["H"] == env.wl_cfg.max_delay_hours
    assert inputs["episode_terminal"] is True


def test_oracle_constrained_mpc_uses_realized_future_with_shared_rollout():
    env, controller = _setup(OracleConstrainedMPCController, horizon=4)

    inputs = controller._prediction_inputs()

    np.testing.assert_allclose(
        inputs["fixed"], env.data.iloc[:4]["fixed_load_mw"])
    np.testing.assert_allclose(
        inputs["wind"], env.data.iloc[:4]["wind_mw"])
    safe_blocks = np.tile([1.0, -1.0, 1.0], 2)
    rollout = controller._simulate(safe_blocks, inputs)
    assert np.min(rollout.constraints) >= -controller.constraint_tolerance
