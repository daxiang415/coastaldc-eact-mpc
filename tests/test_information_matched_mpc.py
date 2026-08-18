"""Information-matched MPC must not access the environment's true future rows."""

import numpy as np

from coastaldc_env import CoastalDCContinuousEnv
from coastaldc_env.continuous_env import ObsIndex
from controllers.mpc import InformationMatchedMPCController


def test_information_matched_inputs_expand_sac_forecast_bins():
    env = CoastalDCContinuousEnv(country="JPN", seed=0)
    obs, _ = env.reset(seed=0, options={"start_hour": 0})
    ctrl = InformationMatchedMPCController(
        env, horizon=24, replan_every=12, maxiter=1, control_block_hours=6)

    inputs = ctrl._prediction_inputs(obs)
    expected_wind = np.repeat(
        obs[ObsIndex.WIND_FC] * env.wind_cfg.wind_capacity_mw, 6)
    expected_carbon = np.repeat(obs[ObsIndex.CARBON_FC] * 1000.0, 6)

    np.testing.assert_allclose(inputs["wind"], expected_wind)
    np.testing.assert_allclose(inputs["ci"], expected_carbon)
    np.testing.assert_allclose(inputs["fixed"], inputs["fixed"][0])
    np.testing.assert_allclose(inputs["flex"], inputs["flex"][0])
    np.testing.assert_allclose(inputs["sst"], inputs["sst"][0])


def test_information_matched_mpc_does_not_access_true_future_window(monkeypatch):
    env = CoastalDCContinuousEnv(country="JPN", seed=0)
    obs, _ = env.reset(seed=0, options={"start_hour": 0})

    def reject_future_access(*args, **kwargs):
        raise AssertionError("IM-MPC must not access exogenous_window")

    monkeypatch.setattr(env, "exogenous_window", reject_future_access)
    ctrl = InformationMatchedMPCController(
        env, horizon=6, replan_every=6, maxiter=1, control_block_hours=6)

    action = ctrl.act(obs)

    assert env.action_space.contains(action)
