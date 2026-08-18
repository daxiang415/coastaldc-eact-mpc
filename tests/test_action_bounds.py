"""Action-space shape, bounds handling and setpoint rate limits."""

import numpy as np
import pandas as pd
import pytest

from coastaldc_env import CoastalDCContinuousEnv


def test_action_space_definition(env):
    assert env.action_space.shape == (3,)
    np.testing.assert_allclose(env.action_space.low, [-1, -1, 0])
    np.testing.assert_allclose(env.action_space.high, [1, 1, 1])


def test_out_of_range_actions_are_clipped(env):
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(np.array([5.0, -5.0, 9.0]))
    assert np.isfinite(r)
    assert info["t_set"] >= env.cool_cfg.t_set_min


def test_wrong_shape_raises(env):
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(np.zeros(4))


def test_setpoint_rate_limit(env):
    env.reset(seed=0)
    t0 = env.cooling.t_set
    env.step(np.array([0.0, 1.0, 0.5], dtype=np.float32))
    assert abs(env.cooling.t_set - t0) <= env.cool_cfg.dt_set_max + 1e-9


def test_setpoint_stays_in_bounds(env):
    env.reset(seed=0)
    c = env.cool_cfg
    for _ in range(50):
        env.step(np.array([0.0, -1.0, 0.5], dtype=np.float32))
    assert env.cooling.t_set >= c.t_set_min - 1e-9
    env.reset(seed=0)
    for _ in range(50):
        env.step(np.array([0.0, 1.0, 0.5], dtype=np.float32))
    assert env.cooling.t_set <= c.t_set_max + 1e-9


def test_observation_shape_and_finite(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(20):
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))


def test_observation_includes_normalized_time_remaining(env):
    from coastaldc_env.continuous_env import ObsIndex as O

    obs, _ = env.reset(seed=0)
    assert obs.shape == (29,)
    assert obs[O.REMAINING] == pytest.approx(1.0)

    obs, *_ = env.step(np.array([0.0, 0.0, 0.5], dtype=np.float32))
    assert obs[O.REMAINING] == pytest.approx(
        (env.episode_hours - 1) / env.episode_hours)


def test_observation_can_mask_perfect_future_forecasts():
    from coastaldc_env.continuous_env import ObsIndex as O

    masked_env = CoastalDCContinuousEnv(
        country="JPN", use_oracle_forecast_observations=False, seed=0)
    obs, info = masked_env.reset(seed=0)

    np.testing.assert_array_equal(obs[O.WIND_FC], 0.0)
    np.testing.assert_array_equal(obs[O.CARBON_FC], 0.0)
    assert not info["oracle_forecast_observations_enabled"]


def test_timestamp_drives_calendar_features():
    from coastaldc_env.continuous_env import ObsIndex as O

    timestamps = pd.date_range("2025-03-03 06:00", periods=30, freq="h")
    data = pd.DataFrame({
        "timestamp": timestamps,
        "fixed_load_mw": 5.0,
        "flexible_arrival_mw": 1.0,
        "sst_c": 15.0,
        "wind_mw": 5.0,
        "carbon_kg_per_mwh": 300.0,
        "price_usd_per_mwh": 100.0,
    })
    timestamp_env = CoastalDCContinuousEnv(
        country="JPN", data=data, episode_hours=2, random_episode_start=False)

    obs, _ = timestamp_env.reset()

    # 06:00 UTC -> sin(pi/2)=1; Monday -> weekday phase starts at zero.
    assert obs[O.TIME.start] == pytest.approx(1.0)
    assert obs[O.TIME.start + 2] == pytest.approx(0.0, abs=1e-7)
    assert obs[O.TIME.start + 3] == pytest.approx(1.0)
