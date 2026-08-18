"""Rule-based controller policy-direction checks."""

import numpy as np

from coastaldc_env.continuous_env import ObsIndex as O
from controllers.rule_based import RuleBasedController


def _neutral_observation() -> np.ndarray:
    obs = np.zeros(29, dtype=np.float32)
    obs[O.SST] = 20.0 / 30.0
    obs[O.T_ROOM] = 0.45
    obs[O.REMAINING] = 1.0
    return obs


def test_low_wind_high_carbon_defers_flexible_workload():
    obs = _neutral_observation()
    obs[O.WIND] = 0.10
    obs[O.CARBON] = 0.80
    obs[O.DEADLINE_PRESSURE] = 0.0

    action = RuleBasedController().act(obs)

    assert action[0] < -0.1


def test_terminal_window_forces_backlog_recovery():
    obs = _neutral_observation()
    obs[O.WIND] = 0.10
    obs[O.CARBON] = 0.80
    obs[O.BACKLOG] = 0.5
    obs[O.REMAINING] = 12.0 / 168.0

    action = RuleBasedController().act(obs)

    assert action[0] == 1.0
