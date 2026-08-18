"""TD3 baseline (stable-baselines3 wrapper)."""

from __future__ import annotations

import numpy as np


class TD3Controller:
    name = "td3"

    def __init__(self, model_path: str | None = None, model=None):
        if model is not None:
            self.model = model
        elif model_path is not None:
            from stable_baselines3 import TD3
            self.model = TD3.load(model_path)
        else:
            raise ValueError("Provide model or model_path")

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def make_td3(env, seed: int = 0, **kwargs):
    import numpy as _np
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    n_actions = env.action_space.shape[0]
    defaults = dict(learning_rate=1e-3, buffer_size=200_000, batch_size=256,
                    gamma=0.99, tau=0.005,
                    action_noise=NormalActionNoise(mean=_np.zeros(n_actions),
                                                   sigma=0.1 * _np.ones(n_actions)),
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=seed)
    defaults.update(kwargs)
    return TD3("MlpPolicy", env, **defaults)
