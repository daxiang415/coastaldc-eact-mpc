"""SAC baseline (stable-baselines3 wrapper)."""

from __future__ import annotations

import numpy as np


class SACController:
    name = "sac"

    def __init__(self, model_path: str | None = None, model=None,
                 device: str = "cpu"):
        if model is not None:
            self.model = model
        elif model_path is not None:
            from stable_baselines3 import SAC
            self.model = SAC.load(model_path, device=device)
        else:
            raise ValueError("Provide model or model_path")

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def make_sac(env, seed: int = 0, **kwargs):
    from stable_baselines3 import SAC
    defaults = dict(learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                    gamma=0.995, tau=0.005, train_freq=1, gradient_steps=1,
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=seed)
    defaults.update(kwargs)
    return SAC("MlpPolicy", env, **defaults)
