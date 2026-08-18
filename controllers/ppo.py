"""PPO baseline (stable-baselines3 wrapper)."""

from __future__ import annotations

import numpy as np


class PPOController:
    name = "ppo"

    def __init__(self, model_path: str | None = None, model=None):
        if model is not None:
            self.model = model
        elif model_path is not None:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
        else:
            raise ValueError("Provide model or model_path")

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def make_ppo(env, seed: int = 0, **kwargs):
    from stable_baselines3 import PPO
    defaults = dict(learning_rate=3e-4, n_steps=2048, batch_size=256, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=[256, 256]), verbose=1, seed=seed)
    defaults.update(kwargs)
    return PPO("MlpPolicy", env, **defaults)
