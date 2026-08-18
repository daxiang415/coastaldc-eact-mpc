"""Behavior cloning: MLP regression from observation to continuous action, tanh-squashed."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ACTION_LOW = torch.tensor([-1.0, -1.0, 0.0])
ACTION_HIGH = torch.tensor([1.0, 1.0, 1.0])


def squash_to_action_range(raw: torch.Tensor) -> torch.Tensor:
    """tanh -> [-1, 1], then affine map to the env action box."""
    t = torch.tanh(raw)
    low = ACTION_LOW.to(raw.device)
    high = ACTION_HIGH.to(raw.device)
    return low + (t + 1.0) * 0.5 * (high - low)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return squash_to_action_range(self.net(obs))


class BehaviorCloningController:
    name = "behavior_cloning"

    def __init__(self, obs_dim: int, model_path: str | None = None, device: str = "cpu"):
        self.device = device
        self.policy = BCPolicy(obs_dim).to(device)
        if model_path:
            self.policy.load_state_dict(torch.load(model_path, map_location=device))
        self.policy.eval()

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        with torch.no_grad():
            a = self.policy(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        return a.cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    def train(self, observations: np.ndarray, actions: np.ndarray,
              epochs: int = 50, batch_size: int = 256, lr: float = 1e-3) -> list[float]:
        ds = TensorDataset(torch.as_tensor(observations, dtype=torch.float32),
                           torch.as_tensor(actions, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        losses = []
        self.policy.train()
        for _ in range(epochs):
            ep = 0.0
            for ob, ac in loader:
                pred = self.policy(ob.to(self.device))
                loss = nn.functional.mse_loss(pred, ac.to(self.device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep += loss.item() * len(ob)
            losses.append(ep / len(ds))
        self.policy.eval()
        return losses
