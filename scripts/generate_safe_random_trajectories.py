"""Generate safe-random trajectories: random continuous actions with emergency override
when thermal or SLA limits are approached.

Usage: python scripts/generate_safe_random_trajectories.py --country JPN --episodes 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.continuous_env import ObsIndex as O  # noqa: E402
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402


class SafeRandomController:
    name = "safe_random"

    def __init__(self, seed: int = 0, t_room_alarm: float = 0.85, pressure_alarm: float = 0.7):
        self.rng = np.random.default_rng(seed)
        self.t_room_alarm = t_room_alarm
        self.pressure_alarm = pressure_alarm

    def reset(self):
        pass

    def act(self, obs: np.ndarray, info: dict | None = None) -> np.ndarray:
        a = np.array([self.rng.uniform(-1, 1),
                      self.rng.uniform(-1, 1),
                      self.rng.uniform(0, 1)], dtype=np.float32)
        # emergency overrides
        if obs[O.T_ROOM] > self.t_room_alarm:        # thermal: cool hard
            a[1] = -1.0
            a[2] = 1.0
        if obs[O.DEADLINE_PRESSURE] > self.pressure_alarm:  # SLA: recover backlog
            a[0] = 1.0
        return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = CoastalDCContinuousEnv(country=args.country, seed=0)
    trajs = []
    for ep in range(args.episodes):
        ctrl = SafeRandomController(seed=3000 + ep)
        traj = run_episode(env, ctrl, seed=3000 + ep)
        trajs.append(traj)
        print(f"episode {ep}: return={traj['episode_return']:.2f}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "data", "trajectories",
                                   f"safe_random_{args.country}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_trajectories(out, trajs, source="safe_random")
    print(f"saved {len(trajs)} trajectories -> {out}")


if __name__ == "__main__":
    main()
