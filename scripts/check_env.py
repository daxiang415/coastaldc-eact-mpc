"""Smoke test: Gymnasium API compliance + one random and one no-control episode.

Usage: python scripts/check_env.py [--country JPN]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import run_episode  # noqa: E402
from controllers.no_control import NoControlController  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    args = parser.parse_args()

    env = CoastalDCContinuousEnv(country=args.country, seed=0)

    try:
        from gymnasium.utils.env_checker import check_env
        check_env(env, skip_render_check=True)
        print("gymnasium check_env: OK")
    except Exception as e:  # noqa: BLE001
        print(f"gymnasium check_env warning: {e}")

    # random episode
    obs, _ = env.reset(seed=42)
    total = 0.0
    for _ in range(env.episode_hours):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        total += r
        if term or trunc:
            break
    print(f"random policy return: {total:.2f}")

    # no-control episode
    traj = run_episode(env, NoControlController(), seed=42, start_hour=0)
    print(f"no-control return:    {traj['episode_return']:.2f}")
    for k, v in traj["metrics"].items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
