"""Generate rule-based controller trajectories for offline training.

Usage: python scripts/generate_rule_based_trajectories.py --country JPN --episodes 30
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402
from controllers.rule_based import RuleBasedController  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = CoastalDCContinuousEnv(country=args.country, seed=0)
    ctrl = RuleBasedController()

    trajs = []
    for ep in range(args.episodes):
        traj = run_episode(env, ctrl, seed=1000 + ep)
        trajs.append(traj)
        print(f"episode {ep}: return={traj['episode_return']:.2f}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "data", "trajectories",
                                   f"rule_based_{args.country}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_trajectories(out, trajs, source="rule_based")
    print(f"saved {len(trajs)} trajectories -> {out}")


if __name__ == "__main__":
    main()
