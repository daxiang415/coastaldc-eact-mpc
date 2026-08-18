"""Generate MPC expert trajectories for offline training.

Usage: python scripts/generate_mpc_trajectories.py --country JPN --episodes 40 --horizon 24
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import run_episode, save_trajectories  # noqa: E402
from controllers.mpc import MPCController  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--replan-every", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    env = CoastalDCContinuousEnv(country=args.country, seed=0)
    ctrl = MPCController(env, horizon=args.horizon, replan_every=args.replan_every)

    trajs = []
    for ep in range(args.episodes):
        traj = run_episode(env, ctrl, seed=2000 + ep)
        trajs.append(traj)
        print(f"episode {ep}: return={traj['episode_return']:.2f}")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                   "data", "trajectories", f"mpc_{args.country}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_trajectories(out, trajs, source="mpc")
    print(f"saved {len(trajs)} trajectories -> {out}")


if __name__ == "__main__":
    main()
