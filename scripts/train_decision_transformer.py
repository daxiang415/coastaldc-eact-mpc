"""Train the continuous Decision Transformer on the mixed offline dataset.

Recommended mixture (README): 40% MPC + 30% rule-based + 20% safe-random (+10% RL expert).
Run the generate_*_trajectories.py scripts first.

Usage:
    python scripts/train_decision_transformer.py --country JPN --epochs 20
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import load_trajectories, run_episode  # noqa: E402
from controllers.decision_transformer_continuous import (  # noqa: E402
    ContinuousDecisionTransformer, DecisionTransformerController, train_dt)

MIXTURE = {"mpc": 0.4, "rule_based": 0.3, "safe_random": 0.2, "rl": 0.1}


def build_mixed_dataset(traj_dir: str, country: str, rng: np.random.Generator) -> list[dict]:
    pools = {}
    for source in MIXTURE:
        paths = glob.glob(os.path.join(traj_dir, f"{source}_{country}.npz"))
        if paths:
            pools[source] = load_trajectories(paths)
    if not pools:
        raise FileNotFoundError(
            f"No trajectory files found in {traj_dir} for {country}. "
            "Run generate_*_trajectories.py first.")

    total = sum(len(v) for v in pools.values())
    mixed = []
    present = {s: w for s, w in MIXTURE.items() if s in pools}
    z = sum(present.values())
    for source, trajs in pools.items():
        n = max(1, int(round(total * present[source] / z)))
        idx = rng.choice(len(trajs), size=min(n, len(trajs)), replace=False)
        mixed += [trajs[i] for i in idx]
    rng.shuffle(mixed)
    print({s: len(v) for s, v in pools.items()}, "-> mixed:", len(mixed))
    return mixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--context-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traj-dir", default=None)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    traj_dir = args.traj_dir or os.path.join(root, "data", "trajectories")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    trajs = build_mixed_dataset(traj_dir, args.country, rng)
    obs_dim = trajs[0]["observations"].shape[1]

    model = ContinuousDecisionTransformer(obs_dim, context_len=args.context_len).to(device)
    losses = train_dt(model, trajs, epochs=args.epochs, device=device, seed=args.seed)
    print("final MSE loss:", losses[-1])

    out_dir = os.path.join(root, "results", "models")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"dt_{args.country}_seed{args.seed}.pt")
    torch.save(model.state_dict(), path)
    print(f"saved -> {path}")

    # quick evaluation, target return = best return in dataset
    returns = [float(t["rewards"].sum()) for t in trajs]
    target = max(returns)
    env = CoastalDCContinuousEnv(country=args.country, seed=args.seed)
    ctrl = DecisionTransformerController(obs_dim, target_return=target,
                                         context_len=args.context_len, device=device)
    ctrl.model.load_state_dict(model.state_dict())
    for ep in range(args.eval_episodes):
        traj = run_episode(env, ctrl, seed=9000 + ep)
        print(f"eval episode {ep}: return={traj['episode_return']:.2f} (target {target:.2f})")


if __name__ == "__main__":
    main()
