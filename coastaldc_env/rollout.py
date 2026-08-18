"""Shared rollout utilities: run a controller for one episode, collect trajectory + metrics."""

from __future__ import annotations

import numpy as np


def run_episode(env, controller, seed: int | None = None,
                start_hour: int | None = None) -> dict:
    options = {"start_hour": start_hour} if start_hour is not None else None
    obs, info = env.reset(seed=seed, options=options)
    reset_info = dict(info)
    if hasattr(controller, "reset"):
        controller.reset()

    observations, actions, applied_actions, rewards, infos = [], [], [], [], []
    done = False
    while not done:
        action = controller.act(obs, info)
        observations.append(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        if hasattr(controller, "update_reward"):     # DT return-to-go bookkeeping
            controller.update_reward(reward)
        actions.append(np.asarray(action, dtype=np.float32))
        applied_actions.append(np.asarray(
            info.get("applied_action", action), dtype=np.float32))
        rewards.append(reward)
        infos.append(info)
        obs = next_obs
        done = terminated or truncated

    reward_terms: dict[str, float] = {}
    for step_info in infos:
        for name, value in step_info.get("reward_terms", {}).items():
            reward_terms[name] = reward_terms.get(name, 0.0) + float(value)

    return {
        "observations": np.stack(observations).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "applied_actions": np.stack(applied_actions).astype(np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "episode_return": float(np.sum(rewards)),
        "metrics": env.episode_summary(),
        "reward_terms": reward_terms,
        "reset_info": reset_info,
    }


def save_trajectories(path: str, trajectories: list[dict], source: str):
    """Save a list of trajectories to a compressed npz."""
    np.savez_compressed(
        path,
        observations=np.stack([t["observations"] for t in trajectories]),
        actions=np.stack([t["actions"] for t in trajectories]),
        rewards=np.stack([t["rewards"] for t in trajectories]),
        returns=np.array([t["episode_return"] for t in trajectories]),
        source=source,
    )


def load_trajectories(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        d = np.load(p, allow_pickle=True)
        for i in range(len(d["returns"])):
            out.append({
                "observations": d["observations"][i],
                "actions": d["actions"][i],
                "rewards": d["rewards"][i],
            })
    return out
