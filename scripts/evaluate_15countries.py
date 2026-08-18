"""Evaluate controllers across the 15 coastal-capacity countries.

Always runs continuous controllers: no_control and rule_based.
Optional: --mpc, --discrete-ablation, and any trained models found in results/models
(sac_/td3_/ppo_ zip, dt_ pt).

Usage:
    python scripts/evaluate_15countries.py --episodes 5 --mpc
    python scripts/evaluate_15countries.py --countries JPN NOR --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys

# Import torch before project modules on Windows GPU environments so CUDA DLLs
# are initialized before SB3 imports torch through nested controller modules.
import torch  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES, CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.rollout import run_episode  # noqa: E402
from controllers.discrete_wrapper import DiscretizedController  # noqa: E402
from controllers.no_control import NoControlController  # noqa: E402
from controllers.rule_based import RuleBasedController  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEST_DATA_DIR = os.path.join(ROOT, "data", "processed_multiyear", "2025")
MODEL_DIR = os.path.join(ROOT, "results", "models_multiyear_v1")
EPISODE_OUT = os.path.join(
    ROOT, "results", "metrics_country_algorithm_episodes_multiyear_v1.csv")
PAIRED_OUT = os.path.join(
    ROOT, "results", "paired_comparisons_multiyear_v1.csv")

PAIR_METRICS = {
    "episode_return": "higher",
    "e_grid_mwh": "lower",
    "co2_kg": "lower",
    "e_total_mwh": "lower",
    "e_cooling_mwh": "lower",
    "e_pump_mwh": "lower",
    "e_wind_unused_mwh": "lower",
    "wind_utilization_pct": "higher",
    "sla_violation_mwh": "lower",
    "terminal_unserved_mwh": "lower",
    "thermal_violation_hours": "lower",
    "safety_interventions": "lower",
    "safety_infeasible_hours": "lower",
    "workload_interventions": "lower",
    "workload_infeasible_hours": "lower",
    "mean_action_override": "lower",
}


def model_matches_env(controller, env) -> bool:
    model_space = getattr(getattr(controller, "model", None), "observation_space", None)
    return (model_space is not None
            and tuple(model_space.shape) == tuple(env.observation_space.shape))


def build_controllers(env, args, country: str) -> list:
    ctrls = [NoControlController(), RuleBasedController()]
    model_dir = getattr(args, "model_dir", MODEL_DIR)
    if args.discrete_ablation:
        ctrls.append(DiscretizedController(RuleBasedController()))

    if args.mpc or args.im_mpc:
        from controllers.mpc import InformationMatchedMPCController, MPCController
        mpc_kwargs = dict(
            horizon=args.mpc_horizon,
            replan_every=args.mpc_replan_every,
            maxiter=args.mpc_maxiter,
            control_block_hours=args.mpc_block_hours,
            gamma=args.mpc_gamma,
        )
        if args.mpc:
            ctrls.append(MPCController(env, **mpc_kwargs))
            if args.discrete_ablation:
                ctrls.append(DiscretizedController(
                    MPCController(env, **mpc_kwargs)))
    if args.im_mpc:
        ctrls.append(InformationMatchedMPCController(env, **mpc_kwargs))
        if args.discrete_ablation:
            ctrls.append(DiscretizedController(
                InformationMatchedMPCController(env, **mpc_kwargs)))

    for algo, cls_path in [("sac", "controllers.sac"), ("td3", "controllers.td3"),
                           ("ppo", "controllers.ppo")]:
        seeds = getattr(args, "seeds", [0]) if algo == "sac" else [0]
        for seed in seeds:
            path = os.path.join(model_dir, f"{algo}_{country}_seed{seed}.zip")
            if os.path.exists(path):
                module = __import__(cls_path, fromlist=["x"])
                controller = getattr(module, f"{algo.upper()}Controller")(model_path=path)
                if model_matches_env(controller, env):
                    controller.name = f"{algo}_seed{seed}"
                    ctrls.append(controller)
                else:
                    print(f"skip incompatible model: {path}")

    dt_path = os.path.join(model_dir, f"dt_{country}_seed0.pt")
    if os.path.exists(dt_path):
        from controllers.decision_transformer_continuous import DecisionTransformerController
        obs_dim = env.observation_space.shape[0]
        ctrls.append(DecisionTransformerController(
            obs_dim, target_return=args.dt_target_return, model_path=dt_path))
    return ctrls


def episode_result_row(country: str, algorithm: str, episode: int,
                       reset_seed: int, traj: dict) -> dict:
    requested = np.asarray(traj["actions"], dtype=float)
    applied = np.asarray(traj.get("applied_actions", requested), dtype=float)
    override = np.abs(requested - applied)
    row = {
        "country": country,
        "algorithm": algorithm,
        "episode": episode,
        "reset_seed": reset_seed,
        "start_hour": int(traj.get("reset_info", {}).get("data_hour", -1)),
        "episode_return": float(traj["episode_return"]),
        **{k: float(v) for k, v in traj["metrics"].items()},
        "requested_workload_mean": float(requested[:, 0].mean()),
        "requested_workload_defer_fraction": float(np.mean(requested[:, 0] < -1e-6)),
        "requested_workload_recover_fraction": float(np.mean(requested[:, 0] > 1e-6)),
        "requested_setpoint_mean": float(requested[:, 1].mean()),
        "requested_pump_mean": float(requested[:, 2].mean()),
        "applied_setpoint_mean": float(applied[:, 1].mean()),
        "applied_pump_mean": float(applied[:, 2].mean()),
        "safety_override_fraction": float(np.mean(np.any(override > 1e-6, axis=1))),
        "mean_abs_action_override": float(override.mean()),
    }
    row.update({f"reward_{k}": float(v)
                for k, v in traj.get("reward_terms", {}).items()})
    return row


def _bootstrap_mean_ci(values: np.ndarray, samples: int,
                       seed: int = 20260713) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or np.allclose(values, values[0]):
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def build_paired_comparisons(episode_df: pd.DataFrame,
                             bootstrap_samples: int = 10_000) -> pd.DataFrame:
    rows = []
    keys = ["country", "episode", "reset_seed"]
    for country, country_df in episode_df.groupby("country"):
        sac_names = sorted(
            name for name in country_df["algorithm"].unique()
            if name.startswith("sac"))
        baseline_names = [
            name for name in ("no_control", "rule_based", "mpc", "oracle_mpc", "im_mpc")
            if name in set(country_df["algorithm"])
        ]
        for sac_name in sac_names:
            sac_df = country_df[country_df["algorithm"] == sac_name]
            for baseline_name in baseline_names:
                baseline_df = country_df[country_df["algorithm"] == baseline_name]
                paired = sac_df.merge(
                    baseline_df, on=keys, suffixes=("_sac", "_baseline"),
                    validate="one_to_one")
                if not np.array_equal(
                        paired["start_hour_sac"].to_numpy(),
                        paired["start_hour_baseline"].to_numpy()):
                    raise ValueError(
                        f"Unpaired episode starts for {country}: "
                        f"{sac_name} vs {baseline_name}")

                for metric, direction in PAIR_METRICS.items():
                    if (f"{metric}_sac" not in paired
                            or f"{metric}_baseline" not in paired):
                        continue
                    sac_values = paired[f"{metric}_sac"].to_numpy(dtype=float)
                    baseline_values = paired[f"{metric}_baseline"].to_numpy(dtype=float)
                    raw_difference = sac_values - baseline_values
                    improvement = (raw_difference if direction == "higher"
                                   else -raw_difference)
                    ci_low, ci_high = _bootstrap_mean_ci(
                        improvement, bootstrap_samples)
                    if np.allclose(improvement, 0.0):
                        p_value = 1.0
                    else:
                        p_value = float(wilcoxon(
                            improvement, alternative="two-sided").pvalue)
                    baseline_mean = float(baseline_values.mean())
                    relative = (100.0 * float(improvement.mean())
                                / abs(baseline_mean)
                                if abs(baseline_mean) > 1e-12 else np.nan)
                    rows.append({
                        "country": country,
                        "sac_algorithm": sac_name,
                        "baseline": baseline_name,
                        "metric": metric,
                        "direction": direction,
                        "n_episodes": len(paired),
                        "sac_mean": float(sac_values.mean()),
                        "baseline_mean": baseline_mean,
                        "mean_difference_sac_minus_baseline": float(raw_difference.mean()),
                        "mean_improvement": float(improvement.mean()),
                        "relative_improvement_pct": relative,
                        "improvement_ci95_low": ci_low,
                        "improvement_ci95_high": ci_high,
                        "wilcoxon_p_value": p_value,
                    })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=COUNTRIES)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--data-dir", default=TEST_DATA_DIR)
    parser.add_argument("--mpc", action="store_true", help="include MPC (slower)")
    parser.add_argument("--im-mpc", action="store_true",
                        help="include information-matched MPC")
    parser.add_argument("--discrete-ablation", action="store_true",
                        help="also evaluate quantized discrete-action variants")
    parser.add_argument("--mpc-horizon", type=int, default=24)
    parser.add_argument("--mpc-replan-every", type=int, default=4)
    parser.add_argument("--mpc-maxiter", type=int, default=10)
    parser.add_argument("--mpc-block-hours", type=int, default=4)
    parser.add_argument("--mpc-gamma", type=float, default=0.995)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--dt-target-return", type=float, default=-50.0)
    parser.add_argument("--out", default=os.path.join(
        ROOT, "results", "metrics_country_algorithm_multiyear_v1.csv"))
    parser.add_argument("--episode-out", default=EPISODE_OUT)
    parser.add_argument("--paired-out", default=PAIRED_OUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()

    rows, episode_rows = [], []
    for country in args.countries:
        env = CoastalDCContinuousEnv(
            country=country, data_dir=args.data_dir, seed=0)
        for ctrl in build_controllers(env, args, country):
            returns, metrics_list = [], []
            for ep in range(args.episodes):
                reset_seed = 5000 + ep
                traj = run_episode(env, ctrl, seed=reset_seed)
                returns.append(traj["episode_return"])
                metrics_list.append(traj["metrics"])
                episode_rows.append(episode_result_row(
                    country, ctrl.name, ep, reset_seed, traj))
            agg = {k: float(np.mean([m[k] for m in metrics_list]))
                   for k in metrics_list[0]}
            row = {"country": country, "algorithm": ctrl.name,
                   "return_mean": float(np.mean(returns)),
                   "return_std": float(np.std(returns)), **agg}
            rows.append(row)
            print(f"{country} | {ctrl.name:28s} | return {row['return_mean']:8.2f} "
                  f"| CO2 {row['co2_kg']/1000:7.1f} t | grid {row['e_grid_mwh']:7.1f} MWh "
                  f"| wind_util {row['wind_utilization_pct']:5.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nsaved -> {args.out}")

    episode_df = pd.DataFrame(episode_rows)
    os.makedirs(os.path.dirname(args.episode_out), exist_ok=True)
    episode_df.to_csv(args.episode_out, index=False)
    print(f"saved -> {args.episode_out}")

    paired_df = build_paired_comparisons(
        episode_df, bootstrap_samples=args.bootstrap_samples)
    os.makedirs(os.path.dirname(args.paired_out), exist_ok=True)
    paired_df.to_csv(args.paired_out, index=False)
    print(f"saved -> {args.paired_out}")

    if args.discrete_ablation:
        abl = df[df["algorithm"].str.startswith(("rule_based", "discrete_rule_based",
                                                 "mpc", "discrete_mpc"))]
        abl_path = os.path.join(ROOT, "results", "continuous_vs_discrete_ablation.csv")
        abl.to_csv(abl_path, index=False)
        print(f"saved -> {abl_path}")


if __name__ == "__main__":
    main()
