"""Train SAC / TD3 / PPO on one country environment.

Usage: python scripts/train_online_rl.py --algo sac --country JPN --timesteps 200000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

# Import torch before project modules on Windows GPU environments so CUDA DLLs
# are initialized before SB3 imports torch through nested modules.
import torch  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import CoastalDCContinuousEnv  # noqa: E402
from coastaldc_env.reward import RewardWeights  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN_DATA_DIR = os.path.join(
    ROOT, "data", "processed_multiyear", "train_2023_2024")
SAFE_MODEL_DIR = os.path.join(ROOT, "results", "models_multiyear_v1")
SAFE_CHECKPOINT_DIR = os.path.join(ROOT, "results", "checkpoints_multiyear_v1")
SAFE_BEST_MODEL_DIR = os.path.join(ROOT, "results", "best_models_multiyear_v1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["sac", "td3", "ppo"], default="sac")
    parser.add_argument("--country", default="JPN")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=TRAIN_DATA_DIR)
    parser.add_argument("--validation-data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--sla-penalty-weight",
        type=float,
        default=50.0,
        help="training-only Lagrangian penalty for deadline and terminal unserved MWh",
    )
    parser.add_argument(
        "--oracle-workload-projection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow the feasibility layer to inspect true future workload",
    )
    parser.add_argument(
        "--oracle-forecast-observations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include perfect next-24h wind and carbon observations",
    )
    parser.add_argument("--progress-bar", action="store_true",
                        help="enable the SB3 progress bar; requires tqdm and rich")
    parser.add_argument("--checkpoint-freq", type=int, default=0,
                        help="save model checkpoints every N environment steps; 0 disables")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=0,
                        help="validation frequency in training steps; 0 disables")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--best-model-dir", default=None)
    return parser


def main():
    args = build_parser().parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda requested, but this Python installation has no CUDA-enabled "
            f"PyTorch (torch={torch.__version__})")
    if args.sla_penalty_weight <= 0.0:
        raise ValueError("--sla-penalty-weight must be positive for paper-facing RL")

    env = CoastalDCContinuousEnv(
        country=args.country,
        data_dir=args.data_dir,
        reward_weights=RewardWeights(w_sla=args.sla_penalty_weight),
        use_oracle_workload_projection=args.oracle_workload_projection,
        use_oracle_forecast_observations=args.oracle_forecast_observations,
        seed=args.seed,
    )

    if args.algo == "sac":
        from controllers.sac import make_sac
        model = make_sac(env, seed=args.seed, device=args.device)
    elif args.algo == "td3":
        from controllers.td3 import make_td3
        model = make_td3(env, seed=args.seed, device=args.device)
    else:
        from controllers.ppo import make_ppo
        model = make_ppo(env, seed=args.seed, device=args.device)

    callbacks = []
    if args.checkpoint_freq > 0:
        from stable_baselines3.common.callbacks import CheckpointCallback

        checkpoint_dir = args.checkpoint_dir or SAFE_CHECKPOINT_DIR
        os.makedirs(checkpoint_dir, exist_ok=True)
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix=f"{args.algo}_{args.country}_seed{args.seed}",
        ))
        print(f"checkpoints -> {checkpoint_dir} every {args.checkpoint_freq} steps")

    eval_env = None
    best_model_dir = args.best_model_dir or os.path.join(
        SAFE_BEST_MODEL_DIR, f"{args.algo}_{args.country}_seed{args.seed}")
    if args.eval_freq > 0:
        if args.validation_data_dir is None:
            raise ValueError("--validation-data-dir is required when --eval-freq > 0")
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor

        os.makedirs(best_model_dir, exist_ok=True)
        eval_env = Monitor(CoastalDCContinuousEnv(
            country=args.country,
            data_dir=args.validation_data_dir,
            reward_weights=RewardWeights(w_sla=args.sla_penalty_weight),
            use_oracle_workload_projection=args.oracle_workload_projection,
            use_oracle_forecast_observations=args.oracle_forecast_observations,
            seed=100_000 + args.seed,
        ))
        callbacks.append(EvalCallback(
            eval_env,
            best_model_save_path=best_model_dir,
            log_path=best_model_dir,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        ))
        print(f"validation -> {args.validation_data_dir} every {args.eval_freq} steps")

    callback = callbacks or None
    if len(callbacks) > 1:
        from stable_baselines3.common.callbacks import CallbackList
        callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=args.progress_bar,
        callback=callback,
        log_interval=args.log_interval,
    )

    out_dir = args.out_dir or SAFE_MODEL_DIR
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.algo}_{args.country}_seed{args.seed}"
    canonical_path = os.path.join(out_dir, name)
    if args.eval_freq > 0:
        final_path = canonical_path + "_final"
        model.save(final_path)
        best_path = os.path.join(best_model_dir, "best_model.zip")
        if os.path.exists(best_path):
            shutil.copy2(best_path, canonical_path + ".zip")
            print(f"selected best validation model -> {canonical_path}.zip")
        else:
            model.save(canonical_path)
            print("validation did not run; using final model")
        print(f"saved final model -> {final_path}.zip")
    else:
        model.save(canonical_path)
        print(f"saved model -> {canonical_path}.zip")

    metadata = {
        "schema_version": 1,
        "algorithm": args.algo,
        "country": args.country,
        "seed": args.seed,
        "timesteps": int(model.num_timesteps),
        "training_data_dir": os.path.abspath(args.data_dir),
        "oracle_workload_projection": args.oracle_workload_projection,
        "oracle_forecast_observations": args.oracle_forecast_observations,
        "sla_penalty_weight": args.sla_penalty_weight,
        "requested_device": args.device,
        "resolved_device": str(model.device),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = canonical_path + ".metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved metadata -> {metadata_path}")
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
