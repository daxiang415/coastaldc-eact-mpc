"""Run paper-scale SAC training jobs sequentially.

Usage:
    python scripts/train_sac_batch.py --device cuda --dry-run
    python scripts/train_sac_batch.py --countries JPN NOR --seeds 0 1 2 --device cuda
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TRAIN_DATA_DIR = os.path.join(
    ROOT, "data", "processed_multiyear", "train_2023_2024")


@dataclass(frozen=True)
class TrainingJob:
    country: str
    seed: int
    command: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=COUNTRIES)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--data-dir", default=TRAIN_DATA_DIR)
    parser.add_argument("--validation-data-dir", default=None)
    parser.add_argument("--out-dir", default=os.path.join(
        ROOT, "results", "models_sac_causal_v1"))
    parser.add_argument("--checkpoint-dir",
                        default=os.path.join(ROOT, "results", "checkpoints_sac_causal_v1"))
    parser.add_argument("--best-model-dir", default=os.path.join(
        ROOT, "results", "best_models_sac_causal_v1"))
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--sla-penalty-weight", type=float, default=50.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true",
                        help="train even when the final model zip already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the training queue without running it")
    return parser


def build_jobs(args: argparse.Namespace) -> list[TrainingJob]:
    jobs: list[TrainingJob] = []
    train_script = str(Path(ROOT) / "scripts" / "train_online_rl.py")
    for country in args.countries:
        for seed in args.seeds:
            model_path = Path(args.out_dir) / f"sac_{country}_seed{seed}.zip"
            if model_path.exists() and not args.rerun_existing:
                continue
            command = [
                sys.executable,
                train_script,
                "--algo",
                "sac",
                "--country",
                country,
                "--seed",
                str(seed),
                "--timesteps",
                str(args.timesteps),
                "--device",
                args.device,
                "--data-dir",
                args.data_dir,
                "--out-dir",
                args.out_dir,
                "--checkpoint-dir",
                args.checkpoint_dir,
                "--checkpoint-freq",
                str(args.checkpoint_freq),
                "--log-interval",
                str(args.log_interval),
                "--eval-freq",
                str(args.eval_freq),
                "--sla-penalty-weight",
                str(args.sla_penalty_weight),
                "--no-oracle-workload-projection",
                "--no-oracle-forecast-observations",
            ]
            if args.eval_freq > 0:
                if args.validation_data_dir is None:
                    raise ValueError(
                        "--validation-data-dir is required when --eval-freq > 0")
                command.extend([
                    "--validation-data-dir",
                    args.validation_data_dir,
                    "--eval-episodes",
                    str(args.eval_episodes),
                    "--best-model-dir",
                    str(Path(args.best_model_dir) / f"sac_{country}_seed{seed}"),
                ])
            if args.progress_bar:
                command.append("--progress-bar")
            jobs.append(TrainingJob(country=country, seed=seed, command=command))
    return jobs


def main() -> None:
    args = build_parser().parse_args()
    jobs = build_jobs(args)
    if not jobs:
        print("no SAC training jobs to run")
        return

    print(f"SAC training jobs: {len(jobs)}")
    for idx, job in enumerate(jobs, start=1):
        command_text = subprocess.list2cmdline(job.command)
        print(f"[{idx}/{len(jobs)}] {job.country} seed={job.seed}")
        print(command_text)
        if not args.dry_run:
            subprocess.run(job.command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
