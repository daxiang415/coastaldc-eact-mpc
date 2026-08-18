"""Run the final 15-country no-shift and persistent-shift MPC matrices."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES  # noqa: E402
from scripts.run_eact_paper_experiments import (  # noqa: E402
    Job,
    job_is_complete,
    run_job,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVALUATOR = os.path.join(ROOT, "scripts", "evaluate_eact_mpc.py")
DEFAULT_OUT = os.path.join(ROOT, "results", "eact_final_seasonal_v1")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed_multiyear", "2025")
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias")
SCENARIOS = ("none", "adverse_bias")
SEASON_TIMESTAMPS = (
    "2025-01-15T00:00:00",
    "2025-04-15T00:00:00",
    "2025-07-15T00:00:00",
    "2025-10-15T00:00:00",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=list(COUNTRIES))
    parser.add_argument("--scenarios", nargs="*", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument("--start-timestamps", nargs="*",
                        default=list(SEASON_TIMESTAMPS))
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_jobs(args) -> list[Job]:
    if args.episode_hours <= 0 or args.max_workers <= 0:
        raise ValueError("Episode hours and worker count must be positive")
    jobs = []
    for scenario in args.scenarios:
        scale = 0.0 if scenario == "none" else 1.0
        stage = "e1_no_shift" if scenario == "none" else "e2_shift"
        for country in args.countries:
            tag = f"{stage}_{country}_b0p10_primary_shield_off"
            out_dir = os.path.join(args.out_root, stage, country)
            command = (
                sys.executable,
                EVALUATOR,
                "--countries", country,
                "--controllers", "nominal", "static", "eact",
                "--data-dir", args.data_dir,
                "--forecast-dir", args.forecast_dir,
                "--episode-hours", str(args.episode_hours),
                "--start-timestamps", *args.start_timestamps,
                "--horizon", "24",
                "--maxiter", "20",
                "--block-hours", "6",
                "--gamma", "0.995",
                "--confidence", "0.90",
                "--constraint-tolerance", "0.0001",
                "--intervention-weight", "0.0",
                "--adaptive-beta-floor", "0.10",
                "--weight-grid", "1.0",
                "--weight-co2", "2.0",
                "--weight-total", "0.2",
                "--weight-smooth", "0.5",
                "--forecast-stress", scenario,
                "--forecast-stress-scale", str(scale),
                "--forecast-stress-start-step", "0",
                "--forecast-stress-seed", "20260717",
                "--progress-every", str(args.episode_hours),
                "--out-dir", out_dir,
                "--tag", tag,
                "--no-oracle-workload-projection",
                "--no-thermal-safety-shield",
            )
            jobs.append(Job(stage, tag, command, out_dir, tag))
    return jobs


def main() -> None:
    args = build_parser().parse_args()
    jobs = build_jobs(args)
    pending = [job for job in jobs if not (args.resume and job_is_complete(job))]
    print(
        f"final seasonal jobs: total={len(jobs)} pending={len(pending)} "
        f"workers={args.max_workers}",
        flush=True,
    )
    if args.dry_run:
        for job in pending:
            print(" ".join(job.command))
        return
    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_job, job): job for job in pending}
        for future in as_completed(futures):
            name, status = future.result()
            print(f"{name}: {status}", flush=True)
            if status != "complete":
                failures.append((name, status))
    if failures:
        raise RuntimeError(f"Final seasonal jobs failed: {failures}")


if __name__ == "__main__":
    main()
