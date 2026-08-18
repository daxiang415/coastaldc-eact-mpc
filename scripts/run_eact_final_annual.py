"""Run the final MPC controllers on continuous 2025 trajectories."""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.run_eact_final_seasonal import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_FORECAST_DIR,
    EVALUATOR,
)
from scripts.run_eact_paper_experiments import (  # noqa: E402
    Job,
    job_is_complete,
    run_job,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_OUT = os.path.join(ROOT, "results", "eact_final_annual_v2")
COUNTRIES = ("JPN", "CHN", "NOR")
CONTROLLERS = ("nominal", "static", "eact")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=list(COUNTRIES))
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--max-workers", type=int, default=9)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_jobs(args) -> list[Job]:
    if args.max_workers <= 0:
        raise ValueError("Worker count must be positive")
    jobs = []
    for country in args.countries:
        for controller in CONTROLLERS:
            tag = (
                f"e3_annual_{country}_{controller}_"
                "b0p10_primary_shield_on"
            )
            out_dir = os.path.join(args.out_root, country, controller)
            command = (
                sys.executable,
                EVALUATOR,
                "--countries", country,
                "--controllers", controller,
                "--data-dir", args.data_dir,
                "--forecast-dir", args.forecast_dir,
                "--continuous-year",
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
                "--forecast-stress", "none",
                "--forecast-stress-scale", "0.0",
                "--forecast-stress-start-step", "0",
                "--forecast-stress-seed", "20260717",
                "--progress-every", "168",
                "--out-dir", out_dir,
                "--tag", tag,
                "--no-oracle-workload-projection",
                "--thermal-safety-shield",
            )
            jobs.append(Job("e3_annual", tag, command, out_dir, tag))
    return jobs


def main() -> None:
    args = build_parser().parse_args()
    jobs = build_jobs(args)
    pending = [job for job in jobs if not (args.resume and job_is_complete(job))]
    print(
        f"final annual jobs: total={len(jobs)} pending={len(pending)} "
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
        raise RuntimeError(f"Final annual jobs failed: {failures}")


if __name__ == "__main__":
    main()
