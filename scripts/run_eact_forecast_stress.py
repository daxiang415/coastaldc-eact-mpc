"""Run the MPC-only forecast-distribution-shift screening matrix."""

from __future__ import annotations

import argparse
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.run_eact_paper_experiments import (  # noqa: E402
    Job,
    job_is_complete,
    run_job,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVALUATOR = os.path.join(ROOT, "scripts", "evaluate_eact_mpc.py")
DEFAULT_OUT = os.path.join(ROOT, "results", "eact_forecast_stress_v1")
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed_multiyear", "2025")
SCENARIOS = ("none", "adverse_bias", "noise", "combined")
CONTROLLERS = ("nominal", "static", "eact")
SEASON_TIMESTAMPS = (
    "2025-01-15T00:00:00",
    "2025-04-15T00:00:00",
    "2025-07-15T00:00:00",
    "2025-10-15T00:00:00",
)


def _scale_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=["JPN", "CHN", "NOR"])
    parser.add_argument("--scenarios", nargs="*", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--controllers", nargs="*", choices=CONTROLLERS,
                        default=list(CONTROLLERS))
    parser.add_argument("--stress-scale", type=float, default=1.0)
    parser.add_argument("--stress-start-step", type=int, default=0)
    parser.add_argument("--stress-seed", type=int, default=20260717)
    parser.add_argument("--adaptive-beta-floor", type=float, default=0.10)
    parser.add_argument("--weight-grid", type=float, default=1.0)
    parser.add_argument("--weight-co2", type=float, default=2.0)
    parser.add_argument("--weight-total", type=float, default=0.2)
    parser.add_argument("--weight-smooth", type=float, default=0.5)
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument("--start-timestamps", nargs="*",
                        default=list(SEASON_TIMESTAMPS))
    parser.add_argument(
        "--thermal-safety-shield",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_jobs(args) -> list[Job]:
    if args.stress_scale < 0.0 or args.stress_start_step < 0 or args.stress_seed < 0:
        raise ValueError("Forecast stress settings must be nonnegative")
    if not 0.0 <= args.adaptive_beta_floor <= 1.0:
        raise ValueError("Adaptive beta floor must be in [0, 1]")
    objective_weights = (
        args.weight_grid, args.weight_co2,
        args.weight_total, args.weight_smooth,
    )
    if any(not math.isfinite(value) or value < 0.0
           for value in objective_weights):
        raise ValueError("Objective weights must be finite and nonnegative")
    if args.episode_hours <= 0 or args.max_workers <= 0:
        raise ValueError("Episode hours and worker count must be positive")
    jobs = []
    shield_slug = "shield_on" if args.thermal_safety_shield else "shield_off"
    beta_slug = _scale_slug(args.adaptive_beta_floor)
    weight_slug = (
        f"wg{_scale_slug(args.weight_grid)}_"
        f"wc{_scale_slug(args.weight_co2)}_"
        f"wt{_scale_slug(args.weight_total)}_"
        f"ws{_scale_slug(args.weight_smooth)}"
    )
    controller_slug = "-".join(args.controllers)
    for scenario in args.scenarios:
        scale = 0.0 if scenario == "none" else float(args.stress_scale)
        scale_slug = _scale_slug(scale)
        for country in args.countries:
            tag = (
                f"stress_{scenario}_s{scale_slug}_b{beta_slug}_"
                f"{weight_slug}_c{controller_slug}_{country}_{shield_slug}")
            out_dir = os.path.join(
                args.out_root, f"b{beta_slug}", weight_slug,
                f"c{controller_slug}", scenario,
                f"s{scale_slug}", country)
            command = [
                sys.executable,
                EVALUATOR,
                "--countries", country,
                "--controllers", *args.controllers,
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
                "--adaptive-beta-floor", str(args.adaptive_beta_floor),
                "--weight-grid", str(args.weight_grid),
                "--weight-co2", str(args.weight_co2),
                "--weight-total", str(args.weight_total),
                "--weight-smooth", str(args.weight_smooth),
                "--forecast-stress", scenario,
                "--forecast-stress-scale", str(scale),
                "--forecast-stress-start-step", str(args.stress_start_step),
                "--forecast-stress-seed", str(args.stress_seed),
                "--progress-every", str(args.episode_hours),
                "--out-dir", out_dir,
                "--tag", tag,
                "--no-oracle-workload-projection",
            ]
            command.append(
                "--thermal-safety-shield"
                if args.thermal_safety_shield
                else "--no-thermal-safety-shield")
            jobs.append(Job("forecast_stress", tag, tuple(command), out_dir, tag))
    return jobs


def main() -> None:
    args = build_parser().parse_args()
    jobs = build_jobs(args)
    pending = [job for job in jobs if not (args.resume and job_is_complete(job))]
    print(
        f"forecast-stress jobs: total={len(jobs)} pending={len(pending)} "
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
        raise RuntimeError(f"Forecast-stress jobs failed: {failures}")


if __name__ == "__main__":
    main()
