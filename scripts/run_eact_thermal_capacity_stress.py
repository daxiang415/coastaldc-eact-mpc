"""Run the preregistered near-capacity thermal stress matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.run_eact_paper_experiments import (  # noqa: E402
    Job,
    job_is_complete,
    run_job,
)


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVALUATOR = os.path.join(ROOT, "scripts", "evaluate_eact_mpc.py")
DEFAULT_OUT = os.path.join(
    ROOT, "results", "ashrae_inlet_thermal_capacity_stress_v1")
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed_multiyear", "2025")
COUNTRIES = ("JPN", "CHN", "NOR")
CONTROLLERS = ("nominal", "static", "eact")
SCENARIOS = ("none", "adverse_bias")
AVAILABILITY_LEVELS = (1.0, 0.75, 0.50)
QUARTERS = (
    ("Q1", "2025-01-01T00:00:00", "2025-04-01T00:00:00"),
    ("Q2", "2025-04-01T00:00:00", "2025-07-01T00:00:00"),
    ("Q3", "2025-07-01T00:00:00", "2025-10-01T00:00:00"),
    ("Q4", "2025-10-01T00:00:00", "2026-01-01T00:00:00"),
)


def _value_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", choices=COUNTRIES,
                        default=list(COUNTRIES))
    parser.add_argument("--controllers", nargs="*", choices=CONTROLLERS,
                        default=list(CONTROLLERS))
    parser.add_argument("--scenarios", nargs="*", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--availability-levels", nargs="*", type=float,
                        default=list(AVAILABILITY_LEVELS))
    parser.add_argument("--episode-hours", type=int, default=168)
    parser.add_argument("--forecast-stress-scale", type=float, default=1.0)
    parser.add_argument("--forecast-stress-seed", type=int, default=20260717)
    parser.add_argument("--adaptive-beta-floor", type=float, default=0.10)
    parser.add_argument("--weight-grid", type=float, default=1.0)
    parser.add_argument("--weight-co2", type=float, default=2.0)
    parser.add_argument("--weight-total", type=float, default=0.2)
    parser.add_argument("--weight-smooth", type=float, default=0.5)
    parser.add_argument(
        "--thermal-safety-shield",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def select_quarterly_high_load_windows(
    path: str,
    *,
    episode_hours: int,
    trailing_forecast_hours: int = 24,
) -> list[dict]:
    if episode_hours <= 0 or trailing_forecast_hours < 0:
        raise ValueError("Window and trailing forecast hours are invalid")
    frame = pd.read_csv(path)
    required = {"timestamp", "fixed_load_mw", "flexible_arrival_mw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"High-load selection missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("High-load timestamps must be unique and increasing")
    spacing = timestamps.diff().dropna()
    if not spacing.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("High-load selection requires contiguous hourly data")
    arrival = (
        frame["fixed_load_mw"].to_numpy(dtype=float)
        + frame["flexible_arrival_mw"].to_numpy(dtype=float)
    )
    if not np.isfinite(arrival).all() or np.any(arrival < 0.0):
        raise ValueError("IT arrival data must be finite and nonnegative")

    latest_start = len(frame) - episode_hours - trailing_forecast_hours
    windows = []
    for quarter, start_text, end_text in QUARTERS:
        quarter_start = pd.Timestamp(start_text)
        quarter_end = pd.Timestamp(end_text)
        candidates = [
            index
            for index in range(max(0, latest_start + 1))
            if timestamps.iloc[index] >= quarter_start
            and timestamps.iloc[index + episode_hours - 1] < quarter_end
        ]
        if not candidates:
            raise ValueError(f"No complete high-load window for {quarter}")
        means = np.asarray([
            arrival[index:index + episode_hours].mean()
            for index in candidates
        ])
        best = candidates[int(np.argmax(means))]
        values = arrival[best:best + episode_hours]
        windows.append({
            "quarter": quarter,
            "start_hour": int(best),
            "start_timestamp": timestamps.iloc[best].strftime(
                "%Y-%m-%dT%H:%M:%S"),
            "mean_it_arrival_mw": float(values.mean()),
            "p95_it_arrival_mw": float(np.quantile(values, 0.95)),
            "max_it_arrival_mw": float(values.max()),
        })
    return windows


def _validate_args(args) -> None:
    if not args.countries or not args.controllers:
        raise ValueError("At least one country and controller are required")
    if not args.scenarios or not args.availability_levels:
        raise ValueError("At least one scenario and availability level are required")
    if args.episode_hours <= 0 or args.max_workers <= 0:
        raise ValueError("Episode hours and worker count must be positive")
    if (not math.isfinite(args.forecast_stress_scale)
            or args.forecast_stress_scale < 0.0
            or args.forecast_stress_seed < 0):
        raise ValueError("Forecast stress settings are invalid")
    if not 0.0 <= args.adaptive_beta_floor <= 1.0:
        raise ValueError("Adaptive beta floor must be in [0, 1]")
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0
           for value in args.availability_levels):
        raise ValueError("Availability levels must be finite and in (0, 1]")
    weights = (
        args.weight_grid, args.weight_co2,
        args.weight_total, args.weight_smooth,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Objective weights must be finite and nonnegative")


def selected_windows(args) -> dict[str, list[dict]]:
    return {
        country: select_quarterly_high_load_windows(
            os.path.join(args.data_dir, f"hourly_inputs_{country}.csv"),
            episode_hours=args.episode_hours,
        )
        for country in args.countries
    }


def build_jobs(args) -> list[Job]:
    _validate_args(args)
    windows_by_country = selected_windows(args)
    jobs = []
    shield_slug = "shield_on" if args.thermal_safety_shield else "shield_off"
    controller_slug = "-".join(args.controllers)
    for availability in args.availability_levels:
        availability_slug = _value_slug(availability)
        for scenario in args.scenarios:
            stress_scale = (
                0.0 if scenario == "none" else args.forecast_stress_scale)
            stress_slug = _value_slug(stress_scale)
            for country in args.countries:
                starts = [
                    item["start_timestamp"]
                    for item in windows_by_country[country]
                ]
                tag = (
                    f"capacity_a{availability_slug}_{scenario}_s{stress_slug}_"
                    f"c{controller_slug}_{country}_{shield_slug}")
                out_dir = os.path.join(
                    args.out_root,
                    f"a{availability_slug}",
                    scenario,
                    f"s{stress_slug}",
                    country,
                )
                command = [
                    sys.executable,
                    EVALUATOR,
                    "--countries", country,
                    "--controllers", *args.controllers,
                    "--data-dir", args.data_dir,
                    "--forecast-dir", args.forecast_dir,
                    "--episode-hours", str(args.episode_hours),
                    "--start-timestamps", *starts,
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
                    "--cooling-conductance-multiplier", str(availability),
                    "--forecast-stress", scenario,
                    "--forecast-stress-scale", str(stress_scale),
                    "--forecast-stress-start-step", "0",
                    "--forecast-stress-seed", str(args.forecast_stress_seed),
                    "--progress-every", str(args.episode_hours),
                    "--out-dir", out_dir,
                    "--tag", tag,
                    "--no-oracle-workload-projection",
                ]
                command.append(
                    "--thermal-safety-shield"
                    if args.thermal_safety_shield
                    else "--no-thermal-safety-shield")
                jobs.append(Job(
                    "thermal_capacity_stress",
                    tag,
                    tuple(command),
                    out_dir,
                    tag,
                ))
    return jobs


def _write_window_manifest(args, windows_by_country: dict[str, list[dict]]) -> None:
    os.makedirs(args.out_root, exist_ok=True)
    path = os.path.join(args.out_root, "high_load_window_selection.json")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            "maximum mean fixed_load_mw plus flexible_arrival_mw over one "
            "complete episode within each calendar quarter"),
        "episode_hours": args.episode_hours,
        "trailing_forecast_hours": 24,
        "data_dir": os.path.abspath(args.data_dir),
        "countries": list(args.countries),
        "windows": windows_by_country,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved -> {path}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    windows_by_country = selected_windows(args)
    _write_window_manifest(args, windows_by_country)
    jobs = build_jobs(args)
    pending = [job for job in jobs if not (args.resume and job_is_complete(job))]
    print(
        f"thermal-capacity jobs: total={len(jobs)} pending={len(pending)} "
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
        raise RuntimeError(f"Thermal-capacity jobs failed: {failures}")


if __name__ == "__main__":
    main()
