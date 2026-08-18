"""Run the preregistered EACT paper experiment matrix with resume support."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES  # noqa: E402


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVALUATOR = os.path.join(ROOT, "scripts", "evaluate_eact_mpc.py")
DEFAULT_OUT = os.path.join(ROOT, "results", "eact_paper_v1")
DEFAULT_FORECAST_DIR = os.path.join(
    ROOT, "results", "causal_forecasts_v3_gated_bias")
SEASON_DATES = ("01-15", "04-15", "07-15", "10-15")


@dataclass(frozen=True)
class Job:
    stage: str
    name: str
    command: tuple[str, ...]
    out_dir: str
    tag: str

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.out_dir, f"manifest_{self.tag}.json")

    @property
    def log_path(self) -> str:
        return os.path.join(self.out_dir, f"run_{self.tag}.log")


def _weight_slug(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _season_timestamps(year: int) -> list[str]:
    return [f"{year}-{date}T00:00:00" for date in SEASON_DATES]


def _evaluation_job(
    *,
    stage: str,
    name: str,
    out_dir: str,
    tag: str,
    countries: list[str],
    controllers: list[str],
    data_dir: str,
    forecast_dir: str,
    intervention_weight: float,
    horizon: int = 24,
    confidence: float = 0.90,
    start_timestamps: list[str] | None = None,
    continuous_year: bool = False,
) -> Job:
    command = [
        sys.executable,
        EVALUATOR,
        "--countries", *countries,
        "--controllers", *controllers,
        "--data-dir", data_dir,
        "--forecast-dir", forecast_dir,
        "--intervention-weight", str(intervention_weight),
        "--horizon", str(horizon),
        "--confidence", str(confidence),
        "--out-dir", out_dir,
        "--tag", tag,
    ]
    if start_timestamps is not None:
        command.extend(["--start-timestamps", *start_timestamps])
    if continuous_year:
        command.append("--continuous-year")
    return Job(stage, name, tuple(command), out_dir, tag)


def build_jobs(args) -> list[Job]:
    stages = set(args.stages)
    if "all" in stages:
        stages = {"calibration", "seasonal", "annual", "ablation", "sensitivity"}
    jobs: list[Job] = []

    if "calibration" in stages:
        for year in (2023, 2024):
            data_dir = os.path.join(ROOT, "data", "processed_multiyear", str(year))
            for country in args.annual_countries:
                for weight in args.calibration_weights:
                    slug = _weight_slug(weight)
                    tag = f"cal_{year}_{country}_w{slug}"
                    out_dir = os.path.join(
                        args.out_root, "calibration", str(year), country, f"w{slug}")
                    jobs.append(_evaluation_job(
                        stage="calibration",
                        name=tag,
                        out_dir=out_dir,
                        tag=tag,
                        countries=[country],
                        controllers=["eact"],
                        data_dir=data_dir,
                        forecast_dir=args.forecast_dir,
                        intervention_weight=weight,
                        start_timestamps=_season_timestamps(year),
                    ))

    if "seasonal" in stages:
        for country in args.countries:
            tag = f"seasonal_2025_{country}"
            jobs.append(_evaluation_job(
                stage="seasonal",
                name=tag,
                out_dir=os.path.join(args.out_root, "seasonal", country),
                tag=tag,
                countries=[country],
                controllers=[
                    "no_control", "rule_based", "nominal", "static", "eact"],
                data_dir=os.path.join(
                    ROOT, "data", "processed_multiyear", "2025"),
                forecast_dir=args.forecast_dir,
                intervention_weight=args.intervention_weight,
                start_timestamps=_season_timestamps(2025),
            ))

    if "annual" in stages:
        for country in args.annual_countries:
            controllers = [
                "no_control", "rule_based", "nominal", "static", "eact"]
            tag = f"annual_2025_{country}"
            jobs.append(_evaluation_job(
                stage="annual",
                name=tag,
                out_dir=os.path.join(args.out_root, "annual", country),
                tag=tag,
                countries=[country],
                controllers=controllers,
                data_dir=os.path.join(
                    ROOT, "data", "processed_multiyear", "2025"),
                forecast_dir=args.forecast_dir,
                intervention_weight=args.intervention_weight,
                continuous_year=True,
            ))

    if "ablation" in stages:
        for country in args.annual_countries:
            tag = f"ablation_2025_{country}"
            controllers = ["static", "eact", "eact_no_bias"]
            if not np.isclose(args.intervention_weight, 0.0):
                controllers.append("eact_no_intervention")
            jobs.append(_evaluation_job(
                stage="ablation",
                name=tag,
                out_dir=os.path.join(args.out_root, "ablation", country),
                tag=tag,
                countries=[country],
                controllers=controllers,
                data_dir=os.path.join(
                    ROOT, "data", "processed_multiyear", "2025"),
                forecast_dir=args.forecast_dir,
                intervention_weight=args.intervention_weight,
                start_timestamps=_season_timestamps(2025),
            ))

    if "sensitivity" in stages:
        for horizon in (12, 24):
            for confidence in (0.80, 0.90, 0.95):
                conf_slug = str(int(round(confidence * 100)))
                tag = f"sensitivity_JPN_h{horizon}_c{conf_slug}"
                jobs.append(_evaluation_job(
                    stage="sensitivity",
                    name=tag,
                    out_dir=os.path.join(
                        args.out_root, "sensitivity", f"h{horizon}_c{conf_slug}"),
                    tag=tag,
                    countries=["JPN"],
                    controllers=["static", "eact"],
                    data_dir=os.path.join(
                        ROOT, "data", "processed_multiyear", "2025"),
                    forecast_dir=args.forecast_dir,
                    intervention_weight=args.intervention_weight,
                    horizon=horizon,
                    confidence=confidence,
                    start_timestamps=_season_timestamps(2025),
                ))
    return jobs


def job_is_complete(job: Job) -> bool:
    if not os.path.exists(job.manifest_path):
        return False
    try:
        with open(job.manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("status") != "complete":
        return False
    if manifest.get("command") != list(job.command):
        return False
    outputs = manifest.get("outputs", {})
    required = {"summary", "episodes", "weekly", "hourly", "solver"}
    return required.issubset(outputs) and all(
        os.path.exists(outputs[name]["path"]) for name in required)


def run_job(job: Job, *, dry_run: bool = False) -> tuple[str, str]:
    if dry_run:
        return job.name, "dry-run"
    os.makedirs(job.out_dir, exist_ok=True)
    with open(job.log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            job.command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        return job.name, f"failed:{completed.returncode}"
    return job.name, "complete" if job_is_complete(job) else "missing-manifest"


def execution_priority(job: Job) -> int:
    """Start long annual jobs first so shorter stages fill remaining slots."""
    return {
        "annual": 0,
        "calibration": 1,
        "seasonal": 2,
        "ablation": 3,
        "sensitivity": 4,
    }.get(job.stage, 99)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages", nargs="+",
        choices=["all", "calibration", "seasonal", "annual", "ablation", "sensitivity"],
        default=["all"],
    )
    parser.add_argument("--countries", nargs="*", default=COUNTRIES)
    parser.add_argument("--annual-countries", nargs="*", default=["JPN", "CHN", "NOR"])
    parser.add_argument("--calibration-weights", nargs="*", type=float,
                        default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--intervention-weight", type=float, default=0.10)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--out-root", default=DEFAULT_OUT)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    jobs = build_jobs(args)
    pending = sorted(
        [job for job in jobs if not (args.resume and job_is_complete(job))],
        key=execution_priority,
    )
    skipped = len(jobs) - len(pending)
    print(f"jobs={len(jobs)} pending={len(pending)} resumed={skipped}")
    if args.dry_run:
        for job in pending:
            print(f"{job.stage:12s} {job.name}: {' '.join(job.command)}")
        return

    failures = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_job, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            name, status = future.result()
            print(f"{job.stage:12s} {name}: {status}", flush=True)
            if status != "complete":
                failures.append((job, status))
    if failures:
        details = ", ".join(f"{job.name}={status}" for job, status in failures)
        raise RuntimeError(f"Experiment jobs failed: {details}")


if __name__ == "__main__":
    main()
