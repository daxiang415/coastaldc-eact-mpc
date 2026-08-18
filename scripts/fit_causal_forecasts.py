"""Fit leakage-free causal forecasts and calibration residuals for EACT-MPC."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES  # noqa: E402
from coastaldc_env.forecasting import (  # noqa: E402
    DEFAULT_FORECAST_COLUMNS,
    CausalRidgeForecaster,
    RidgeForecastConfig,
    expanding_window_residuals,
    select_ewma_beta,
    select_ridge_alpha,
)
from coastaldc_env.offshore_wind import WindConfig  # noqa: E402
from coastaldc_env.workload import WorkloadConfig  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_DATA_ROOT = os.path.join(ROOT, "data", "processed_multiyear")
DEFAULT_OUT = os.path.join(ROOT, "results", "causal_forecasts_v3_gated_bias")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=COUNTRIES)
    parser.add_argument("--years", nargs="*", type=int, default=[2023, 2024])
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--calibration-year", type=int, default=2024)
    parser.add_argument("--block-hours", type=int, default=24 * 30)
    parser.add_argument("--residual-window", type=int, default=720)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--alpha-candidates", nargs="*", type=float,
        default=[0.01, 0.1, 1.0, 10.0, 100.0])
    return parser


def load_country_data(data_root: str, country: str, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = os.path.join(data_root, str(year), f"hourly_inputs_{country}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data["timestamp"].duplicated().any() or not data["timestamp"].is_monotonic_increasing:
        raise ValueError(f"Non-unique or unsorted timestamps for {country}")
    return data


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "calibration_summary.csv")
    if args.resume and os.path.exists(summary_path):
        existing_summary = pd.read_csv(summary_path)
        summaries: list[dict] = existing_summary.to_dict("records")
    else:
        existing_summary = pd.DataFrame()
        summaries = []
    wl_cfg = WorkloadConfig()
    wind_cfg = WindConfig()
    bounds = {
        "fixed_load_mw": (0.0, wl_cfg.it_capacity_mw),
        "flexible_arrival_mw": (0.0, wl_cfg.it_capacity_mw),
        "sst_c": (-2.0, 40.0),
        "wind_mw": (0.0, wind_cfg.wind_capacity_mw),
        "carbon_kg_per_mwh": (0.0, 2000.0),
    }

    for country in args.countries:
        model_path = os.path.join(args.out_dir, f"causal_ridge_{country}.npz")
        residual_path = os.path.join(
            args.out_dir, f"calibration_residuals_{country}.npz")
        complete_existing = (
            args.resume
            and os.path.exists(model_path)
            and os.path.exists(residual_path)
            and not existing_summary.empty
            and country in set(existing_summary["country"].astype(str)))
        if complete_existing:
            print(f"{country}: existing artifacts found, skipped")
            continue
        summaries = [
            row for row in summaries if str(row.get("country")) != country]
        data = load_country_data(args.data_root, country, args.years)
        calibration_positions = np.flatnonzero(
            data["timestamp"].dt.year.to_numpy() >= args.calibration_year)
        if not len(calibration_positions):
            raise ValueError(f"No calibration rows for {country}")
        calibration_start = int(calibration_positions[0])
        alpha = select_ridge_alpha(
            data.iloc[:calibration_start],
            args.alpha_candidates,
            columns=DEFAULT_FORECAST_COLUMNS,
            horizon=args.horizon,
            bounds=bounds,
        )
        config = RidgeForecastConfig(horizon=args.horizon, alpha=alpha)
        origins, residuals = expanding_window_residuals(
            data,
            calibration_start,
            config=config,
            columns=DEFAULT_FORECAST_COLUMNS,
            block_hours=args.block_hours,
            bounds=bounds,
        )
        model = CausalRidgeForecaster(
            config=config, columns=DEFAULT_FORECAST_COLUMNS, bounds=bounds).fit(data)
        model.save(model_path)

        residual_tail = residuals[-args.residual_window:]
        beta = select_ewma_beta(
            residuals, DEFAULT_FORECAST_COLUMNS,
            candidates=[0.0, 0.01, 0.05, 0.10, 0.20])
        history_tail = data.iloc[-max(config.lags):]
        np.savez_compressed(
            residual_path,
            origins=origins[-args.residual_window:],
            residuals=residual_tail,
            columns=np.asarray(DEFAULT_FORECAST_COLUMNS),
            horizon=np.asarray(args.horizon),
            beta=np.asarray([beta[column] for column in DEFAULT_FORECAST_COLUMNS]),
            history_values=history_tail.loc[:, DEFAULT_FORECAST_COLUMNS].to_numpy(),
            history_timestamps=history_tail["timestamp"].to_numpy(dtype="datetime64[ns]"),
        )
        for column_index, column in enumerate(DEFAULT_FORECAST_COLUMNS):
            values = residuals[:, column_index, :].astype(float)
            summaries.append({
                "country": country,
                "column": column,
                "alpha": alpha,
                "beta": beta[column],
                "n_origins": len(origins),
                "mae": float(np.abs(values).mean()),
                "rmse": float(np.sqrt(np.square(values).mean())),
                "model_path": model_path,
                "residual_path": residual_path,
            })
        print(
            f"{country}: alpha={alpha:g}, calibration_origins={len(origins)}, "
            f"model={model_path}")

    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary.sort_values(["country", "column"]).reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
