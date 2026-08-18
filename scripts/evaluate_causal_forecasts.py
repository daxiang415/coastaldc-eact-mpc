"""Evaluate causal forecast accuracy and one-sided risk-bound coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env import COUNTRIES  # noqa: E402
from coastaldc_env.forecasting import (  # noqa: E402
    CausalRidgeForecaster,
    OnlineResidualAdaptor,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEFAULT_DATA_DIR = os.path.join(ROOT, "data", "processed_multiyear", "2025")
DEFAULT_FORECAST_DIR = os.path.join(ROOT, "results", "causal_forecasts_v3_gated_bias")
UPPER_BOUND_COLUMNS = {
    "fixed_load_mw", "flexible_arrival_mw", "sst_c", "carbon_kg_per_mwh"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=COUNTRIES)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--forecast-dir", default=DEFAULT_FORECAST_DIR)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--adaptive-beta-floor", type=float, default=0.0)
    parser.add_argument("--start-hour", type=int, default=0)
    parser.add_argument("--hours", type=int)
    parser.add_argument("--out-dir", default=os.path.join(
        ROOT, "results", "causal_forecast_evaluation"))
    parser.add_argument("--tag", default="2025")
    return parser


def load_artifacts(forecast_dir: str, country: str):
    model = CausalRidgeForecaster.load(os.path.join(
        forecast_dir, f"causal_ridge_{country}.npz"))
    residual_path = os.path.join(
        forecast_dir, f"calibration_residuals_{country}.npz")
    with np.load(residual_path, allow_pickle=False) as archive:
        columns = tuple(str(value) for value in archive["columns"].tolist())
        if columns != tuple(model.columns):
            raise ValueError("Forecast model and calibration columns differ")
        residuals = archive["residuals"].astype(float)
        beta = dict(zip(columns, archive["beta"].astype(float)))
        history = pd.DataFrame(
            archive["history_values"].astype(float), columns=columns)
        history.insert(
            0, "timestamp", pd.to_datetime(archive["history_timestamps"]))
    return model, residuals, beta, history


def evaluate_sequence(
    data: pd.DataFrame,
    model: CausalRidgeForecaster,
    calibration_residuals: np.ndarray,
    beta: dict[str, float],
    history_tail: pd.DataFrame,
    *,
    horizon: int = 24,
    start_hour: int = 0,
    hours: int | None = None,
    adaptive_beta_floor: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate and horizon-specific causal forecast metrics."""
    if not 1 <= horizon <= model.config.horizon:
        raise ValueError("Evaluation horizon exceeds the fitted forecast horizon")
    if not 0.0 <= adaptive_beta_floor <= 1.0:
        raise ValueError("Adaptive beta floor must be in [0, 1]")
    frame = data.copy().reset_index(drop=True)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if not 0 <= start_hour < len(frame) - horizon:
        raise ValueError("start_hour leaves no complete forecast horizon")
    max_hours = len(frame) - horizon - start_hour
    evaluation_hours = max_hours if hours is None else min(int(hours), max_hours)
    if evaluation_hours <= 0:
        raise ValueError("hours must be positive")

    columns = tuple(model.columns)
    static = OnlineResidualAdaptor(
        columns, model.config.horizon, beta=beta, window=720,
        confidence=0.90, initial_residuals=calibration_residuals)
    adaptive = OnlineResidualAdaptor(
        columns, model.config.horizon,
        beta={
            column: max(float(value), float(adaptive_beta_floor))
            for column, value in beta.items()
        },
        window=720,
        confidence=0.90, initial_residuals=calibration_residuals)
    modes = ("nominal", "static", "adaptive")
    shape = (len(modes), len(columns), horizon)
    counts = np.zeros(shape, dtype=np.int64)
    error_sum = np.zeros(shape, dtype=float)
    absolute_error_sum = np.zeros(shape, dtype=float)
    squared_error_sum = np.zeros(shape, dtype=float)
    bound_counts = np.zeros(shape, dtype=np.int64)
    coverage_sum = np.zeros(shape, dtype=float)
    interval_width_sum = np.zeros(shape, dtype=float)
    static_bounds = static.one_sided_bounds(horizon=horizon)

    for local_index in range(evaluation_hours):
        data_index = start_hour + local_index
        if local_index > 0:
            adaptive.observe(data_index, {
                column: float(frame.iloc[data_index][column])
                for column in columns})
        observed_columns = (
            ["timestamp", *columns]
            if "timestamp" in frame.columns else list(columns))
        max_lag = max(model.config.lags)
        observed_start = max(0, data_index - max_lag + 1)
        observed = frame.iloc[
            observed_start:data_index + 1][observed_columns]
        needed_history = max(0, max_lag - len(observed))
        history = pd.concat([
            history_tail.iloc[-needed_history:] if needed_history else
            history_tail.iloc[0:0],
            observed,
        ], ignore_index=True)
        if "timestamp" in history.columns:
            history = history.drop_duplicates(
                "timestamp", keep="last").reset_index(drop=True)
        base = model.predict(history, origin=len(history))
        predictions = {
            "nominal": base,
            "static": static.correct(base),
            "adaptive": adaptive.register_forecast(data_index + 1, base),
        }
        bounds = {
            "static": static_bounds,
            "adaptive": adaptive.one_sided_bounds(horizon=horizon),
        }
        actual = frame.iloc[
            data_index + 1:data_index + 1 + horizon]

        for mode_index, mode in enumerate(modes):
            mode_predictions = predictions[mode]
            for column_index, column in enumerate(columns):
                prediction = _clip(
                    mode_predictions[column][:horizon], model.bounds[column])
                actual_values = actual[column].to_numpy(dtype=float)
                errors = actual_values - prediction
                counts[mode_index, column_index] += 1
                error_sum[mode_index, column_index] += errors
                absolute_error_sum[mode_index, column_index] += np.abs(errors)
                squared_error_sum[mode_index, column_index] += np.square(errors)
                if mode != "nominal":
                    lower, upper = bounds[mode]
                    if column in UPPER_BOUND_COLUMNS:
                        risk_bound = _clip(
                            prediction + upper[column][:horizon],
                            model.bounds[column])
                        coverage = actual_values <= risk_bound
                        width = risk_bound - prediction
                    else:
                        risk_bound = _clip(
                            prediction - lower[column][:horizon],
                            model.bounds[column])
                        coverage = actual_values >= risk_bound
                        width = prediction - risk_bound
                    bound_counts[mode_index, column_index] += 1
                    coverage_sum[mode_index, column_index] += coverage
                    interval_width_sum[mode_index, column_index] += width

    by_horizon_rows = []
    summary_rows = []
    for mode_index, mode in enumerate(modes):
        for column_index, column in enumerate(columns):
            for horizon_index in range(horizon):
                n = int(counts[mode_index, column_index, horizon_index])
                bound_n = int(bound_counts[
                    mode_index, column_index, horizon_index])
                by_horizon_rows.append({
                    "mode": mode,
                    "column": column,
                    "horizon": horizon_index + 1,
                    "n_forecasts": n,
                    "mae": absolute_error_sum[
                        mode_index, column_index, horizon_index] / n,
                    "rmse": np.sqrt(squared_error_sum[
                        mode_index, column_index, horizon_index] / n),
                    "bias": error_sum[
                        mode_index, column_index, horizon_index] / n,
                    "coverage": (
                        coverage_sum[mode_index, column_index, horizon_index]
                        / bound_n if bound_n else np.nan),
                    "mean_interval_width": (
                        interval_width_sum[
                            mode_index, column_index, horizon_index]
                        / bound_n if bound_n else np.nan),
                })
            n_total = int(counts[mode_index, column_index].sum())
            bound_total = int(bound_counts[mode_index, column_index].sum())
            summary_rows.append({
                "mode": mode,
                "column": column,
                "n_forecasts": n_total,
                "mae": absolute_error_sum[
                    mode_index, column_index].sum() / n_total,
                "rmse": np.sqrt(squared_error_sum[
                    mode_index, column_index].sum() / n_total),
                "bias": error_sum[
                    mode_index, column_index].sum() / n_total,
                "coverage": (
                    coverage_sum[mode_index, column_index].sum()
                    / bound_total if bound_total else np.nan),
                "mean_interval_width": (
                    interval_width_sum[mode_index, column_index].sum()
                    / bound_total if bound_total else np.nan),
            })
    by_horizon = pd.DataFrame(by_horizon_rows)
    summary = pd.DataFrame(summary_rows)
    by_horizon["adaptive_beta_floor"] = float(adaptive_beta_floor)
    summary["adaptive_beta_floor"] = float(adaptive_beta_floor)
    return summary, by_horizon


def _clip(values: np.ndarray, bounds: tuple[float | None, float | None]):
    low, high = bounds
    return np.clip(
        np.asarray(values, dtype=float),
        -np.inf if low is None else float(low),
        np.inf if high is None else float(high),
    )


def _file_record(path: str) -> dict:
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        return {"path": resolved, "exists": False}
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": resolved,
        "exists": True,
        "size_bytes": os.path.getsize(resolved),
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.adaptive_beta_floor <= 1.0:
        raise ValueError("--adaptive-beta-floor must be in [0, 1]")
    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    horizon_rows = []
    for country in args.countries:
        data_path = os.path.join(
            args.data_dir, f"hourly_inputs_{country}.csv")
        data = pd.read_csv(data_path)
        model, residuals, beta, history = load_artifacts(
            args.forecast_dir, country)
        summary, by_horizon = evaluate_sequence(
            data, model, residuals, beta, history,
            horizon=args.horizon, start_hour=args.start_hour,
            hours=args.hours,
            adaptive_beta_floor=args.adaptive_beta_floor)
        summary.insert(0, "country", country)
        by_horizon.insert(0, "country", country)
        summaries.append(summary)
        horizon_rows.append(by_horizon)
        print(f"{country}: evaluated {int(summary.n_forecasts.max())} forecasts")

    outputs = {
        "summary": pd.concat(summaries, ignore_index=True),
        "by_horizon": pd.concat(horizon_rows, ignore_index=True),
    }
    output_records = {}
    for name, output in outputs.items():
        path = os.path.join(args.out_dir, f"{name}_{args.tag}.csv")
        output.to_csv(path, index=False)
        output_records[name] = {
            "path": os.path.abspath(path),
            "rows": int(len(output)),
        }
        print(f"saved -> {path}")
    inputs = []
    for country in args.countries:
        inputs.extend([
            _file_record(os.path.join(
                args.data_dir, f"hourly_inputs_{country}.csv")),
            _file_record(os.path.join(
                args.forecast_dir, f"causal_ridge_{country}.npz")),
            _file_record(os.path.join(
                args.forecast_dir, f"calibration_residuals_{country}.npz")),
        ])
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "countries": list(args.countries),
            "horizon": args.horizon,
            "start_hour": args.start_hour,
            "hours": args.hours,
            "adaptive_beta_floor": args.adaptive_beta_floor,
            "data_dir": os.path.abspath(args.data_dir),
            "forecast_dir": os.path.abspath(args.forecast_dir),
            "tag": args.tag,
        },
        "inputs": inputs,
        "outputs": output_records,
    }
    manifest_path = os.path.join(
        args.out_dir, f"manifest_{args.tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {manifest_path}")


if __name__ == "__main__":
    main()
