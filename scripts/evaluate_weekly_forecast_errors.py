"""Compute causal weekly forecast-error scores for mechanism analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env.forecasting import OnlineResidualAdaptor  # noqa: E402
from scripts.evaluate_causal_forecasts import load_artifacts  # noqa: E402


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def weekly_error_scores(
    data: pd.DataFrame,
    model,
    calibration_residuals: np.ndarray,
    beta: dict[str, float],
    history_tail: pd.DataFrame,
    *,
    horizon: int = 24,
    target_hours: int = 8736,
) -> pd.DataFrame:
    """Aggregate adaptive point errors by target week and calibration MAE."""
    if not 1 <= horizon <= model.config.horizon:
        raise ValueError("Evaluation horizon exceeds the fitted forecast horizon")
    frame = data.copy().reset_index(drop=True)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    target_hours = min(int(target_hours), len(frame))
    if target_hours < 168:
        raise ValueError("Weekly error analysis requires at least 168 target hours")

    columns = tuple(model.columns)
    adaptor = OnlineResidualAdaptor(
        columns,
        model.config.horizon,
        beta=beta,
        window=720,
        confidence=0.90,
        initial_residuals=calibration_residuals,
    )
    calibration_mae = np.maximum(
        np.mean(np.abs(calibration_residuals), axis=0), 1e-6)
    n_weeks = target_hours // 168
    score_sums = np.zeros(n_weeks, dtype=float)
    raw_error_sums = np.zeros(n_weeks, dtype=float)
    counts = np.zeros(n_weeks, dtype=np.int64)
    max_lag = max(model.config.lags)
    observed_columns = (
        ["timestamp", *columns]
        if "timestamp" in frame.columns else list(columns))

    for data_index in range(target_hours - 1):
        if data_index > 0:
            adaptor.observe(data_index, {
                column: float(frame.iloc[data_index][column])
                for column in columns
            })
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
        corrected = adaptor.register_forecast(data_index + 1, base)
        available = min(horizon, target_hours - data_index - 1)
        actual = frame.iloc[data_index + 1:data_index + 1 + available]
        for horizon_index in range(available):
            target_index = data_index + 1 + horizon_index
            week = target_index // 168
            if week >= n_weeks:
                continue
            actual_values = actual.iloc[horizon_index][list(columns)].to_numpy(
                dtype=float)
            forecast_values = np.array([
                corrected[column][horizon_index] for column in columns],
                dtype=float,
            )
            absolute_error = np.abs(actual_values - forecast_values)
            scale = calibration_mae[:, horizon_index]
            score_sums[week] += float(np.sum(absolute_error / scale))
            raw_error_sums[week] += float(np.sum(absolute_error))
            counts[week] += len(columns)

    rows = []
    for week in range(n_weeks):
        if counts[week] == 0:
            raise ValueError(f"No forecast errors accumulated for week {week}")
        rows.append({
            "week": week,
            "start_hour": week * 168,
            "n_errors": int(counts[week]),
            "normalized_mae": float(score_sums[week] / counts[week]),
            "raw_mae": float(raw_error_sums[week] / counts[week]),
        })
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--countries", nargs="*", default=["JPN", "CHN", "NOR"])
    parser.add_argument("--data-dir", default=os.path.join(
        ROOT, "data", "processed_multiyear", "2025"))
    parser.add_argument("--forecast-dir", default=os.path.join(
        ROOT, "results", "causal_forecasts_v3_gated_bias"))
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--target-hours", type=int, default=8736)
    parser.add_argument("--out-dir", default=os.path.join(
        ROOT, "results", "eact_paper_v1", "forecast_error"))
    parser.add_argument("--tag", default="annual_2025")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    frames = []
    for country in args.countries:
        data = pd.read_csv(os.path.join(
            args.data_dir, f"hourly_inputs_{country}.csv"))
        model, residuals, beta, history = load_artifacts(
            args.forecast_dir, country)
        result = weekly_error_scores(
            data, model, residuals, beta, history,
            horizon=args.horizon, target_hours=args.target_hours)
        result.insert(0, "country", country)
        frames.append(result)
        print(f"{country}: {len(result)} complete weeks")
    output = pd.concat(frames, ignore_index=True)
    output_path = os.path.join(
        args.out_dir, f"weekly_forecast_error_{args.tag}.csv")
    output.to_csv(output_path, index=False)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "countries": args.countries,
        "horizon": args.horizon,
        "target_hours": args.target_hours,
        "output": os.path.abspath(output_path),
        "rows": len(output),
    }
    manifest_path = os.path.join(
        args.out_dir, f"manifest_forecast_error_{args.tag}.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"saved -> {output_path}")
    print(f"saved -> {manifest_path}")


if __name__ == "__main__":
    main()
