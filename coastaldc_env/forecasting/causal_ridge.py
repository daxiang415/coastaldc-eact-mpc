"""Direct multi-horizon Ridge forecasts using only observations before origin."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

DEFAULT_FORECAST_COLUMNS = (
    "fixed_load_mw",
    "flexible_arrival_mw",
    "sst_c",
    "wind_mw",
    "carbon_kg_per_mwh",
)
DEFAULT_LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
DEFAULT_BOUNDS = {
    "fixed_load_mw": (0.0, None),
    "flexible_arrival_mw": (0.0, None),
    "sst_c": (-2.0, 40.0),
    "wind_mw": (0.0, None),
    "carbon_kg_per_mwh": (0.0, None),
}


@dataclass(frozen=True)
class RidgeForecastConfig:
    horizon: int = 48
    lags: tuple[int, ...] = DEFAULT_LAGS
    alpha: float = 1.0

    def __post_init__(self):
        if self.horizon <= 0:
            raise ValueError("Forecast horizon must be positive")
        if not self.lags or any(lag <= 0 for lag in self.lags):
            raise ValueError("Forecast lags must be positive")
        if tuple(sorted(set(self.lags))) != self.lags:
            raise ValueError("Forecast lags must be sorted and unique")
        if self.alpha <= 0.0:
            raise ValueError("Ridge alpha must be positive")


@dataclass
class _RidgeModel:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    coefficients: np.ndarray


class CausalRidgeForecaster:
    """Fit one direct multi-output Ridge model per forecast variable.

    `origin` always denotes the first target row. Features for that forecast
    use rows strictly before `origin`; lag 1 is the latest observed row.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        config: RidgeForecastConfig | None = None,
        columns: Iterable[str] = DEFAULT_FORECAST_COLUMNS,
        bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    ):
        self.config = config or RidgeForecastConfig()
        self.columns = tuple(columns)
        if not self.columns:
            raise ValueError("At least one forecast column is required")
        self.bounds = dict(DEFAULT_BOUNDS)
        if bounds:
            self.bounds.update(bounds)
        self._models: dict[str, _RidgeModel] = {}

    @property
    def fitted(self) -> bool:
        return set(self._models) == set(self.columns)

    def fit(
        self,
        data: pd.DataFrame,
        origins: Iterable[int] | None = None,
    ) -> "CausalRidgeForecaster":
        frame = _validated_frame(data, self.columns)
        max_lag = max(self.config.lags)
        last_origin = len(frame) - self.config.horizon
        if origins is None:
            selected = np.arange(max_lag, last_origin + 1, dtype=int)
        else:
            selected = np.asarray(list(origins), dtype=int)
        if selected.size == 0:
            raise ValueError("No valid forecast origins are available for fitting")
        if selected.min() < max_lag or selected.max() > last_origin:
            raise ValueError(
                f"Training origins must be in [{max_lag}, {last_origin}]")

        time_features = np.vstack([
            _time_features_at_origin(frame, int(origin)) for origin in selected
        ])
        ridge_eye = np.eye(len(self.config.lags) + time_features.shape[1])
        models: dict[str, _RidgeModel] = {}

        for column in self.columns:
            values = frame[column].to_numpy(dtype=float)
            lag_features = np.vstack([
                [values[int(origin) - lag] for lag in self.config.lags]
                for origin in selected
            ])
            x = np.hstack([lag_features, time_features])
            y = np.vstack([
                values[int(origin): int(origin) + self.config.horizon]
                for origin in selected
            ])

            x_mean = x.mean(axis=0)
            x_scale = x.std(axis=0)
            x_scale[x_scale < 1e-12] = 1.0
            xs = (x - x_mean) / x_scale
            y_mean = y.mean(axis=0)
            yc = y - y_mean
            lhs = xs.T @ xs + self.config.alpha * ridge_eye
            coefficients = np.linalg.solve(lhs, xs.T @ yc)
            models[column] = _RidgeModel(
                x_mean=x_mean,
                x_scale=x_scale,
                y_mean=y_mean,
                coefficients=coefficients,
            )

        self._models = models
        return self

    def predict(
        self,
        history: pd.DataFrame,
        origin: int | None = None,
        horizon: int | None = None,
    ) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Forecaster must be fitted before prediction")
        frame = _validated_frame(history, self.columns)
        if origin is None:
            origin = len(frame)
        origin = int(origin)
        if origin > len(frame):
            raise ValueError("Prediction origin cannot exceed available history")
        if origin < max(self.config.lags):
            raise ValueError("Insufficient causal history for the configured lags")
        requested_horizon = self.config.horizon if horizon is None else int(horizon)
        if not 1 <= requested_horizon <= self.config.horizon:
            raise ValueError(
                f"Prediction horizon must be in [1, {self.config.horizon}]")

        time_features = _time_features_at_origin(frame, origin)
        forecasts: dict[str, np.ndarray] = {}
        for column in self.columns:
            values = frame[column].to_numpy(dtype=float)
            lag_features = np.array(
                [values[origin - lag] for lag in self.config.lags], dtype=float)
            x = np.concatenate([lag_features, time_features])
            model = self._models[column]
            xs = (x - model.x_mean) / model.x_scale
            prediction = model.y_mean + xs @ model.coefficients
            forecasts[column] = self._clip(column, prediction[:requested_horizon])
        return forecasts

    def save(self, path: str | os.PathLike[str]) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted forecaster")
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        metadata = {
            "format_version": self.FORMAT_VERSION,
            "horizon": self.config.horizon,
            "lags": list(self.config.lags),
            "alpha": self.config.alpha,
            "columns": list(self.columns),
            "bounds": {
                key: list(value) for key, value in self.bounds.items()
            },
        }
        arrays: dict[str, np.ndarray] = {
            "metadata": np.asarray(json.dumps(metadata)),
        }
        for index, column in enumerate(self.columns):
            model = self._models[column]
            prefix = f"model_{index}"
            arrays[f"{prefix}_x_mean"] = model.x_mean
            arrays[f"{prefix}_x_scale"] = model.x_scale
            arrays[f"{prefix}_y_mean"] = model.y_mean
            arrays[f"{prefix}_coefficients"] = model.coefficients
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CausalRidgeForecaster":
        with np.load(os.fspath(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != cls.FORMAT_VERSION:
                raise ValueError("Unsupported causal forecast model format")
            config = RidgeForecastConfig(
                horizon=int(metadata["horizon"]),
                lags=tuple(int(value) for value in metadata["lags"]),
                alpha=float(metadata["alpha"]),
            )
            bounds = {
                key: tuple(value) for key, value in metadata["bounds"].items()
            }
            forecaster = cls(
                config=config, columns=metadata["columns"], bounds=bounds)
            for index, column in enumerate(forecaster.columns):
                prefix = f"model_{index}"
                forecaster._models[column] = _RidgeModel(
                    x_mean=archive[f"{prefix}_x_mean"].copy(),
                    x_scale=archive[f"{prefix}_x_scale"].copy(),
                    y_mean=archive[f"{prefix}_y_mean"].copy(),
                    coefficients=archive[f"{prefix}_coefficients"].copy(),
                )
        return forecaster

    def _clip(self, column: str, values: np.ndarray) -> np.ndarray:
        low, high = self.bounds.get(column, (None, None))
        lower = -np.inf if low is None else float(low)
        upper = np.inf if high is None else float(high)
        return np.clip(np.asarray(values, dtype=float), lower, upper)


def select_ridge_alpha(
    data: pd.DataFrame,
    candidates: Iterable[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    *,
    columns: Iterable[str] = DEFAULT_FORECAST_COLUMNS,
    horizon: int = 48,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    validation_hours: int = 24 * 90,
    sample_step: int = 6,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> float:
    """Select Ridge strength on a final blocked causal validation period."""
    frame = _validated_frame(data, tuple(columns))
    min_train = max(lags) + horizon + 1
    validation_hours = min(validation_hours, len(frame) - min_train)
    if validation_hours < horizon:
        raise ValueError("Not enough data for blocked Ridge validation")
    train_end = len(frame) - validation_hours
    origins = range(train_end, len(frame) - horizon + 1, sample_step)
    best_alpha = None
    best_mae = np.inf
    for alpha in candidates:
        model = CausalRidgeForecaster(
            RidgeForecastConfig(horizon=horizon, lags=lags, alpha=float(alpha)),
            columns=columns,
            bounds=bounds,
        ).fit(frame.iloc[:train_end])
        absolute_error = 0.0
        count = 0
        for origin in origins:
            predictions = model.predict(frame.iloc[:origin], horizon=horizon)
            for column in model.columns:
                actual = frame[column].to_numpy(dtype=float)[origin:origin + horizon]
                absolute_error += float(np.abs(actual - predictions[column]).sum())
                count += horizon
        mae = absolute_error / max(count, 1)
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)
    if best_alpha is None:
        raise RuntimeError("Ridge alpha selection produced no candidate")
    return best_alpha


def expanding_window_residuals(
    data: pd.DataFrame,
    calibration_start: int,
    *,
    config: RidgeForecastConfig,
    columns: Iterable[str] = DEFAULT_FORECAST_COLUMNS,
    block_hours: int = 24 * 30,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-sample residuals with expanding-window refits.

    Returns `(origins, residuals)`, where residuals have shape
    `(n_origins, n_columns, horizon)`.
    """
    columns = tuple(columns)
    frame = _validated_frame(data, columns)
    if calibration_start < max(config.lags) + config.horizon:
        raise ValueError("Calibration start leaves insufficient training data")
    last_origin = len(frame) - config.horizon
    if calibration_start > last_origin:
        raise ValueError("Calibration period has no complete forecast horizon")
    if block_hours <= 0:
        raise ValueError("Calibration block size must be positive")

    origin_chunks: list[np.ndarray] = []
    residual_chunks: list[np.ndarray] = []
    for block_start in range(calibration_start, last_origin + 1, block_hours):
        block_end = min(block_start + block_hours, last_origin + 1)
        model = CausalRidgeForecaster(
            config=config, columns=columns, bounds=bounds).fit(
                frame.iloc[:block_start])
        block_origins = np.arange(block_start, block_end, dtype=int)
        block_residuals = np.empty(
            (len(block_origins), len(columns), config.horizon), dtype=np.float32)
        for row_index, origin in enumerate(block_origins):
            predictions = model.predict(frame.iloc[:origin])
            for column_index, column in enumerate(columns):
                actual = frame[column].to_numpy(dtype=float)[
                    origin:origin + config.horizon]
                block_residuals[row_index, column_index] = (
                    actual - predictions[column])
        origin_chunks.append(block_origins)
        residual_chunks.append(block_residuals)
    return np.concatenate(origin_chunks), np.concatenate(residual_chunks, axis=0)


def _validated_frame(data: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Forecast data must be a pandas DataFrame")
    columns = tuple(columns)
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Forecast data missing columns: {missing}")
    frame = data.reset_index(drop=True)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Forecast column contains non-finite values: {column}")
    if "timestamp" in frame.columns:
        timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
        if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
            raise ValueError("Forecast timestamps must be unique and increasing")
        if len(timestamps) > 1:
            differences = timestamps.diff().dropna()
            if not (differences == pd.Timedelta(hours=1)).all():
                raise ValueError("Forecast data must have an hourly time index")
    return frame


def _time_features_at_origin(frame: pd.DataFrame, origin: int) -> np.ndarray:
    if origin <= 0 or origin > len(frame):
        raise ValueError("Forecast origin must follow an observed row")
    if "timestamp" in frame.columns:
        timestamp = pd.Timestamp(frame.iloc[origin - 1]["timestamp"]) + pd.Timedelta(hours=1)
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        days_in_year = 366 if timestamp.is_leap_year else 365
        day_of_year = timestamp.dayofyear - 1
    else:
        hour = origin % 24
        day_of_week = (origin // 24) % 7
        days_in_year = 365
        day_of_year = (origin // 24) % days_in_year
    return np.array([
        np.sin(2.0 * np.pi * hour / 24.0),
        np.cos(2.0 * np.pi * hour / 24.0),
        np.sin(2.0 * np.pi * day_of_week / 7.0),
        np.cos(2.0 * np.pi * day_of_week / 7.0),
        np.sin(2.0 * np.pi * day_of_year / days_in_year),
        np.cos(2.0 * np.pi * day_of_year / days_in_year),
    ], dtype=float)
