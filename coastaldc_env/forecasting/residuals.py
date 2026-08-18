"""Online horizon-specific bias and empirical forecast-error bounds."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


def select_ewma_beta(
    residuals: np.ndarray,
    columns: Iterable[str],
    candidates: Iterable[float] = (0.0, 0.01, 0.05, 0.10, 0.20),
    warmup: int = 168,
) -> dict[str, float]:
    """Choose one EWMA rate per variable from causal residual sequences."""
    values = np.asarray(residuals, dtype=float)
    columns = tuple(columns)
    if values.ndim != 3 or values.shape[1] != len(columns):
        raise ValueError("Residuals must have shape (samples, columns, horizon)")
    if len(values) <= warmup:
        raise ValueError("Residual sequence is shorter than EWMA warmup")
    choices = tuple(float(candidate) for candidate in candidates)
    if not choices or any(not 0.0 <= candidate <= 1.0 for candidate in choices):
        raise ValueError("EWMA candidates must be in [0, 1]")

    selected: dict[str, float] = {}
    for column_index, column in enumerate(columns):
        column_residuals = values[:, column_index, :]
        horizon = column_residuals.shape[1]
        horizon_indices = np.arange(horizon, dtype=int)
        best_beta = None
        best_mae = np.inf
        for beta in choices:
            # Before forecast origin i, the residual made at origin r for
            # horizon h is observable only when r + h < i. Initialize from
            # the causally available warmup subset, then reveal one delayed
            # residual per horizon before scoring each new origin.
            initial_counts = np.maximum(0, warmup - horizon_indices - 1)
            bias = np.zeros(horizon, dtype=float)
            for horizon_index, count in enumerate(initial_counts):
                if count:
                    bias[horizon_index] = column_residuals[
                        :count, horizon_index].mean()
            absolute_error = 0.0
            count = 0
            for origin_index in range(warmup, len(column_residuals)):
                revealed_origins = origin_index - horizon_indices - 1
                revealed = revealed_origins >= 0
                bias[revealed] = (
                    (1.0 - beta) * bias[revealed]
                    + beta * column_residuals[
                        revealed_origins[revealed], horizon_indices[revealed]])
                residual = column_residuals[origin_index]
                absolute_error += float(np.abs(residual - bias).sum())
                count += residual.size
            mae = absolute_error / max(count, 1)
            if mae < best_mae:
                best_mae = mae
                best_beta = beta
        selected[column] = float(best_beta)
    return selected


@dataclass(frozen=True)
class _PendingForecast:
    column_index: int
    horizon_index: int
    base_value: float
    corrected_value: float


class OnlineResidualAdaptor:
    """Update forecasts only when their target observations become available."""

    def __init__(
        self,
        columns: Iterable[str],
        horizon: int,
        *,
        beta: float | Mapping[str, float] = 0.1,
        window: int = 720,
        confidence: float = 0.90,
        initial_residuals: np.ndarray | None = None,
        bias_correction: bool = True,
    ):
        self.columns = tuple(columns)
        self.horizon = int(horizon)
        self.window = int(window)
        self.confidence = float(confidence)
        self.bias_correction = bool(bias_correction)
        if not self.columns or self.horizon <= 0 or self.window <= 0:
            raise ValueError("Columns, horizon, and residual window must be positive")
        if not 0.5 < self.confidence < 1.0:
            raise ValueError("One-sided confidence must be in (0.5, 1.0)")

        if isinstance(beta, Mapping):
            beta_values = np.array([float(beta[column]) for column in self.columns])
        else:
            beta_values = np.full(len(self.columns), float(beta))
        if np.any((beta_values < 0.0) | (beta_values > 1.0)):
            raise ValueError("EWMA beta values must be in [0, 1]")
        self._beta = beta_values
        self._bias = np.zeros((len(self.columns), self.horizon), dtype=float)
        self._errors = [
            [deque(maxlen=self.window) for _ in range(self.horizon)]
            for _ in self.columns
        ]
        self._pending: dict[int, list[_PendingForecast]] = defaultdict(list)
        if initial_residuals is not None:
            self.initialize(initial_residuals)

    @property
    def bias(self) -> np.ndarray:
        return self._bias.copy()

    @property
    def beta(self) -> dict[str, float]:
        return {
            column: float(self._beta[index])
            for index, column in enumerate(self.columns)
        }

    @property
    def pending_targets(self) -> tuple[int, ...]:
        return tuple(sorted(self._pending))

    def sample_counts(self) -> np.ndarray:
        return np.array([
            [len(self._errors[c][h]) for h in range(self.horizon)]
            for c in range(len(self.columns))
        ], dtype=int)

    def initialize(self, residuals: np.ndarray) -> None:
        values = np.asarray(residuals, dtype=float)
        expected_tail = (len(self.columns), self.horizon)
        if values.ndim != 3 or values.shape[1:] != expected_tail:
            raise ValueError(
                "Initial residuals must have shape "
                f"(samples, {len(self.columns)}, {self.horizon})")
        if not np.isfinite(values).all():
            raise ValueError("Initial residuals must be finite")
        tail = values[-self.window:]
        self._bias = (
            tail.mean(axis=0) if self.bias_correction
            else np.zeros((len(self.columns), self.horizon), dtype=float)
        )
        centered = tail - self._bias[None, :, :]
        for column_index in range(len(self.columns)):
            for horizon_index in range(self.horizon):
                self._errors[column_index][horizon_index].clear()
                self._errors[column_index][horizon_index].extend(
                    centered[:, column_index, horizon_index].tolist())

    def correct(self, base_forecasts: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        base = self._validated_forecasts(base_forecasts)
        return {
            column: base[column] + self._bias[index]
            for index, column in enumerate(self.columns)
        }

    def register_forecast(
        self,
        origin: int,
        base_forecasts: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Register targets beginning at `origin` and return corrected forecasts."""
        origin = int(origin)
        base = self._validated_forecasts(base_forecasts)
        corrected = self.correct(base)
        for column_index, column in enumerate(self.columns):
            for horizon_index in range(self.horizon):
                target = origin + horizon_index
                self._pending[target].append(_PendingForecast(
                    column_index=column_index,
                    horizon_index=horizon_index,
                    base_value=float(base[column][horizon_index]),
                    corrected_value=float(corrected[column][horizon_index]),
                ))
        return corrected

    def observe(self, index: int, actual: Mapping[str, float]) -> int:
        """Reveal one target row and update only forecasts targeting that row."""
        entries = self._pending.pop(int(index), [])
        if not entries:
            return 0
        missing = [column for column in self.columns if column not in actual]
        if missing:
            raise ValueError(f"Actual observation missing columns: {missing}")
        for entry in entries:
            column = self.columns[entry.column_index]
            actual_value = float(actual[column])
            if not np.isfinite(actual_value):
                raise ValueError("Actual observations must be finite")
            base_error = actual_value - entry.base_value
            corrected_error = actual_value - entry.corrected_value
            beta = self._beta[entry.column_index]
            if self.bias_correction:
                self._bias[entry.column_index, entry.horizon_index] = (
                    (1.0 - beta)
                    * self._bias[entry.column_index, entry.horizon_index]
                    + beta * base_error)
            self._errors[entry.column_index][entry.horizon_index].append(
                corrected_error)
        return len(entries)

    def one_sided_bounds(
        self,
        confidence: float | None = None,
        horizon: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        level = self.confidence if confidence is None else float(confidence)
        if not 0.5 < level < 1.0:
            raise ValueError("One-sided confidence must be in (0.5, 1.0)")
        requested_horizon = self.horizon if horizon is None else int(horizon)
        if not 1 <= requested_horizon <= self.horizon:
            raise ValueError(
                f"Bound horizon must be in [1, {self.horizon}]")
        lower: dict[str, np.ndarray] = {}
        upper: dict[str, np.ndarray] = {}
        for column_index, column in enumerate(self.columns):
            lower_values = np.zeros(requested_horizon, dtype=float)
            upper_values = np.zeros(requested_horizon, dtype=float)
            for horizon_index in range(requested_horizon):
                errors = np.asarray(
                    self._errors[column_index][horizon_index], dtype=float)
                if errors.size:
                    upper_values[horizon_index] = max(
                        0.0, float(np.quantile(errors, level)))
                    lower_values[horizon_index] = max(
                        0.0, float(np.quantile(-errors, level)))
            lower[column] = lower_values
            upper[column] = upper_values
        return lower, upper

    def _validated_forecasts(
        self, forecasts: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        missing = [column for column in self.columns if column not in forecasts]
        if missing:
            raise ValueError(f"Forecasts missing columns: {missing}")
        validated: dict[str, np.ndarray] = {}
        for column in self.columns:
            values = np.asarray(forecasts[column], dtype=float)
            if values.shape != (self.horizon,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Forecast {column} must be finite with shape ({self.horizon},)")
            validated[column] = values.copy()
        return validated
