"""Deterministic residual-scaled forecast stress for closed-loop MPC tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


FORECAST_STRESS_MODES = ("none", "adverse_bias", "noise", "combined")

_ADVERSE_DIRECTIONS = {
    "fixed_load_mw": -1.0,
    "flexible_arrival_mw": -1.0,
    "sst_c": -1.0,
    "wind_mw": 1.0,
    "carbon_kg_per_mwh": -1.0,
}


@dataclass(frozen=True)
class ForecastStressConfig:
    mode: str = "none"
    scale: float = 0.0
    start_step: int = 0
    seed: int = 20260717

    def __post_init__(self) -> None:
        if self.mode not in FORECAST_STRESS_MODES:
            raise ValueError(f"Unknown forecast stress mode: {self.mode}")
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("Forecast stress scale must be finite and nonnegative")
        if self.start_step < 0 or self.seed < 0:
            raise ValueError("Forecast stress start step and seed must be nonnegative")


class ResidualScaledForecastStress:
    """Apply identical deterministic perturbations to any controller forecast."""

    def __init__(
        self,
        columns: Iterable[str],
        calibration_residuals: np.ndarray,
        *,
        bounds: Mapping[str, tuple[float | None, float | None]],
        config: ForecastStressConfig,
        window: int = 720,
    ):
        self.columns = tuple(columns)
        self.config = config
        self.bounds = dict(bounds)
        residuals = np.asarray(calibration_residuals, dtype=float)
        if residuals.ndim != 3 or residuals.shape[1] != len(self.columns):
            raise ValueError(
                "Calibration residuals must have shape (samples, columns, horizon)")
        if not np.isfinite(residuals).all() or len(residuals) == 0:
            raise ValueError("Calibration residuals must be nonempty and finite")
        tail = residuals[-min(int(window), len(residuals)):]
        ddof = 1 if len(tail) > 1 else 0
        self.residual_std = np.std(tail, axis=0, ddof=ddof)

    def apply(
        self,
        forecasts: Mapping[str, np.ndarray],
        *,
        origin: int,
        episode_step: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, float | bool | str]]:
        values = {
            column: np.asarray(forecasts[column], dtype=float).copy()
            for column in self.columns
        }
        active = (
            self.config.mode != "none"
            and self.config.scale > 0.0
            and episode_step >= self.config.start_step
        )
        absolute_changes = []
        if active:
            for column_index, column in enumerate(self.columns):
                base = values[column]
                sigma = self.residual_std[column_index, :len(base)]
                change = np.zeros_like(base)
                if self.config.mode in ("adverse_bias", "combined"):
                    direction = _ADVERSE_DIRECTIONS.get(column)
                    if direction is None:
                        raise ValueError(
                            f"No adverse forecast direction configured for {column}")
                    change += direction * self.config.scale * sigma
                if self.config.mode in ("noise", "combined"):
                    sequence = np.random.SeedSequence([
                        self.config.seed, int(origin), column_index])
                    noise = np.random.default_rng(sequence).standard_normal(len(base))
                    change += self.config.scale * sigma * noise
                stressed = base + change
                low, high = self.bounds.get(column, (None, None))
                values[column] = np.clip(
                    stressed,
                    -np.inf if low is None else low,
                    np.inf if high is None else high,
                )
                absolute_changes.extend(np.abs(values[column] - base).tolist())
        mean_change = float(np.mean(absolute_changes)) if absolute_changes else 0.0
        max_change = float(np.max(absolute_changes)) if absolute_changes else 0.0
        return values, {
            "forecast_stress_mode": self.config.mode,
            "forecast_stress_active": bool(active),
            "forecast_stress_mean_abs": mean_change,
            "forecast_stress_max_abs": max_change,
        }
