import numpy as np
import pandas as pd

from coastaldc_env.forecasting import CausalRidgeForecaster, RidgeForecastConfig
from scripts.evaluate_weekly_forecast_errors import weekly_error_scores


def test_weekly_error_scores_return_complete_finite_weeks():
    hours = 700
    t = np.arange(hours, dtype=float)
    data = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=hours, freq="h"),
        "fixed_load_mw": 6.0 + np.sin(2 * np.pi * t / 24),
        "flexible_arrival_mw": 1.0 + 0.1 * np.cos(2 * np.pi * t / 24),
        "sst_c": 15.0 + np.sin(2 * np.pi * t / 168),
        "wind_mw": 5.0 + np.cos(2 * np.pi * t / 12),
        "carbon_kg_per_mwh": 400.0 + 10.0 * np.sin(2 * np.pi * t / 24),
    })
    model = CausalRidgeForecaster(
        RidgeForecastConfig(horizon=4, alpha=0.1)).fit(data.iloc[:350])
    history = data.iloc[182:350][["timestamp", *model.columns]].reset_index(drop=True)
    residuals = np.ones((20, len(model.columns), 4), dtype=float)

    result = weekly_error_scores(
        data.iloc[350:].reset_index(drop=True), model, residuals,
        {column: 0.1 for column in model.columns}, history,
        horizon=4, target_hours=336)

    assert result.week.tolist() == [0, 1]
    assert np.isfinite(result.normalized_mae).all()
    assert (result.n_errors > 0).all()
