import numpy as np
import pandas as pd

from coastaldc_env.forecasting import CausalRidgeForecaster, RidgeForecastConfig
from scripts.evaluate_causal_forecasts import evaluate_sequence


def _frame(hours: int = 560) -> pd.DataFrame:
    t = np.arange(hours, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=hours, freq="h"),
        "fixed_load_mw": 5.0 + 0.4 * np.sin(2 * np.pi * t / 24),
        "flexible_arrival_mw": 1.5 + 0.2 * np.cos(2 * np.pi * t / 24),
        "sst_c": 16.0 + np.sin(2 * np.pi * t / (24 * 30)),
        "wind_mw": 6.0 + 2.0 * np.sin(2 * np.pi * t / 12),
        "carbon_kg_per_mwh": 400.0 + 30.0 * np.cos(2 * np.pi * t / 24),
    })


def test_forecast_evaluation_reports_modes_horizons_and_coverage():
    data = _frame()
    config = RidgeForecastConfig(horizon=6, alpha=0.1)
    model = CausalRidgeForecaster(config).fit(data.iloc[:420])
    test = data.iloc[420:].reset_index(drop=True)
    history = data.iloc[252:420].reset_index(drop=True)
    residuals = np.linspace(-0.5, 0.5, 30)[:, None, None]
    residuals = np.broadcast_to(
        residuals, (30, len(model.columns), config.horizon)).copy()

    summary, by_horizon = evaluate_sequence(
        test, model, residuals,
        {column: 0.01 for column in model.columns}, history,
        horizon=4, hours=12, adaptive_beta_floor=0.10)

    assert set(summary["mode"]) == {"nominal", "static", "adaptive"}
    assert set(by_horizon["horizon"]) == {1, 2, 3, 4}
    assert set(summary["column"]) == set(model.columns)
    assert (summary["n_forecasts"] == 12 * 4).all()
    bounded = summary[summary["mode"] != "nominal"]
    assert bounded["coverage"].between(0.0, 1.0).all()
    assert (bounded["mean_interval_width"] >= 0.0).all()
    assert summary[summary["mode"] == "nominal"]["coverage"].isna().all()
    assert np.allclose(summary["adaptive_beta_floor"], 0.10)
    assert np.allclose(by_horizon["adaptive_beta_floor"], 0.10)
