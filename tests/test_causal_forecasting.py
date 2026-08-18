import numpy as np
import pandas as pd

from coastaldc_env.forecasting import (
    DEFAULT_FORECAST_COLUMNS,
    CausalRidgeForecaster,
    OnlineResidualAdaptor,
    RidgeForecastConfig,
    expanding_window_residuals,
    select_ewma_beta,
)


def _forecast_frame(hours: int = 520) -> pd.DataFrame:
    t = np.arange(hours, dtype=float)
    return pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=hours, freq="h"),
        "fixed_load_mw": 5.0 + 0.5 * np.sin(2 * np.pi * t / 24),
        "flexible_arrival_mw": 1.5 + 0.2 * np.cos(2 * np.pi * t / 24),
        "sst_c": 15.0 + 2.0 * np.sin(2 * np.pi * t / (24 * 30)),
        "wind_mw": 6.0 + 3.0 * np.sin(2 * np.pi * t / 12),
        "carbon_kg_per_mwh": 400.0 + 50.0 * np.cos(2 * np.pi * t / 24),
    })


def test_causal_forecast_does_not_use_rows_at_or_after_origin():
    data = _forecast_frame()
    config = RidgeForecastConfig(horizon=12, alpha=0.1)
    model = CausalRidgeForecaster(config).fit(data.iloc[:360])
    origin = 420

    expected = model.predict(data, origin=origin)
    modified = data.copy()
    modified.loc[origin:, list(DEFAULT_FORECAST_COLUMNS)] = 1e6
    actual = model.predict(modified, origin=origin)

    for column in DEFAULT_FORECAST_COLUMNS:
        np.testing.assert_allclose(actual[column], expected[column])
        assert actual[column].shape == (12,)


def test_causal_forecast_save_load_round_trip(tmp_path):
    data = _forecast_frame()
    model = CausalRidgeForecaster(
        RidgeForecastConfig(horizon=6, alpha=1.0)).fit(data.iloc[:400])
    path = tmp_path / "forecast_model.npz"
    model.save(path)
    loaded = CausalRidgeForecaster.load(path)

    before = model.predict(data.iloc[:450])
    after = loaded.predict(data.iloc[:450])
    for column in DEFAULT_FORECAST_COLUMNS:
        np.testing.assert_allclose(before[column], after[column])


def test_minimal_lag_history_matches_full_history_prediction():
    data = _forecast_frame()
    model = CausalRidgeForecaster(
        RidgeForecastConfig(horizon=6, alpha=0.1)).fit(data.iloc[:400])
    origin = 480
    full = model.predict(data, origin=origin)
    max_lag = max(model.config.lags)
    recent = data.iloc[origin - max_lag:origin].reset_index(drop=True)
    minimal = model.predict(recent, origin=len(recent))

    for column in model.columns:
        np.testing.assert_allclose(minimal[column], full[column])


def test_expanding_residuals_are_out_of_sample_and_shaped():
    data = _forecast_frame()
    config = RidgeForecastConfig(horizon=6, alpha=1.0)
    origins, residuals = expanding_window_residuals(
        data, calibration_start=360, config=config, block_hours=48)

    assert origins[0] == 360
    assert origins[-1] == len(data) - config.horizon
    assert residuals.shape == (len(origins), len(DEFAULT_FORECAST_COLUMNS), 6)
    assert np.isfinite(residuals).all()


def test_residual_adaptor_updates_only_revealed_target():
    columns = ("wind_mw", "sst_c")
    adaptor = OnlineResidualAdaptor(columns, horizon=3, beta=0.2, window=10)
    base = {column: np.zeros(3) for column in columns}
    adaptor.register_forecast(origin=10, base_forecasts=base)

    assert adaptor.observe(9, {"wind_mw": 1.0, "sst_c": 2.0}) == 0
    np.testing.assert_allclose(adaptor.bias, 0.0)
    assert adaptor.observe(10, {"wind_mw": 1.0, "sst_c": 2.0}) == 2

    expected = np.zeros((2, 3))
    expected[:, 0] = [0.2, 0.4]
    np.testing.assert_allclose(adaptor.bias, expected)
    np.testing.assert_array_equal(adaptor.sample_counts()[:, 1:], 0)


def test_ewma_beta_selection_respects_horizon_revelation_delay():
    origins = np.arange(160, dtype=float)
    residuals = np.empty((len(origins), 1, 12), dtype=float)
    for horizon in range(12):
        residuals[:, 0, horizon] = np.sin(
            2.0 * np.pi * (origins + horizon) / 24.0)

    selected = select_ewma_beta(
        residuals, ("wind_mw",), candidates=(0.01, 0.20), warmup=48)

    assert selected["wind_mw"] == 0.01


def test_zero_beta_keeps_bias_fixed_but_updates_error_window():
    adaptor = OnlineResidualAdaptor(("wind_mw",), horizon=1, beta=0.0)
    adaptor.register_forecast(origin=10, base_forecasts={"wind_mw": np.zeros(1)})

    assert adaptor.observe(10, {"wind_mw": 3.0}) == 1

    np.testing.assert_allclose(adaptor.bias, 0.0)
    np.testing.assert_array_equal(adaptor.sample_counts(), [[1]])
    _, upper = adaptor.one_sided_bounds()
    np.testing.assert_allclose(upper["wind_mw"], [3.0])


def test_disabled_bias_correction_keeps_zero_bias_and_updates_quantiles():
    residuals = np.ones((8, 1, 2), dtype=float)
    adaptor = OnlineResidualAdaptor(
        ("wind_mw",), horizon=2, beta=0.5, initial_residuals=residuals,
        bias_correction=False)
    initial_counts = adaptor.sample_counts().copy()

    adaptor.register_forecast(10, {"wind_mw": np.zeros(2)})
    assert adaptor.observe(10, {"wind_mw": 2.0}) == 1

    np.testing.assert_allclose(adaptor.bias, 0.0)
    assert adaptor.sample_counts()[0, 0] == initial_counts[0, 0] + 1
    _, upper = adaptor.one_sided_bounds()
    assert upper["wind_mw"][0] > 0.0


def test_larger_calibration_errors_produce_wider_bounds():
    columns = ("wind_mw",)
    small = np.zeros((20, 1, 2))
    large = np.linspace(-5.0, 5.0, 20)[:, None, None] * np.ones((1, 1, 2))
    small_adaptor = OnlineResidualAdaptor(
        columns, horizon=2, initial_residuals=small)
    large_adaptor = OnlineResidualAdaptor(
        columns, horizon=2, initial_residuals=large)

    small_lower, small_upper = small_adaptor.one_sided_bounds()
    large_lower, large_upper = large_adaptor.one_sided_bounds()

    assert np.all(large_lower["wind_mw"] > small_lower["wind_mw"])
    assert np.all(large_upper["wind_mw"] > small_upper["wind_mw"])


def test_residual_bounds_can_be_limited_to_requested_horizon():
    residuals = np.linspace(-2.0, 2.0, 20)[:, None, None]
    residuals = np.broadcast_to(residuals, (20, 1, 4)).copy()
    adaptor = OnlineResidualAdaptor(
        ("wind_mw",), horizon=4, initial_residuals=residuals)

    full_lower, full_upper = adaptor.one_sided_bounds()
    short_lower, short_upper = adaptor.one_sided_bounds(horizon=2)

    np.testing.assert_allclose(short_lower["wind_mw"], full_lower["wind_mw"][:2])
    np.testing.assert_allclose(short_upper["wind_mw"], full_upper["wind_mw"][:2])


def test_forecast_physical_bounds_are_applied():
    data = _forecast_frame()
    bounds = {column: (0.0, 1.0) for column in DEFAULT_FORECAST_COLUMNS}
    model = CausalRidgeForecaster(
        RidgeForecastConfig(horizon=4), bounds=bounds).fit(data.iloc[:400])
    forecast = model.predict(data.iloc[:450])

    for values in forecast.values():
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)
