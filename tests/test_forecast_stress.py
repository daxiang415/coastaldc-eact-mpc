import numpy as np

from coastaldc_env.forecasting import (
    ForecastStressConfig,
    ResidualScaledForecastStress,
)


COLUMNS = (
    "fixed_load_mw",
    "flexible_arrival_mw",
    "sst_c",
    "wind_mw",
    "carbon_kg_per_mwh",
)


def _stressor(mode="combined", scale=1.0, start_step=0, seed=7):
    residuals = np.linspace(-2.0, 2.0, 40)[:, None, None]
    residuals = np.broadcast_to(residuals, (40, len(COLUMNS), 4)).copy()
    return ResidualScaledForecastStress(
        COLUMNS,
        residuals,
        bounds={column: (0.0, None) for column in COLUMNS},
        config=ForecastStressConfig(
            mode=mode, scale=scale, start_step=start_step, seed=seed),
    )


def _forecasts():
    return {column: np.full(4, 10.0) for column in COLUMNS}


def test_forecast_stress_is_deterministic_for_shared_origin():
    first, first_diagnostics = _stressor().apply(
        _forecasts(), origin=123, episode_step=10)
    second, second_diagnostics = _stressor().apply(
        _forecasts(), origin=123, episode_step=10)

    for column in COLUMNS:
        np.testing.assert_allclose(first[column], second[column])
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["forecast_stress_active"] is True


def test_adverse_bias_uses_constraint_worsening_directions():
    stressed, _ = _stressor(mode="adverse_bias").apply(
        _forecasts(), origin=1, episode_step=0)

    assert np.all(stressed["fixed_load_mw"] < 10.0)
    assert np.all(stressed["flexible_arrival_mw"] < 10.0)
    assert np.all(stressed["sst_c"] < 10.0)
    assert np.all(stressed["carbon_kg_per_mwh"] < 10.0)
    assert np.all(stressed["wind_mw"] > 10.0)


def test_forecast_stress_starts_only_at_configured_episode_step():
    stressor = _stressor(mode="noise", start_step=24)
    before, diagnostics = stressor.apply(
        _forecasts(), origin=1, episode_step=23)
    after, after_diagnostics = stressor.apply(
        _forecasts(), origin=2, episode_step=24)

    for column in COLUMNS:
        np.testing.assert_allclose(before[column], 10.0)
    assert diagnostics["forecast_stress_active"] is False
    assert after_diagnostics["forecast_stress_active"] is True
    assert any(not np.allclose(after[column], 10.0) for column in COLUMNS)
