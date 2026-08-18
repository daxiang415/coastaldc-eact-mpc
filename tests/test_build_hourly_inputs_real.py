"""Timestamp alignment and multi-year CLI for real hourly inputs."""

import numpy as np
import pandas as pd
import pytest

from scripts import build_hourly_inputs_real as build
from scripts import download_sst_multiyear
from scripts import download_wind_openmeteo_era5


def test_expected_hourly_index_handles_leap_years():
    assert len(build.expected_hourly_index(2025)) == 8760
    assert len(build.expected_hourly_index(2024)) == 8784


def test_validate_hourly_series_accepts_complete_utc_year():
    index = build.expected_hourly_index(2025)
    values = np.arange(len(index), dtype=float)

    result = build.validate_hourly_series(
        pd.Series(values, index=index), 2025, "test")

    assert result.index.equals(index)
    np.testing.assert_allclose(result.values, values)


def test_validate_hourly_series_rejects_missing_timestamp():
    index = build.expected_hourly_index(2025).delete(100)

    with pytest.raises(ValueError, match="missing 1 hourly timestamps"):
        build.validate_hourly_series(
            pd.Series(np.ones(len(index)), index=index), 2025, "test")


def test_validate_hourly_series_rejects_duplicate_timestamp():
    index = build.expected_hourly_index(2025)
    duplicate = index.insert(10, index[10])

    with pytest.raises(ValueError, match="duplicate timestamps"):
        build.validate_hourly_series(
            pd.Series(np.ones(len(duplicate)), index=duplicate), 2025, "test")


def test_real_input_parser_accepts_multiple_years():
    args = build.build_parser().parse_args(
        ["--repo", "source", "--years", "2023", "2024", "2025"])

    assert args.years == [2023, 2024, 2025]


def test_sst_downloader_supports_leap_year():
    timestamps = download_sst_multiyear.expected_timestamps(2024)

    assert len(timestamps) == 8784
    assert timestamps[0] == "2024-01-01 00:00"
    assert timestamps[-1] == "2024-12-31 23:00"


def test_openmeteo_era5_csv_is_supported(tmp_path):
    timestamps = build.expected_hourly_index(2025)
    frame = pd.DataFrame({
        "timestamp": timestamps,
        "wind_speed_10m": 6.0,
        "wind_speed_100m": 8.0,
    })
    frame.to_csv(tmp_path / "OW_021_Tokyo_era5_wind_2025.csv", index=False)

    result = build.load_wind_mw(
        str(tmp_path / "missing_netcdf"), "OW_021_", 15.0, 2025,
        str(tmp_path), "openmeteo")

    assert len(result) == 8760
    assert (result > 0.0).all()


def test_wind_source_can_be_selected_explicitly():
    args = build.build_parser().parse_args(
        ["--repo", "source", "--wind-source", "openmeteo"])

    assert args.wind_source == "openmeteo"


def test_wind_downloader_uses_nearest_grid_by_default():
    args = download_wind_openmeteo_era5.build_parser().parse_args(
        ["--input", "manifest.xlsx", "--year", "2025", "--output-dir", "out"])

    assert args.cell_selection == "nearest"
