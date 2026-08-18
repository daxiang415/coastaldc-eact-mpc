import numpy as np
import pandas as pd
import pytest

from scripts.run_eact_thermal_capacity_stress import (
    build_jobs,
    build_parser,
    select_quarterly_high_load_windows,
)


def _write_country(path, peaks=(240, 2400, 4560, 6720)):
    hours = 8760
    flexible = np.full(hours, 2.0)
    for start in peaks:
        flexible[start:start + 168] = 3.0
    pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=hours, freq="h"),
        "fixed_load_mw": np.full(hours, 4.0),
        "flexible_arrival_mw": flexible,
    }).to_csv(path, index=False)


def test_high_load_windows_use_fixed_quarter_rule(tmp_path):
    path = tmp_path / "hourly_inputs_JPN.csv"
    peaks = (240, 2400, 4560, 6720)
    _write_country(path, peaks=peaks)

    windows = select_quarterly_high_load_windows(
        str(path), episode_hours=168)

    assert [item["quarter"] for item in windows] == ["Q1", "Q2", "Q3", "Q4"]
    assert [item["start_hour"] for item in windows] == list(peaks)
    assert all(item["mean_it_arrival_mw"] == pytest.approx(7.0)
               for item in windows)


def test_default_matrix_is_fixed_and_information_matched(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for country in ("JPN", "CHN", "NOR"):
        _write_country(data_dir / f"hourly_inputs_{country}.csv")
    args = build_parser().parse_args([
        "--data-dir", str(data_dir),
        "--out-root", str(tmp_path / "out"),
    ])

    jobs = build_jobs(args)

    assert len(jobs) == 18
    assert {job.stage for job in jobs} == {"thermal_capacity_stress"}
    assert all("--no-thermal-safety-shield" in job.command for job in jobs)
    assert all("--no-oracle-workload-projection" in job.command for job in jobs)
    assert all(
        job.command[job.command.index("--controllers") + 1:
                    job.command.index("--data-dir")]
        == ("nominal", "static", "eact")
        for job in jobs
    )
    levels = {
        job.command[
            job.command.index("--cooling-conductance-multiplier") + 1]
        for job in jobs
    }
    assert levels == {"1.0", "0.75", "0.5"}
    assert sum(
        len(job.command[job.command.index("--start-timestamps") + 1:
                        job.command.index("--horizon")])
        for job in jobs
    ) == 18 * 4


@pytest.mark.parametrize("value", ["0", "-0.5", "1.1", "nan"])
def test_availability_levels_must_be_in_unit_interval(tmp_path, value):
    args = build_parser().parse_args([
        "--data-dir", str(tmp_path),
        "--availability-levels", value,
    ])

    with pytest.raises(ValueError, match="Availability levels"):
        build_jobs(args)
