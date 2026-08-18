"""Audit downloaded SST and ERA5 wind files before building model inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def expected_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", inclusive="left", freq="h")


def audit_csv(path: Path, year: int) -> list[str]:
    errors: list[str] = []
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        return [f"{path}: missing timestamp column"]
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    expected = expected_index(year)
    if len(frame) != len(expected):
        errors.append(f"{path}: rows={len(frame)}, expected={len(expected)}")
    if timestamps.duplicated().any():
        errors.append(f"{path}: duplicate timestamps")
    if not timestamps.equals(expected):
        errors.append(f"{path}: timestamp axis mismatch")
    missing = int(frame.isna().sum().sum())
    if missing:
        errors.append(f"{path}: {missing} missing values")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst-dir", required=True, type=Path)
    parser.add_argument("--wind-dir", required=True, type=Path)
    parser.add_argument("--processed-root", type=Path, default=None)
    parser.add_argument("--years", nargs="+", required=True, type=int)
    parser.add_argument("--processed-years", nargs="+", type=int, default=None)
    parser.add_argument("--expected-sites", type=int, default=15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    errors: list[str] = []
    for year in args.years:
        sst = args.sst_dir / f"sea_surface_temperature_{year}_openmeteo.csv"
        if not sst.exists():
            errors.append(f"missing SST file: {sst}")
        else:
            errors.extend(audit_csv(sst, year))

        wind_files = sorted(args.wind_dir.glob(f"*_era5_wind_{year}.csv"))
        if len(wind_files) != args.expected_sites:
            errors.append(
                f"wind {year}: files={len(wind_files)}, expected={args.expected_sites}")
        for path in wind_files:
            errors.extend(audit_csv(path, year))

        print(f"raw {year}: SST=1, wind={len(wind_files)}, "
              f"hours={len(expected_index(year))}")

    if args.processed_root is not None:
        for year in args.processed_years or args.years:
            processed_files = sorted(
                (args.processed_root / str(year)).glob("hourly_inputs_*.csv"))
            if len(processed_files) != args.expected_sites:
                errors.append(
                    f"processed {year}: files={len(processed_files)}, "
                    f"expected={args.expected_sites}")
            for path in processed_files:
                errors.extend(audit_csv(path, year))
            print(f"processed {year}: countries={len(processed_files)}, "
                  f"hours={len(expected_index(year))}")

    if errors:
        raise SystemExit("\n".join(errors))
    print("multiyear input audit: OK")


if __name__ == "__main__":
    main()
