"""Download hourly ERA5 10 m and 100 m wind speeds for selected sea points.

Open-Meteo's Historical Weather API exposes Copernicus ERA5 without requiring
CDS credentials. Outputs are one auditable UTC CSV per city and year.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


API_URL = "https://archive-api.open-meteo.com/v1/archive"


def expected_timestamps(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{year}-01-01", f"{year + 1}-01-01", inclusive="left", freq="h")


def output_filename(row: pd.Series, year: int) -> str:
    point_id = f"OW_{int(row['repo_city_index']):03d}"
    city = str(row["datacentermap_market"]).replace(" ", "_")
    return f"{point_id}_{city}_era5_wind_{year}.csv"


def fetch_point(row: pd.Series, year: int, timeout: int,
                cell_selection: str) -> pd.DataFrame:
    params = {
        "latitude": float(row["offshore_wind_lat"]),
        "longitude": float(row["offshore_wind_lon"]),
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": "wind_speed_10m,wind_speed_100m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "models": "era5",
        "cell_selection": cell_selection,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "coastaldc-era5-wind/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("error"):
        raise RuntimeError(payload.get("reason", "Open-Meteo API error"))

    hourly = payload.get("hourly", {})
    timestamps = pd.to_datetime(hourly.get("time", []))
    expected = expected_timestamps(year)
    if not pd.DatetimeIndex(timestamps).equals(expected):
        raise RuntimeError(
            f"{row['datacentermap_market']}: ERA5 timestamp axis is incomplete")

    frame = pd.DataFrame({
        "timestamp": expected,
        "wind_speed_10m": pd.to_numeric(
            hourly.get("wind_speed_10m", []), errors="coerce"),
        "wind_speed_100m": pd.to_numeric(
            hourly.get("wind_speed_100m", []), errors="coerce"),
        "requested_latitude": float(row["offshore_wind_lat"]),
        "requested_longitude": float(row["offshore_wind_lon"]),
        "resolved_latitude": float(payload["latitude"]),
        "resolved_longitude": float(payload["longitude"]),
    })
    if len(frame) != len(expected) or frame.isna().any().any():
        raise RuntimeError(
            f"{row['datacentermap_market']}: incomplete ERA5 wind values")
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Selected City_manifest XLSX")
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cell-selection", choices=["nearest", "sea", "land"],
                        default="nearest")
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    targets = pd.read_excel(args.input, sheet_name="City_manifest")
    targets = targets[targets["toolkit_ready"].astype(bool)].copy()
    if args.cities:
        targets = targets[targets["datacentermap_market"].isin(args.cities)].copy()
        missing_cities = sorted(set(args.cities) - set(targets["datacentermap_market"]))
        if missing_cities:
            raise ValueError(f"Cities not found in selected manifest: {missing_cities}")
    required = ["repo_city_index", "datacentermap_market",
                "offshore_wind_lat", "offshore_wind_lon"]
    if targets[required].isna().any().any():
        raise ValueError("Selected manifest contains missing wind coordinates")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (_, row) in enumerate(targets.iterrows(), start=1):
        output = output_dir / output_filename(row, args.year)
        if output.exists() and not args.overwrite:
            existing = pd.read_csv(output, usecols=[
                "timestamp", "wind_speed_10m", "wind_speed_100m"])
            if len(existing) == len(expected_timestamps(args.year)) \
                    and not existing.isna().any().any():
                print(f"[{index}/{len(targets)}] skip complete {output.name}")
                continue

        last_error = None
        for attempt in range(args.retries + 1):
            try:
                frame = fetch_point(
                    row, args.year, args.timeout, args.cell_selection)
                temporary = output.with_suffix(output.suffix + ".tmp")
                frame.to_csv(temporary, index=False)
                os.replace(temporary, output)
                print(f"[{index}/{len(targets)}] wrote {output.name}")
                last_error = None
                break
            except (OSError, RuntimeError, urllib.error.URLError) as error:
                last_error = error
                if attempt < args.retries:
                    wait = min(60.0, 2.0 * (2 ** attempt))
                    print(f"[{index}/{len(targets)}] retry in {wait:.0f}s: {error}")
                    time.sleep(wait)
        if last_error is not None:
            raise RuntimeError(
                f"Failed {row['datacentermap_market']}: {last_error}") from last_error
        time.sleep(args.pause)


if __name__ == "__main__":
    main()
