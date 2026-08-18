"""Compare 2025 CDS NetCDF and Open-Meteo ERA5 wind-power inputs."""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.build_hourly_inputs_real import COUNTRY_CITY, load_wind_mw  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--wind-csv-dir", required=True)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--wind-capacity-mw", type=float, default=15.0)
    parser.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    netcdf_dir = os.path.join(
        args.repo, "data", "offshore_wind_download_toolkit", "offshore_wind")
    rows = []
    for country, (_, city, prefix) in COUNTRY_CITY.items():
        cds = load_wind_mw(
            netcdf_dir, prefix, args.wind_capacity_mw, args.year,
            wind_source="netcdf")
        openmeteo = load_wind_mw(
            netcdf_dir, prefix, args.wind_capacity_mw, args.year,
            args.wind_csv_dir, "openmeteo")
        difference = openmeteo.to_numpy() - cds.to_numpy()
        rows.append({
            "country": country,
            "city": city,
            "correlation": float(np.corrcoef(cds, openmeteo)[0, 1]),
            "mae_mw": float(np.mean(np.abs(difference))),
            "rmse_mw": float(np.sqrt(np.mean(difference ** 2))),
            "bias_mw": float(np.mean(difference)),
            "capacity_factor_cds": float(cds.mean() / args.wind_capacity_mw),
            "capacity_factor_openmeteo": float(
                openmeteo.mean() / args.wind_capacity_mw),
        })

    result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))
    print("\nsummary")
    print(f"mean correlation: {result.correlation.mean():.4f}")
    print(f"minimum correlation: {result.correlation.min():.4f}")
    print(f"mean MAE: {result.mae_mw.mean():.3f} MW")
    print("mean absolute capacity-factor difference: "
          f"{np.mean(np.abs(result.capacity_factor_openmeteo - result.capacity_factor_cds)):.4f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
