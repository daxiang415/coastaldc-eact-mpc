"""Build 15-country hourly inputs from the REAL coastal-zero-carbon-datacenter pipeline.

Sources (rank-1 city per country from the city manifest):
  SST      data/sst_download_toolkit/sea_surface_temperature_2025_openmeteo.csv (Open-Meteo)
  Wind     data/offshore_wind_download_toolkit/offshore_wind/OW_*.nc (ERA5; zip-wrapped)
  Carbon   data/ci_download_toolkit/city_grid_carbon_intensity_electricitymaps_10y.csv
  Workload data/Workload/GoogleClusteData_CPU_Data_Hourly_1.csv (or Alibaba traces)
  Price    CountryPricesRates xlsx (--price-table); constants fallback with warning

Wind conversion mirrors the original repo's renewables/wind_power.py:
power-law hub-height extrapolation + generic cubic offshore power curve.

Usage:
    python scripts/build_hourly_inputs_real.py --repo <path-to-coastal-zero-carbon-datacenter> \
        --price-table data/raw/CountryPricesRates.xlsx
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULT_YEAR = 2025

# ISO3 -> (manifest country name, rank-1 city / data column name, OW file prefix)
COUNTRY_CITY = {
    "USA": ("USA", "Los Angeles", "OW_001_"),
    "CHN": ("China", "Shanghai", "OW_006_"),
    "GBR": ("United Kingdom", "Cardiff", "OW_011_"),
    "AUS": ("Australia", "Sydney", "OW_016_"),
    "JPN": ("Japan", "Tokyo", "OW_021_"),
    "IRL": ("Ireland", "Dublin", "OW_026_"),
    "NOR": ("Norway", "Oslo", "OW_031_"),
    "IND": ("India", "Mumbai", "OW_036_"),
    "MYS": ("Malaysia", "Johor Bahru", "OW_041_"),
    "PRT": ("Portugal", "Lisbon", "OW_051_"),
    "KOR": ("South Korea", "Incheon", "OW_056_"),
    "NLD": ("The Netherlands", "Amsterdam", "OW_061_"),
    "CAN": ("Canada", "Vancouver", "OW_066_"),
    "IDN": ("Indonesia", "Jakarta", "OW_081_"),
    "SGP": ("Singapore", "Singapore", "OW_086_"),
}

# fallback electricity prices, USD/MWh (rough constants) -- replaced by --price-table
PLACEHOLDER_PRICES = {
    "CHN": 90.0, "USA": 80.0, "JPN": 140.0, "AUS": 100.0, "NOR": 60.0,
    "IND": 70.0, "PRT": 110.0, "MYS": 65.0, "GBR": 150.0, "SGP": 120.0,
    "KOR": 95.0, "IDN": 75.0, "NLD": 130.0, "IRL": 140.0, "CAN": 55.0,
}


# ------------------------------------------------------------------ wind (from wind_power.py)
def expected_hourly_index(year: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=f"{year}-01-01 00:00",
        end=f"{year + 1}-01-01 00:00",
        inclusive="left",
        freq="h",
        name="timestamp",
    )


def validate_hourly_series(series: pd.Series, year: int, label: str) -> pd.Series:
    """Return a complete, ordered UTC hourly series for one calendar year."""
    index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True)).tz_convert(None)
    values = pd.to_numeric(pd.Series(series.values), errors="coerce").to_numpy()
    clean = pd.Series(values, index=index, name=series.name).sort_index()
    if clean.index.duplicated().any():
        count = int(clean.index.duplicated().sum())
        raise ValueError(f"{label}: found {count} duplicate timestamps")

    expected = expected_hourly_index(year)
    missing = expected.difference(clean.index)
    extra = clean.index.difference(expected)
    if len(missing):
        raise ValueError(f"{label}: missing {len(missing)} hourly timestamps for {year}")
    if len(extra):
        raise ValueError(f"{label}: found {len(extra)} timestamps outside {year}")

    clean = clean.reindex(expected).interpolate(limit_direction="both")
    if clean.isna().any():
        raise ValueError(f"{label}: contains {int(clean.isna().sum())} unresolved values")
    return clean.astype(float)


def hub_height_wind(v10, v100, hub_height_m=150.0, default_alpha=0.11):
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.log(v100 / v10) / np.log(100.0 / 10.0)
    alpha = np.where(np.isfinite(alpha), np.clip(alpha, -0.05, 0.40), default_alpha)
    return v100 * (hub_height_m / 100.0) ** alpha


def offshore_power_fraction(v, cut_in=3.0, rated=12.0, cut_out=25.0):
    pf = np.zeros_like(v)
    mid = (v >= cut_in) & (v < rated)
    pf[mid] = np.clip((v[mid] ** 3 - cut_in**3) / (rated**3 - cut_in**3), 0, 1)
    pf[(v >= rated) & (v <= cut_out)] = 1.0
    return pf


def load_wind_mw(ow_dir: str, prefix: str, wind_capacity_mw: float,
                 year: int, wind_csv_dir: str | None = None,
                 wind_source: str = "auto") -> pd.Series:
    if wind_source not in {"auto", "netcdf", "openmeteo"}:
        raise ValueError("wind_source must be auto, netcdf, or openmeteo")
    matches = glob.glob(os.path.join(
        ow_dir, f"{prefix}*{year}-01-01_{year}-12-31.nc"))
    if matches and wind_source in {"auto", "netcdf"}:
        import xarray as xr
        path = matches[0]
        with tempfile.TemporaryDirectory() as td:
            if zipfile.is_zipfile(path):  # CDS delivers zip-wrapped netCDF
                with zipfile.ZipFile(path) as z:
                    z.extractall(td)
                inner = glob.glob(os.path.join(td, "*.nc"))[0]
            else:
                inner = path
            with xr.open_dataset(inner) as ds:
                time_coord = next(
                    (name for name in ds.coords if "time" in name.lower()), None)
                if time_coord is None:
                    raise ValueError(f"{path}: no time coordinate found")
                timestamps = pd.to_datetime(ds[time_coord].values, utc=True)
                u100 = ds["u100"].values.squeeze()
                v100 = ds["v100"].values.squeeze()
                u10 = ds["u10"].values.squeeze()
                v10 = ds["v10"].values.squeeze()
        s100 = np.hypot(u100, v100)
        s10 = np.hypot(u10, v10)
        label = f"ERA5 wind {prefix}"
    else:
        csv_matches = [] if wind_csv_dir is None else glob.glob(os.path.join(
            wind_csv_dir, f"{prefix}*era5_wind_{year}.csv"))
        if wind_source == "netcdf":
            raise FileNotFoundError(
                f"No {year} ERA5 NetCDF for {prefix} in {ow_dir}")
        if not csv_matches:
            raise FileNotFoundError(
                f"No {year} ERA5 NetCDF or Open-Meteo CSV for {prefix}")
        path = csv_matches[0]
        frame = pd.read_csv(path)
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        s10 = pd.to_numeric(frame["wind_speed_10m"], errors="coerce").to_numpy()
        s100 = pd.to_numeric(frame["wind_speed_100m"], errors="coerce").to_numpy()
        label = f"Open-Meteo ERA5 wind {prefix}"

    hub = hub_height_wind(np.maximum(s10, 0.1), np.maximum(s100, 0.1))
    power = offshore_power_fraction(hub) * wind_capacity_mw
    return validate_hourly_series(
        pd.Series(power, index=timestamps, name="wind_mw"),
        year,
        label,
    )


# ------------------------------------------------------------------ other sources
def load_sst(repo: str, city: str, year: int,
             sst_dir: str | None = None) -> pd.Series:
    candidates = []
    if sst_dir is not None:
        candidates.append(os.path.join(
            sst_dir, f"sea_surface_temperature_{year}_openmeteo.csv"))
    candidates.append(os.path.join(
        repo, "data", "sst_download_toolkit",
        f"sea_surface_temperature_{year}_openmeteo.csv"))
    path = next((candidate for candidate in candidates if os.path.exists(candidate)), None)
    if path is None:
        raise FileNotFoundError(f"No {year} SST file in: {candidates}")
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["timestamp", city])
    return validate_hourly_series(
        pd.Series(df[city].values, index=df["timestamp"], name="sst_c"),
        year,
        f"Open-Meteo SST {city}",
    )


def load_carbon(repo: str, city: str, year: int) -> pd.Series:
    path = os.path.join(repo, "data", "ci_download_toolkit",
                        "city_grid_carbon_intensity_electricitymaps_10y.csv")
    df = pd.read_csv(path, usecols=["timestamp", city])
    ts = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    mask = (ts >= f"{year}-01-01") & (ts < f"{year + 1}-01-01")
    return validate_hourly_series(
        pd.Series(df.loc[mask, city].values, index=ts[mask], name="carbon_kg_per_mwh"),
        year,
        f"Electricity Maps carbon {city}",
    )


def load_workload(repo: str, trace: str, it_capacity_mw: float,
                  hours: int,
                  flexible_fraction: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    fname = {
        "google": "GoogleClusteData_CPU_Data_Hourly_1.csv",
        "alibaba1": "Alibaba_CPU_Data_Hourly_1.csv",
        "alibaba2": "Alibaba_CPU_Data_Hourly_2.csv",
    }[trace]
    df = pd.read_csv(os.path.join(repo, "data", "Workload", fname))
    cpu = pd.to_numeric(df["cpu_load"], errors="coerce").interpolate(limit_direction="both").values
    if len(cpu) < hours:
        cpu = np.tile(cpu, int(np.ceil(hours / len(cpu))))
    cpu = np.clip(cpu[:hours], 0.0, 1.0)
    total = cpu * it_capacity_mw
    flexible = flexible_fraction * total
    return total - flexible, flexible


def load_prices(price_table: str | None) -> dict[str, float]:
    if price_table is None:
        print("WARNING: no --price-table given; using PLACEHOLDER country price constants.")
        return dict(PLACEHOLDER_PRICES)
    xl = pd.ExcelFile(price_table)
    sheet = next((s for s in xl.sheet_names if "elec" in s.lower() or "price" in s.lower()),
                 xl.sheet_names[0])
    df, ccol, pcol = None, None, None
    for header in (0, 1, 2):  # header row position varies in the workbook
        cand = xl.parse(sheet, header=header)
        cols = {str(c).lower().strip(): c for c in cand.columns}
        ccol = cols.get("country")
        pcol = next((c for k, c in cols.items() if "usd" in k and "price" in k), None)
        if ccol is not None and pcol is not None:
            df = cand.dropna(subset=[ccol, pcol])
            break
    if df is None:
        raise ValueError(f"Could not find Country / USD price columns in sheet '{sheet}'")
    name_to_iso = {v[0].lower(): k for k, v in COUNTRY_CITY.items()}
    extra = {"united states": "USA", "united states of america": "USA",
             "korea": "KOR", "south korea": "KOR",
             "netherlands": "NLD", "the netherlands": "NLD", "china": "CHN",
             "united kingdom": "GBR", "uk": "GBR"}
    name_to_iso.update(extra)
    prices = dict(PLACEHOLDER_PRICES)
    matched = 0
    for _, row in df.iterrows():
        iso = name_to_iso.get(str(row[ccol]).lower().strip())
        val = pd.to_numeric(row[pcol], errors="coerce")
        if iso and np.isfinite(val):
            prices[iso] = float(val)
            matched += 1
    print(f"price table: matched {matched} countries from '{sheet}'")
    return prices


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="path to coastal-zero-carbon-datacenter repository")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
    parser.add_argument("--price-table", default=None, help="CountryPricesRates xlsx")
    parser.add_argument("--sst-dir", default=None,
                        help="optional directory containing additional yearly SST CSVs")
    parser.add_argument("--wind-csv-dir", default=None,
                        help="optional directory containing Open-Meteo ERA5 wind CSVs")
    parser.add_argument("--wind-source", choices=["auto", "netcdf", "openmeteo"],
                        default="auto")
    parser.add_argument("--workload", choices=["google", "alibaba1", "alibaba2"],
                        default="google")
    parser.add_argument("--years", nargs="+", type=int, default=[DEFAULT_YEAR],
                        help="UTC calendar years to align and concatenate")
    parser.add_argument("--it-capacity-mw", type=float, default=10.0)
    parser.add_argument("--wind-capacity-mw", type=float, default=15.0)
    return parser


def main():
    args = build_parser().parse_args()

    os.makedirs(args.out, exist_ok=True)
    ow_dir = os.path.join(args.repo, "data", "offshore_wind_download_toolkit", "offshore_wind")
    prices = load_prices(args.price_table)
    total_hours = sum(len(expected_hourly_index(year)) for year in args.years)
    fixed, flexible = load_workload(
        args.repo, args.workload, args.it_capacity_mw, total_hours)

    summary = []
    for iso, (country, city, prefix) in COUNTRY_CITY.items():
        frames = []
        offset = 0
        for year in args.years:
            timestamps = expected_hourly_index(year)
            n = len(timestamps)
            sst = load_sst(args.repo, city, year, args.sst_dir)
            wind = load_wind_mw(
                ow_dir, prefix, args.wind_capacity_mw, year, args.wind_csv_dir,
                args.wind_source)
            ci = load_carbon(args.repo, city, year)
            frames.append(pd.DataFrame({
                "timestamp": timestamps,
                "hour": np.arange(offset, offset + n),
                "fixed_load_mw": fixed[offset:offset + n],
                "flexible_arrival_mw": flexible[offset:offset + n],
                "sst_c": sst.to_numpy(),
                "wind_mw": wind.to_numpy(),
                "carbon_kg_per_mwh": ci.to_numpy(),
                "price_usd_per_mwh": np.full(n, prices[iso]),
            }))
            offset += n
        df = pd.concat(frames, ignore_index=True)
        assert not df.isna().any().any(), f"NaN remaining in {iso}"
        path = os.path.join(args.out, f"hourly_inputs_{iso}.csv")
        df.to_csv(path, index=False)
        summary.append({"country": iso, "city": city, "years": ",".join(map(str, args.years)),
                        "rows": len(df),
                        "sst_mean": df.sst_c.mean(),
                        "wind_cf": df.wind_mw.mean() / args.wind_capacity_mw,
                        "ci_mean": df.carbon_kg_per_mwh.mean(), "price": prices[iso]})
        print(f"{iso} ({city}): rows={len(df)} sst={df.sst_c.mean():.1f}C "
              f"cf={df.wind_mw.mean()/args.wind_capacity_mw:.2f} "
              f"ci={df.carbon_kg_per_mwh.mean():.0f} price={prices[iso]:.1f}")

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "country_price_inputs.csv"), index=False)
    print(f"\nwrote {len(summary)} country files -> {args.out}")


if __name__ == "__main__":
    main()
