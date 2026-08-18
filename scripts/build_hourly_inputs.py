"""Build 15-country hourly input CSVs.

Currently uses the synthetic placeholder generator (interface-compatible with the real
pipeline). To switch to real data, replace generate_country_inputs() calls with loaders
for: coastal city manifest xlsx, Open-Meteo SST, ERA5 offshore wind, Electricity Maps
carbon intensity, and the country electricity-price table.

Usage:
    python scripts/build_hourly_inputs.py [--out data/processed] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from coastaldc_env.synthetic import COUNTRY_PARAMS, generate_country_inputs  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--it-capacity-mw", type=float, default=10.0)
    parser.add_argument("--wind-capacity-mw", type=float, default=15.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    price_rows = []
    for i, country in enumerate(COUNTRY_PARAMS):
        df = generate_country_inputs(country, it_capacity_mw=args.it_capacity_mw,
                                     wind_capacity_mw=args.wind_capacity_mw,
                                     seed=args.seed + i)
        path = os.path.join(args.out, f"hourly_inputs_{country}.csv")
        df.to_csv(path, index=False)
        price_rows.append({"country": country,
                           "price_usd_per_mwh_mean": df["price_usd_per_mwh"].mean(),
                           "carbon_kg_per_mwh_mean": df["carbon_kg_per_mwh"].mean(),
                           "sst_c_mean": df["sst_c"].mean(),
                           "wind_cf_mean": df["wind_mw"].mean() / args.wind_capacity_mw})
        print(f"wrote {path} ({len(df)} rows)")

    pd.DataFrame(price_rows).to_csv(os.path.join(args.out, "country_price_inputs.csv"), index=False)
    print("wrote country_price_inputs.csv")


if __name__ == "__main__":
    main()
