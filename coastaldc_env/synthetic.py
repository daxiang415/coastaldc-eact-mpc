"""Synthetic placeholder input data for the 15 coastal-capacity countries.

Interface-compatible with the real data pipeline (Open-Meteo SST, ERA5 offshore wind,
Electricity Maps carbon intensity, country electricity prices). Replace the generated
CSVs in data/processed/ with real data without touching the environment code.

Columns of hourly_inputs_<ISO3>.csv:
    hour, fixed_load_mw, flexible_arrival_mw, sst_c, wind_mw,
    carbon_kg_per_mwh, price_usd_per_mwh
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ISO3 -> (sst_mean, sst_amp, south_hemisphere, wind_cf, carbon_mean_kg_per_mwh, price_usd_per_mwh)
COUNTRY_PARAMS = {
    "CHN": (18.0, 8.0, False, 0.30, 550.0, 90.0),
    "USA": (15.0, 7.0, False, 0.38, 380.0, 80.0),
    "JPN": (17.0, 8.0, False, 0.32, 480.0, 140.0),
    "AUS": (19.0, 5.0, True, 0.40, 600.0, 100.0),
    "NOR": (8.0, 4.0, False, 0.45, 30.0, 60.0),
    "IND": (28.0, 2.0, False, 0.28, 650.0, 70.0),
    "PRT": (16.0, 3.0, False, 0.38, 200.0, 110.0),
    "MYS": (29.0, 1.0, False, 0.18, 550.0, 65.0),
    "GBR": (11.0, 4.0, False, 0.45, 220.0, 150.0),
    "SGP": (29.0, 1.0, False, 0.15, 400.0, 120.0),
    "KOR": (15.0, 9.0, False, 0.32, 450.0, 95.0),
    "IDN": (29.0, 1.0, False, 0.17, 650.0, 75.0),
    "NLD": (11.0, 5.0, False, 0.42, 350.0, 130.0),
    "IRL": (11.0, 3.0, False, 0.48, 290.0, 140.0),
    "CAN": (8.0, 6.0, False, 0.40, 120.0, 55.0),
}

HOURS_PER_YEAR = 8760


def generate_country_inputs(country: str, it_capacity_mw: float = 10.0,
                            wind_capacity_mw: float = 15.0,
                            n_hours: int = HOURS_PER_YEAR,
                            seed: int | None = None) -> pd.DataFrame:
    """Generate one year of plausible hourly inputs for a country."""
    if country not in COUNTRY_PARAMS:
        raise ValueError(f"Unknown country '{country}'. Choose from {sorted(COUNTRY_PARAMS)}")
    sst_mean, sst_amp, south, wind_cf, ci_mean, price_base = COUNTRY_PARAMS[country]

    rng = np.random.default_rng(seed if seed is not None else abs(hash(country)) % (2**32))
    h = np.arange(n_hours)
    hod = h % 24
    doy = (h // 24) % 365

    # --- sea-surface temperature: annual sinusoid + slow noise ---
    phase = np.pi if south else 0.0
    sst = (sst_mean
           + sst_amp * np.sin(2 * np.pi * (doy - 105) / 365 + phase)
           + _smooth_noise(rng, n_hours, sigma=0.6, tau=72))

    # --- offshore wind: AR(1) capacity factor around country mean ---
    cf = np.clip(wind_cf + _smooth_noise(rng, n_hours, sigma=0.22, tau=12), 0.0, 1.0)
    # occasional lulls / storms
    wind = cf * wind_capacity_mw

    # --- grid carbon intensity: diurnal swing + noise ---
    ci = ci_mean * (1.0
                    + 0.20 * np.sin(2 * np.pi * (hod - 14) / 24)
                    + _smooth_noise(rng, n_hours, sigma=0.06, tau=24))
    ci = np.clip(ci, 5.0, None)

    # --- electricity price: base + peak-hour uplift + noise ---
    peak = ((hod >= 8) & (hod <= 20)).astype(float)
    price = price_base * (1.0 + 0.25 * peak
                          + _smooth_noise(rng, n_hours, sigma=0.08, tau=24))
    price = np.clip(price, 5.0, None)

    # --- workload: fairly flat AI load with mild diurnal shape ---
    diurnal = 1.0 + 0.10 * np.sin(2 * np.pi * (hod - 15) / 24)
    weekly = 1.0 - 0.05 * (((h // 24) % 7) >= 5)
    total_arrival = 0.72 * it_capacity_mw * diurnal * weekly \
        * (1.0 + _smooth_noise(rng, n_hours, sigma=0.05, tau=6))
    total_arrival = np.clip(total_arrival, 0.0, it_capacity_mw)
    flexible = 0.30 * total_arrival
    fixed = total_arrival - flexible

    return pd.DataFrame({
        "hour": h,
        "fixed_load_mw": fixed,
        "flexible_arrival_mw": flexible,
        "sst_c": sst,
        "wind_mw": wind,
        "carbon_kg_per_mwh": ci,
        "price_usd_per_mwh": price,
    })


def _smooth_noise(rng: np.random.Generator, n: int, sigma: float, tau: float) -> np.ndarray:
    """AR(1) noise with correlation time `tau` hours and stationary std `sigma`."""
    rho = np.exp(-1.0 / tau)
    eps = rng.normal(0.0, sigma * np.sqrt(1 - rho**2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x
