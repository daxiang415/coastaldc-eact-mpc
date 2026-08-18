"""Offshore-wind matching: same-hour wind use, residual grid purchase, CO2 and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WindConfig:
    wind_capacity_mw: float = 15.0    # nameplate offshore-wind capacity allocated to the DC


def match_wind(e_total_mwh: float, e_wind_available_mwh: float,
               carbon_intensity_kg_per_mwh: float, price_usd_per_mwh: float) -> dict:
    """Same-hour matching: wind first, residual bought from the grid."""
    e_wind_used = min(e_total_mwh, e_wind_available_mwh)
    e_grid = max(0.0, e_total_mwh - e_wind_available_mwh)
    e_wind_unused = max(0.0, e_wind_available_mwh - e_total_mwh)
    return {
        "e_wind_used_mwh": e_wind_used,
        "e_grid_mwh": e_grid,
        "e_wind_unused_mwh": e_wind_unused,
        "co2_kg": e_grid * carbon_intensity_kg_per_mwh,
        "cost_usd": e_grid * price_usd_per_mwh,
    }
