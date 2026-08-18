"""Verify the packaged controller inputs and result-source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COUNTRIES = {
    "AUS", "CAN", "CHN", "GBR", "IDN", "IND", "IRL", "JPN",
    "KOR", "MYS", "NLD", "NOR", "PRT", "SGP", "USA",
}
EXPECTED_ROWS = {"2023": 8760, "2024": 8784, "2025": 8760}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs() -> None:
    root = ROOT / "data" / "processed_multiyear"
    for year, expected_rows in EXPECTED_ROWS.items():
        paths = sorted((root / year).glob("hourly_inputs_*.csv"))
        countries = {path.stem.rsplit("_", 1)[-1] for path in paths}
        if countries != COUNTRIES:
            raise AssertionError(f"{year}: unexpected country set {countries}")
        for path in paths:
            if len(pd.read_csv(path)) != expected_rows:
                raise AssertionError(f"{path}: expected {expected_rows} rows")

    train_paths = sorted((root / "train_2023_2024").glob("hourly_inputs_*.csv"))
    if len(train_paths) != len(COUNTRIES):
        raise AssertionError("Expected 15 combined training files")
    for path in train_paths:
        if len(pd.read_csv(path)) != 17544:
            raise AssertionError(f"{path}: expected 17544 rows")


def verify_figure_data() -> None:
    package = ROOT / "results" / "figure_data"
    manifest = json.loads(
        (package / "figure_build_manifest.json").read_text(encoding="utf-8")
    )
    if len(manifest["data_files"]) != 6:
        raise AssertionError("Expected six figure-data entries")
    for item in manifest["data_files"]:
        path = package / item["file"]
        if sha256(path) != item["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: {path}")
        if len(pd.read_csv(path)) != int(item["rows"]):
            raise AssertionError(f"Row-count mismatch: {path}")


def verify_capacity_statistics() -> dict[str, float]:
    table_data = ROOT / "results" / "figure_data" / "table_data"
    pairs = pd.read_csv(table_data / "table_s3_capacity_pair_distribution.csv")
    loo = pd.read_csv(table_data / "table_s3_capacity_leave_one_out.csv")
    if len(pairs) != 12 or len(loo) != 12:
        raise AssertionError("Expected 12 paired capacity-stress observations")

    static_events = float(pairs["static_recommended_exceedance_hours"].sum())
    eact_events = float(pairs["eact_recommended_exceedance_hours"].sum())
    static_degree = float(pairs["static_recommended_exceedance_degc_h"].sum())
    eact_degree = float(pairs["eact_recommended_exceedance_degc_h"].sum())
    reduction_pct = 100.0 * (static_degree - eact_degree) / static_degree
    loo_min = float(loo["degree_hour_reduction_pct"].min())

    if not np.isclose(static_events, 45.0) or not np.isclose(eact_events, 7.0):
        raise AssertionError("Capacity-stress event-hour totals changed")
    if not np.isclose(static_degree, 18.181674, atol=1e-6):
        raise AssertionError("Static degree-hour total changed")
    if not np.isclose(eact_degree, 1.500015, atol=1e-6):
        raise AssertionError("EACT degree-hour total changed")
    if not 91.7 <= reduction_pct < 91.8:
        raise AssertionError("Aggregate degree-hour reduction changed")
    if not 83.4 <= loo_min < 83.5:
        raise AssertionError("Leave-one-pair-out minimum changed")

    return {
        "static_event_hours": static_events,
        "eact_event_hours": eact_events,
        "static_degree_hours": static_degree,
        "eact_degree_hours": eact_degree,
        "degree_hour_reduction_pct": reduction_pct,
        "leave_one_out_min_reduction_pct": loo_min,
    }


def main() -> None:
    verify_inputs()
    verify_figure_data()
    summary = verify_capacity_statistics()
    print(json.dumps(summary, indent=2))
    print("Release verification: OK")


if __name__ == "__main__":
    main()
