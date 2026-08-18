# Data provenance and structure

## Experimental split

- **2023-2024:** causal forecast fitting, internal model selection, residual
  calibration, and controller configuration.
- **2025:** held-out controller evaluation.

The 15 country codes are `AUS`, `CAN`, `CHN`, `GBR`, `IDN`, `IND`, `IRL`,
`JPN`, `KOR`, `MYS`, `NLD`, `NOR`, `PRT`, `SGP`, and `USA`. Timestamps are
hourly UTC. The 2023 and 2025 files contain 8,760 rows per country; 2024
contains 8,784 rows; and each combined 2023-2024 training file contains 17,544
rows.

## Controller inputs

The exact controller input files are under:

```text
data/processed_multiyear/2023/
data/processed_multiyear/2024/
data/processed_multiyear/2025/
data/processed_multiyear/train_2023_2024/
```

Each country file contains hourly workload, seawater temperature, offshore
wind, grid carbon intensity, and the associated time fields required by the
environment and forecast pipeline. `country_price_inputs.csv` is retained for
schema compatibility; the final energy-accounting comparisons do not require
an electricity-price objective.

## Source data

### Sea-surface temperature

Sea-surface temperature was obtained from the Open-Meteo Marine API at the
representative sea-point coordinates in
`data/raw/multiyear/selected_15_city_manifest.xlsx`. The 2023 and 2024 source
files are included. The included 2025 Open-Meteo redownload contains gaps and
is retained only as a source-consistency record; the exact 2025 values used by
the controller are preserved in `data/processed_multiyear/2025/`.

### Offshore wind

Wind data were obtained from the Open-Meteo Historical Weather API with the
ERA5 model, hourly 10 m and 100 m wind speed, UTC timestamps, and nearest-cell
selection at the audited offshore coordinates. The same extraction and power
conversion pipeline was used in all three years. The included comparison CSV
records the 2025 consistency check against the original Copernicus ERA5 data.

### Grid carbon intensity

Hourly direct operational carbon intensity was derived from Electricity Maps
source data and filtered independently for each year and city mapping. The
third-party raw source is not redistributed; the transformed controller inputs
used in the experiments are included.

### Workload

The workload profile was derived from a Google cluster CPU trace, scaled to a
10 MW IT capacity scenario, with 70% fixed and 30% flexible demand. It is a
controlled workload scenario rather than observed annual demand for a specific
data centre.

## Result data

- `results/reduced_outputs/` retains summary, weekly, run-manifest, sensitivity,
  and statistical-analysis files while excluding hourly trajectories, solver
  traces, logs, and checkpoints.
- `results/figure_data/plot_data/` contains exactly one CSV for each of the six
  figures.
- `results/figure_data/table_data/` contains the table-source and pair-level
  audit CSVs.
- `results/causal_forecasts_v3_gated_bias/` contains the frozen Ridge models
  and calibration residual arrays for all 15 countries.

Run `python scripts/verify_release.py` to verify row counts, source-data hashes,
and key aggregate statistics. `FILE_MANIFEST_SHA256.csv` records a repository
file inventory generated immediately before the Git commit.

## Redistribution note

This repository is initially private. Open-Meteo files are included with their
provenance, while some processed columns derive from third-party sources whose
redistribution terms must be reviewed before any public release. The code and
data are provided for internal reproducibility; no implicit licence is granted.
