# Multiyear input provenance

## Experiment split

- 2023-2024: development data used for causal model fitting, internal model
  selection, forecast-residual calibration, and controller configuration
- 2025: the only held-out evaluation year

Within the development period, Ridge regularisation is selected by a trailing
blocked causal split inside 2023. Expanding-window forecasts over 2024 produce
out-of-sample calibration residuals, after which the final forecasting model is
refit on the complete 2023-2024 data. Thus, 2024 is not a separate paper-level
validation set and no 2025 result is used for model or controller selection.

All timestamps are hourly UTC. The selected manifest contains the same 15
rank-1 coastal cities used by the model inputs.

## Sea-surface temperature

Source: Open-Meteo Marine API, variable `sea_surface_temperature`, using the
representative sea-point coordinates in `selected_15_city_manifest.xlsx`.

- 2023: `sst/sea_surface_temperature_2023_openmeteo.csv`
- 2024: `sst/sea_surface_temperature_2024_openmeteo.csv`
- 2025 authoritative input: the complete original Open-Meteo extraction is not
  redistributed here; its exact transformed values are retained in
  `data/processed_multiyear/2025/`.

The local 2025 redownload is retained only as a source-consistency check. The
API response contained 4,680 missing cells that the downloader interpolated,
so it is not used to build the 2025 processed test set. Its mean absolute
difference from the complete original file is 0.026 C, with a maximum absolute
difference of 0.1 C.

The incomplete 2022 SST download is excluded from every processed dataset.

## Offshore wind

Source: Open-Meteo Historical Weather API with `models=era5`, hourly
`wind_speed_10m` and `wind_speed_100m`, UTC, and `cell_selection=nearest` at
the audited ERA5 sea-point coordinates. Files are under `wind/`.

The same downloader, coordinates, variables, grid-selection rule, hub-height
extrapolation, and turbine power curve are used for 2023, 2024, and 2025.

The 2025 Open-Meteo data were checked against the original Copernicus CDS ERA5
NetCDF inputs. Across 15 countries, the minimum hourly wind-power correlation
is 0.999967 and the mean absolute error is 0.015 MW. The comparison table is
`results/wind_source_comparison_2025.csv`.

## Grid carbon intensity

Source: Electricity Maps hourly direct operational carbon intensity in the
original 10-year city-level file. The same UTC file and city mapping are
filtered independently for 2023, 2024, and 2025.

## Workload

Source: the same Google Cluster CPU trace for all years, scaled to 10 MW IT
capacity and split into 70% fixed and 30% flexible demand. This is a controlled
workload scenario, not a claim of year-specific observed data-center demand.

## Processed outputs

- `data/processed_multiyear/2023`
- `data/processed_multiyear/2024`
- `data/processed_multiyear/2025`

Audit command:

```powershell
python scripts/audit_multiyear_inputs.py --sst-dir data/raw/multiyear/sst --wind-dir data/raw/multiyear/wind --years 2023 2024 2025 --processed-root data/processed_multiyear --processed-years 2023 2024 2025 --expected-sites 15
```
