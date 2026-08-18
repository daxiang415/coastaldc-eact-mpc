# Reproducibility workflow

Run all commands from the repository root.

## 1. Verify the packaged evidence

```bash
python scripts/verify_release.py
python scripts/check_env.py --country JPN
python -m pytest tests -q
```

The first command is lightweight and does not rerun MPC. It verifies the exact
input tables, figure-data hashes, and compound-stress aggregates included in
the repository.

## 2. Audit the multiyear inputs

```bash
python scripts/audit_multiyear_inputs.py \
  --sst-dir data/raw/multiyear/sst \
  --wind-dir data/raw/multiyear/wind \
  --years 2023 2024 2025 \
  --processed-root data/processed_multiyear \
  --processed-years 2023 2024 2025 \
  --expected-sites 15
```

## 3. Refit and evaluate causal forecasts

```bash
python scripts/fit_causal_forecasts.py
python scripts/evaluate_causal_forecasts.py
```

The frozen fitted models and residual arrays used for the packaged evaluations
are already available under `results/causal_forecasts_v3_gated_bias/`.

## 4. Rerun the controller experiments

The following entry points expose their complete protocol through `--help`:

```bash
python scripts/run_eact_final_seasonal.py --help
python scripts/run_eact_final_annual.py --help
python scripts/run_eact_thermal_capacity_stress.py --help
python scripts/run_eact_beta_ablation.py --help
python scripts/run_eact_weight_sensitivity.py --help
```

Full reruns are computationally expensive and recreate the hourly outputs that
were omitted from GitHub. Use explicit output directories when retaining both
the packaged reduced data and a new run.

## 5. Recompute statistics

Use the analysis script paired with each run family:

```bash
python scripts/analyze_ashrae_inlet_seasonal.py --help
python scripts/analyze_ashrae_inlet_annual.py --help
python scripts/analyze_ashrae_inlet_thermal_capacity.py --help
python scripts/analyze_ashrae_inlet_beta_ablation.py --help
python scripts/analyze_ashrae_inlet_weight_sensitivity.py --help
python scripts/analyze_ashrae_inlet_stress_boundary.py --help
```

After recreating the full capacity-stress hourly results, run:

```bash
python scripts/audit_capacity_thermal_statistics.py
```

## 6. Regenerate figures

```bash
python results/figure_data/scripts/plot_figures.py
```

The script reads only `results/figure_data/plot_data/*.csv` and writes generated
graphics to `results/figure_data/output/`. Generated graphics are ignored by
Git and are not part of this code-and-data repository.
