# CoastalDC EACT-MPC: code and data

This repository contains the executable environment, controllers, experiment
scripts, processed inputs, reduced result outputs, and figure-source data for
the CoastalDC EACT-MPC study. Manuscript source, submission documents, compiled
figures, and PDFs are intentionally excluded.

## Repository layout

```text
coastaldc_env/                 Continuous coastal data-centre environment
controllers/                  Nominal, static-robust, and EACT-MPC controllers
scripts/                      Data, experiment, analysis, and audit entry points
tests/                        Unit and integration tests
data/processed_multiyear/     Exact 2023-2025 controller input tables
data/raw/multiyear/           Redistributable source and provenance files
results/causal_forecasts_v3_gated_bias/
                               Frozen causal forecast models and residuals
results/reduced_outputs/      Summary, weekly, manifest, and analysis outputs
results/figure_data/          One-CSV-per-figure and table-source data
```

The original hourly experiment outputs and training checkpoints total more
than 2.6 GB and are not stored here. They can be regenerated from the included
inputs and scripts. The reduced outputs retain the controller summaries,
weekly estimates, run settings, pair-level stress outcomes, and statistical
audits needed to inspect the reported comparisons.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick verification

```bash
python scripts/verify_release.py
python scripts/check_env.py --country JPN
python -m pytest tests -q
```

`verify_release.py` checks the multiyear input dimensions, figure-data hashes,
and the aggregate and leave-one-pair-out compound-stress statistics directly
from the included CSV files.

## Reproduction paths

- Inspect the packaged evidence: run `python scripts/verify_release.py`.
- Refit causal forecasts: run `python scripts/fit_causal_forecasts.py`.
- Rerun controller experiments: use the `run_eact_*` and
  `run_ashrae_inlet_*` scripts.
- Recompute analyses: use the matching `analyze_*` scripts.
- Regenerate figures from source CSVs: run
  `python results/figure_data/scripts/plot_figures.py`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the staged workflow and
[DATA.md](DATA.md) for provenance, splits, schemas, and redistribution notes.

## Scope

The repository supports simulation and numerical reproduction. It does not
constitute field validation, equipment certification, or a general recursive
feasibility guarantee. No software or data licence is granted by the absence
of a licence file; third-party data terms must be checked before making this
private repository public.
