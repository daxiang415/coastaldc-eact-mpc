import numpy as np
import pandas as pd

from coastaldc_env import COUNTRIES
from scripts.analyze_ashrae_inlet_beta_ablation import paired_comparisons
from scripts.analyze_ashrae_inlet_pilot import START_HOURS


def test_beta_comparison_reports_positive_effect_when_final_is_better():
    rows = []
    for country in COUNTRIES:
        for start_hour in START_HOURS:
            for variant, common_cost, p95 in (
                ("beta0", 100.0, 27.0),
                ("beta0p10", 99.0, 26.8),
            ):
                rows.append({
                    "country": country,
                    "start_hour": start_hour,
                    "variant": variant,
                    "episode_return": -common_cost,
                    "e_grid_mwh": common_cost,
                    "co2_kg": common_cost,
                    "e_total_mwh": common_cost,
                    "recommended_exceedance_degc_h": p95 - 26.0,
                    "recommended_exceedance_hours": p95 - 26.0,
                    "recommended_compliance_pct": 100.0 - (p95 - 26.0),
                    "p95_t_inlet_c": p95,
                    "p99_t_inlet_c": p95,
                    "max_t_inlet_c": p95,
                    "min_allowable_headroom_c": 32.0 - p95,
                    "allowable_exceedance_hours": 0.0,
                    "allowable_exceedance_degc_h": 0.0,
                    "temporal_rci_hi_pct": 100.0 - (p95 - 26.0),
                })

    result = paired_comparisons(
        pd.DataFrame(rows), samples=100, seed=1)
    common = result[result.metric == "common_cost"].iloc[0]
    p95 = result[result.metric == "p95_t_inlet_c"].iloc[0]

    assert np.isclose(common.mean_relative_improvement_pct, 1.0)
    assert np.isclose(p95.mean_improvement, 0.2)
