import numpy as np
import pandas as pd

from scripts.analyze_ashrae_inlet_annual import annual_thermal_comparisons


def test_annual_thermal_comparison_preserves_favorable_direction():
    rows = []
    for country in ("JPN", "CHN", "NOR"):
        for week in range(52):
            for algorithm, p95 in (
                ("nominal_causal_mpc", 27.1),
                ("static_robust_mpc", 27.0),
                ("eact_mpc", 26.8),
            ):
                rows.append({
                    "country": country,
                    "week": week,
                    "start_step": week * 168,
                    "algorithm": algorithm,
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

    result = annual_thermal_comparisons(
        pd.DataFrame(rows), samples=100, seed=1)
    static_p95 = result[
        (result.baseline == "static_robust_mpc")
        & (result.metric == "p95_t_inlet_c")
    ].iloc[0]

    assert np.isclose(static_p95.mean_improvement, 0.2)
