import numpy as np
import pandas as pd

from coastaldc_env import COUNTRIES
from scripts.analyze_ashrae_inlet_pilot import START_HOURS
from scripts.analyze_ashrae_inlet_seasonal import no_shift_noninferiority


def test_no_shift_noninferiority_uses_one_sided_upper_bound():
    rows = []
    for country in COUNTRIES:
        for start_hour in START_HOURS:
            rows.extend([
                {
                    "country": country,
                    "start_hour": start_hour,
                    "forecast_stress": "none",
                    "algorithm": "static_robust_mpc",
                    "episode_return": -100.0,
                },
                {
                    "country": country,
                    "start_hour": start_hour,
                    "forecast_stress": "none",
                    "algorithm": "eact_mpc",
                    "episode_return": -100.5,
                },
            ])

    result = no_shift_noninferiority(
        pd.DataFrame(rows),
        samples=100,
        margin_pct=1.0,
        seed=1,
    ).iloc[0]

    assert np.isclose(result.mean_cost_increase_pct, 0.5)
    assert np.isclose(result.one_sided_95_upper_pct, 0.5)
    assert bool(result.noninferiority_passed)
