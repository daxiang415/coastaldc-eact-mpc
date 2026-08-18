"""Reward function: sign, weighting, smoothness term, and energy accounting."""

import numpy as np

from coastaldc_env.offshore_wind import match_wind
from coastaldc_env.reward import RewardFunction, RewardNormalizers, RewardWeights


def _zero_args():
    return dict(e_grid_mwh=0.0, co2_kg=0.0, sla_violation_mwh=0.0,
                thermal_violation_k=0.0, unused_wind_mwh=0.0,
                action=np.zeros(3), prev_action=np.zeros(3),
                e_total_mwh=0.0, pump_energy_mwh=0.0)


def test_perfect_step_zero_reward():
    r, terms = RewardFunction()(**_zero_args())
    assert r == 0.0
    assert all(v == 0.0 for v in terms.values())


def test_reward_nonpositive_and_decreasing():
    fn = RewardFunction(RewardWeights(w_thermal=1.0))
    args = _zero_args()
    r0, _ = fn(**args)
    args["co2_kg"] = 5000.0
    r1, _ = fn(**args)
    args["thermal_violation_k"] = 2.0
    r2, _ = fn(**args)
    assert r0 == 0.0 and r1 < r0 and r2 < r1


def test_weights_applied():
    w = RewardWeights(w_co2=2.0, w_grid=1.0)
    n = RewardNormalizers(co2_scale=1.0, grid_scale=1.0)
    fn = RewardFunction(w, n)
    args = _zero_args()
    args["co2_kg"] = 1.0
    _, terms_co2 = fn(**args)
    args = _zero_args()
    args["e_grid_mwh"] = 1.0
    _, terms_grid = fn(**args)
    assert terms_co2["co2"] == 2.0 * terms_grid["grid"]


def test_reward_uses_grid_purchase_not_electricity_price():
    fn = RewardFunction()
    args = _zero_args()
    args["e_grid_mwh"] = 4.0
    r, terms = fn(**args)
    assert "grid" in terms
    assert "cost" not in terms
    assert terms["grid"] > 0
    assert r < 0


def test_default_reward_penalizes_total_energy_not_pump_separately():
    fn = RewardFunction()
    args = _zero_args()
    args["e_total_mwh"] = 10.0
    args["pump_energy_mwh"] = 0.4

    reward, terms = fn(**args)

    assert terms["total"] == 0.2
    assert terms["pump"] == 0.0
    assert reward == -0.2


def test_smoothness_penalty():
    fn = RewardFunction()
    args = _zero_args()
    args["action"] = np.array([1.0, 1.0, 1.0])
    args["prev_action"] = np.array([-1.0, -1.0, 0.0])
    r, terms = fn(**args)
    assert terms["smooth"] > 0 and r < 0


def test_soft_thermal_and_safety_override_penalties():
    fn = RewardFunction(RewardWeights(
        w_thermal_risk=1.0, w_safety_override=1.0))
    args = _zero_args()
    args["thermal_risk_k"] = 0.5
    args["safety_override"] = 0.25

    reward, terms = fn(**args)

    assert terms["thermal_risk"] > 0.0
    assert terms["safety_override"] > 0.0
    assert reward < 0.0


def test_default_reward_keeps_hard_constraints_out_of_objective():
    fn = RewardFunction()
    args = _zero_args()
    args.update(
        sla_violation_mwh=1.0,
        thermal_violation_k=1.0,
        thermal_risk_k=1.0,
        safety_override=1.0,
        unused_wind_mwh=5.0,
    )

    reward, terms = fn(**args)

    assert reward == 0.0
    for key in ("sla", "thermal", "thermal_risk", "safety_override",
                "unused_wind"):
        assert terms[key] == 0.0


def test_wind_matching_accounting():
    out = match_wind(10.0, 4.0, carbon_intensity_kg_per_mwh=500.0, price_usd_per_mwh=100.0)
    assert out["e_wind_used_mwh"] == 4.0
    assert out["e_grid_mwh"] == 6.0
    assert out["e_wind_unused_mwh"] == 0.0
    assert out["co2_kg"] == 3000.0
    assert out["cost_usd"] == 600.0

    out2 = match_wind(3.0, 10.0, 500.0, 100.0)
    assert out2["e_grid_mwh"] == 0.0
    assert out2["e_wind_unused_mwh"] == 7.0
    assert out2["co2_kg"] == 0.0


def test_episode_metrics_consistency(env):
    env.reset(seed=4, options={"start_hour": 100})
    for _ in range(env.episode_hours):
        *_, trunc, info = env.step(np.array([0.0, 0.0, 0.5], dtype=np.float32))
        if trunc:
            break
    m = env.episode_summary()
    # grid + wind_used must equal total electricity
    np.testing.assert_allclose(m["e_grid_mwh"] + m["e_wind_used_mwh"],
                               m["e_total_mwh"], rtol=1e-6)
    assert 0.0 <= m["wind_utilization_pct"] <= 100.0
    assert m["mean_thermal_safety_override"] >= 0.0
    assert m["mean_rate_limit_override"] >= 0.0


def test_environment_separates_rate_limit_and_thermal_safety_overrides(env):
    env.reset(seed=4, options={"start_hour": 100})
    action = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    *_, info = env.step(action)

    unshielded = np.asarray(info["unshielded_action"])
    applied = np.asarray(info["applied_action"])
    expected_safety = np.mean(np.abs(unshielded[1:] - applied[1:]))
    expected_rate_limit = np.mean(np.abs(action[1:] - unshielded[1:]))
    np.testing.assert_allclose(info["thermal_safety_override"], expected_safety)
    np.testing.assert_allclose(info["rate_limit_override"], expected_rate_limit)
