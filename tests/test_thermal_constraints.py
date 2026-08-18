"""Thermal model sanity: cooling lowers temperature, violations detected, COP behaviour."""

import numpy as np

from coastaldc_env.swhp_cooling import CoolingConfig, SWHPCoolingModel


def test_aggressive_cooling_beats_no_cooling():
    cfg = CoolingConfig(enforce_thermal_safety=False)
    hot = SWHPCoolingModel(cfg)
    cold = SWHPCoolingModel(cfg)
    for _ in range(48):
        hot.step(a_setpoint=1.0, a_pump=0.0, q_it_mwh=8.0, t_sea=15.0)
        cold.step(a_setpoint=-1.0, a_pump=1.0, q_it_mwh=8.0, t_sea=15.0)
    assert cold.t_room < hot.t_room


def test_thermal_violation_flag():
    m = SWHPCoolingModel(CoolingConfig(enforce_thermal_safety=False))
    out = None
    for _ in range(200):  # no pump, setpoint up -> room heats past the limit
        out = m.step(a_setpoint=1.0, a_pump=0.0, q_it_mwh=10.0, t_sea=28.0)
    assert out["thermal_violation_k"] > 0
    assert out["t_inlet"] > m.cfg.t_inlet_allowable_max_c


def test_cop_monotonic_in_sea_temperature():
    m = SWHPCoolingModel(CoolingConfig())
    assert m.cop(t_sea=8.0) > m.cop(t_sea=25.0)


def test_cop_bounds():
    m = SWHPCoolingModel(CoolingConfig())
    for t_sea in [-2, 5, 15, 30, 45]:
        assert m.cfg.cop_min <= m.cop(t_sea) <= m.cfg.cop_max


def test_pump_power_cubic():
    m = SWHPCoolingModel(CoolingConfig())
    out_lo = m.step(0.0, 0.0, q_it_mwh=5.0, t_sea=15.0)
    m.reset()
    out_hi = m.step(0.0, 1.0, q_it_mwh=5.0, t_sea=15.0)
    assert out_hi["e_pump_mwh"] > out_lo["e_pump_mwh"]
    np.testing.assert_allclose(out_hi["e_pump_mwh"], m.cfg.pump_rated_mw, rtol=1e-6)


def test_env_thermal_safety_reachable(env):
    """A sane controller (cool hard) should avoid thermal violations for a whole episode."""
    env.reset(seed=3, options={"start_hour": 0})
    violations = 0
    for _ in range(env.episode_hours):
        _, _, _, trunc, info = env.step(np.array([0.0, -0.5, 0.8], dtype=np.float32))
        violations += info["thermal_violation_k"] > 0
        if trunc:
            break
    assert violations == 0


def test_safety_projection_prevents_feasible_violation():
    m = SWHPCoolingModel(CoolingConfig(enforce_thermal_safety=True))
    m.t_set = 24.0
    m.t_inlet = 29.0

    out = m.step(
        a_setpoint=1.0, a_pump=0.0, q_it_mwh=18.0, t_sea=28.0,
        q_it_next_max=10.0)

    assert out["safety_intervened"]
    assert out["safety_feasible"]
    assert out["thermal_violation_k"] == 0.0
    assert out["t_inlet"] <= m.cfg.t_inlet_allowable_max_c
    assert out["applied_a_pump"] > 0.0


def test_safety_projection_preserves_next_step_recoverability():
    m = SWHPCoolingModel(CoolingConfig(enforce_thermal_safety=True))
    m.t_set = 24.0
    m.t_inlet = 28.0

    first = m.step(
        a_setpoint=1.0, a_pump=0.0, q_it_mwh=10.0, t_sea=28.0,
        q_it_next_max=10.0)
    second = m.step(
        a_setpoint=-1.0, a_pump=1.0, q_it_mwh=10.0, t_sea=28.0,
        q_it_next_max=10.0)

    assert first["safety_feasible"]
    assert first["thermal_violation_k"] == 0.0
    assert second["safety_feasible"]
    assert second["thermal_violation_k"] == 0.0


def test_recommended_exceedance_starts_below_allowable_limit():
    m = SWHPCoolingModel(CoolingConfig(enforce_thermal_safety=False))
    m.t_set = m.cfg.t_set_max
    m.t_inlet = m.cfg.t_inlet_recommended_max_c + 0.2

    out = m.step(a_setpoint=0.0, a_pump=0.0, q_it_mwh=5.0, t_sea=15.0)

    assert out["recommended_exceedance_c"] > 0.0
    assert out["allowable_exceedance_c"] == 0.0
    assert out["thermal_risk_k"] > 0.0
    assert out["thermal_violation_k"] == 0.0


def test_safety_projection_reports_infeasible_state():
    m = SWHPCoolingModel(CoolingConfig(enforce_thermal_safety=True))
    m.t_set = 27.0
    m.t_inlet = 50.0

    out = m.step(a_setpoint=1.0, a_pump=0.0, q_it_mwh=10.0, t_sea=28.0)

    assert out["safety_intervened"]
    assert not out["safety_feasible"]
    assert out["thermal_violation_k"] > 0.0


def test_ashrae_a1_thresholds_and_legacy_aliases_are_consistent():
    cfg = CoolingConfig()
    model = SWHPCoolingModel(cfg)

    assert cfg.t_inlet_recommended_min_c == 18.0
    assert cfg.t_inlet_recommended_max_c == 27.0
    assert cfg.t_inlet_allowable_min_c == 15.0
    assert cfg.t_inlet_allowable_max_c == 32.0
    assert cfg.t_room_soft_max == cfg.t_inlet_recommended_max_c
    assert cfg.t_room_max == cfg.t_inlet_allowable_max_c
    assert cfg.thermal_safety_margin_k == 0.0

    model.t_room = 26.5
    assert model.t_inlet == 26.5
    assert model.t_room == model.t_inlet
