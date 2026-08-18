"""Flexible workload must be conserved: arrived == executed + backlog + violated."""

import numpy as np

from coastaldc_env.workload import WorkloadConfig, WorkloadModel


def test_conservation_random_actions():
    rng = np.random.default_rng(0)
    wl = WorkloadModel(WorkloadConfig())
    for _ in range(500):
        wl.step(rng.uniform(-1, 1), fixed_load_mw=rng.uniform(3, 7),
                flexible_arrival_mw=rng.uniform(0, 3))
        assert abs(wl.conservation_error()) < 1e-6


def test_conservation_in_env(env):
    rng = np.random.default_rng(1)
    env.reset(seed=1)
    for _ in range(env.episode_hours):
        a = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 1)],
                     dtype=np.float32)
        env.step(a)
    assert abs(env.workload.conservation_error()) < 1e-6


def test_backlog_nonnegative():
    rng = np.random.default_rng(2)
    wl = WorkloadModel()
    for _ in range(300):
        wl.step(rng.uniform(-1, 1), 5.0, rng.uniform(0, 3))
        assert wl.backlog_mwh >= -1e-9
        assert np.all(wl.queue >= -1e-9)


def test_full_recovery_empties_backlog():
    wl = WorkloadModel(WorkloadConfig(it_capacity_mw=100.0))
    wl.step(-1.0, 5.0, 2.0)          # defer everything
    assert wl.backlog_mwh > 1.9
    wl.step(1.0, 5.0, 0.0)           # huge capacity: recover all
    assert wl.backlog_mwh < 1e-9


def test_deferral_beyond_window_counts_as_violation():
    cfg = WorkloadConfig(max_delay_hours=4, enforce_deadlines=False)
    wl = WorkloadModel(cfg)
    wl.step(-1.0, 5.0, 1.0)          # defer 1 MWh
    violated_total = 0.0
    for _ in range(cfg.max_delay_hours + 1):
        _, v, _ = wl.step(-1.0, 5.0, 0.0)
        violated_total += v
    assert violated_total > 0.99     # the deferred MWh expired


def test_deadline_enforcement_recovers_expiring_work_when_capacity_exists():
    cfg = WorkloadConfig(it_capacity_mw=10.0, max_delay_hours=2,
                         enforce_deadlines=True)
    wl = WorkloadModel(cfg)
    wl.step(-1.0, fixed_load_mw=5.0, flexible_arrival_mw=2.0)
    wl.step(-1.0, fixed_load_mw=5.0, flexible_arrival_mw=0.0)

    _, violated, executed_flexible = wl.step(
        -1.0, fixed_load_mw=5.0, flexible_arrival_mw=0.0)

    assert violated == 0.0
    assert executed_flexible == 2.0
    assert wl.backlog_mwh == 0.0


def test_terminal_settlement_converts_backlog_to_unserved_work():
    wl = WorkloadModel(WorkloadConfig(it_capacity_mw=10.0))
    wl.step(-1.0, fixed_load_mw=5.0, flexible_arrival_mw=2.0)

    unserved = wl.settle_terminal_backlog()

    assert unserved == 2.0
    assert wl.backlog_mwh == 0.0
    assert wl.total_violated == 2.0
    assert abs(wl.conservation_error()) < 1e-9


def test_terminal_settlement_ignores_numerical_queue_residue():
    wl = WorkloadModel(WorkloadConfig(it_capacity_mw=10.0))
    wl.queue[0] = 5e-10
    wl.total_arrived = 5e-10

    unserved = wl.settle_terminal_backlog()

    assert unserved == 0.0
    assert wl.backlog_mwh == 0.0
    assert wl.total_violated == 0.0
    assert abs(wl.conservation_error()) <= 1e-9


def test_env_terminal_projection_prevents_controllable_backlog():
    from coastaldc_env import CoastalDCContinuousEnv

    short_env = CoastalDCContinuousEnv(country="JPN", episode_hours=2, seed=0)
    short_env.reset(seed=0, options={"start_hour": 0})
    short_env.step(np.array([-1.0, 0.0, 0.5], dtype=np.float32))
    _, _, _, truncated, info = short_env.step(
        np.array([-1.0, 0.0, 0.5], dtype=np.float32))

    assert truncated
    assert info["terminal_unserved_mwh"] == 0.0
    assert info["reward_terms"]["sla"] == 0.0
    assert short_env.workload.backlog_mwh == 0.0
    assert info["episode_metrics"]["terminal_unserved_mwh"] == 0.0
    assert info["episode_metrics"]["workload_interventions"] >= 1
    assert info["workload_intervened"]
    assert info["workload_feasible"]
    assert abs(short_env.workload.conservation_error()) < 1e-6


def test_environment_can_disable_oracle_workload_projection(monkeypatch):
    from coastaldc_env import CoastalDCContinuousEnv

    causal_env = CoastalDCContinuousEnv(
        country="JPN", episode_hours=2,
        use_oracle_workload_projection=False, seed=0)
    causal_env.reset(seed=0, options={"start_hour": 0})

    def reject_future_access():
        raise AssertionError("Causal evaluation must not inspect future workload")

    monkeypatch.setattr(causal_env, "_future_workload_spare_profile", reject_future_access)
    _, _, _, _, info = causal_env.step(np.array([-1.0, -1.0, 1.0]))

    assert not info["oracle_workload_projection_enabled"]
    assert not info["workload_intervened"]


def test_terminal_projection_forces_recovery_and_blocks_deferral():
    wl = WorkloadModel(WorkloadConfig(max_delay_hours=4))
    wl.step(-1.0, fixed_load_mw=5.0, flexible_arrival_mw=2.0)

    applied, intervened = wl.project_terminal_action(-0.5, remaining_hours=4)

    assert applied == 1.0
    assert intervened

    empty = WorkloadModel(WorkloadConfig(max_delay_hours=4))
    applied, intervened = empty.project_terminal_action(-0.5, remaining_hours=4)
    assert applied == -0.5
    assert not intervened


def test_recoverability_projection_uses_hourly_future_capacity():
    wl = WorkloadModel(WorkloadConfig(
        it_capacity_mw=10.0, max_delay_hours=4))

    applied, intervened, feasible = wl.project_recoverable_action(
        -1.0, fixed_load_mw=8.0, flexible_arrival_mw=2.0,
        future_spare_mwh=np.array([0.0, 2.0, 0.0, 0.0]))

    assert applied == -1.0
    assert not intervened
    assert feasible


def test_recoverability_projection_blocks_unrecoverable_deferral():
    wl = WorkloadModel(WorkloadConfig(
        it_capacity_mw=10.0, max_delay_hours=4))

    applied, intervened, feasible = wl.project_recoverable_action(
        -1.0, fixed_load_mw=8.0, flexible_arrival_mw=2.0,
        future_spare_mwh=np.zeros(4))

    assert abs(applied) < 1e-4
    assert intervened
    assert feasible


def test_recoverability_projection_respects_oldest_deadline():
    wl = WorkloadModel(WorkloadConfig(
        it_capacity_mw=10.0, max_delay_hours=4))
    wl.queue[-2] = 1.0

    applied, intervened, feasible = wl.project_recoverable_action(
        0.0, fixed_load_mw=10.0, flexible_arrival_mw=0.0,
        future_spare_mwh=np.array([0.0, 1.0, 0.0, 0.0]))

    assert applied == 1.0
    assert intervened
    assert not feasible
