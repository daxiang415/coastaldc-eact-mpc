"""Controller selection for paper-facing evaluation."""

import inspect
from types import SimpleNamespace

import torch  # noqa: F401

from coastaldc_env import CoastalDCContinuousEnv
from scripts import evaluate_15countries
from scripts.evaluate_15countries import (
    build_controllers,
    build_paired_comparisons,
    episode_result_row,
)

import numpy as np
import pandas as pd


def _args(**overrides):
    values = dict(
        mpc=False,
        im_mpc=False,
        mpc_horizon=24,
        mpc_replan_every=4,
        mpc_maxiter=10,
        mpc_block_hours=4,
        mpc_gamma=0.995,
        discrete_ablation=False,
        dt_target_return=-50.0,
        seeds=[0],
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_evaluation_uses_continuous_controllers_only(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_15countries, "MODEL_DIR", str(tmp_path))
    env = CoastalDCContinuousEnv(country="JPN", seed=0)

    names = [ctrl.name for ctrl in build_controllers(env, _args(), "JPN")]

    assert names == ["no_control", "rule_based"]


def test_discrete_ablation_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_15countries, "MODEL_DIR", str(tmp_path))
    env = CoastalDCContinuousEnv(country="JPN", seed=0)

    names = [
        ctrl.name
        for ctrl in build_controllers(env, _args(discrete_ablation=True), "JPN")
    ]

    assert "discrete_rule_based" in names


def test_evaluation_imports_torch_before_project_modules_on_windows_gpu():
    source = inspect.getsource(evaluate_15countries)

    torch_index = source.index("import torch")
    assert torch_index < source.index("import numpy")
    assert torch_index < source.index("import pandas")
    assert torch_index < source.index("from coastaldc_env")


def test_model_observation_shape_must_match_environment():
    env = CoastalDCContinuousEnv(country="JPN", seed=0)
    compatible = SimpleNamespace(model=SimpleNamespace(
        observation_space=SimpleNamespace(shape=(29,))))
    legacy = SimpleNamespace(model=SimpleNamespace(
        observation_space=SimpleNamespace(shape=(28,))))

    assert evaluate_15countries.model_matches_env(compatible, env)
    assert not evaluate_15countries.model_matches_env(legacy, env)


def test_evaluation_defaults_to_held_out_2025_and_multiyear_models():
    assert evaluate_15countries.MODEL_DIR.endswith("models_multiyear_v1")
    assert evaluate_15countries.TEST_DATA_DIR.endswith(
        "processed_multiyear\\2025")


def test_mpc_speed_controls_are_forwarded():
    env = CoastalDCContinuousEnv(country="JPN", seed=0)

    controllers = build_controllers(
        env,
        _args(mpc=True, mpc_replan_every=6, mpc_maxiter=7,
              mpc_block_hours=3, mpc_gamma=0.97),
        "JPN",
    )
    mpc = next(ctrl for ctrl in controllers if ctrl.name == "oracle_mpc")

    assert mpc.replan_every == 6
    assert mpc.maxiter == 7
    assert mpc.control_block_hours == 3
    assert mpc.gamma == 0.97


def test_information_matched_mpc_is_opt_in_and_uses_shared_settings(
        monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate_15countries, "MODEL_DIR", str(tmp_path))
    env = CoastalDCContinuousEnv(country="JPN", seed=0)

    controllers = build_controllers(
        env,
        _args(im_mpc=True, mpc_replan_every=6, mpc_maxiter=7,
              mpc_block_hours=3, mpc_gamma=0.97),
        "JPN",
    )
    im_mpc = next(ctrl for ctrl in controllers if ctrl.name == "im_mpc")

    assert [ctrl.name for ctrl in controllers] == [
        "no_control", "rule_based", "im_mpc"]
    assert im_mpc.replan_every == 6
    assert im_mpc.maxiter == 7
    assert im_mpc.control_block_hours == 3
    assert im_mpc.gamma == 0.97


def test_paired_comparisons_include_information_matched_mpc():
    rows = []
    for episode, start_hour in enumerate([100, 200, 300]):
        common = dict(country="JPN", episode=episode,
                      reset_seed=5000 + episode, start_hour=start_hour,
                      e_grid_mwh=10.0 + episode)
        rows.append({**common, "algorithm": "im_mpc"})
        rows.append({**common, "algorithm": "sac_seed0",
                     "e_grid_mwh": 9.0 + episode})

    paired = build_paired_comparisons(
        pd.DataFrame(rows), bootstrap_samples=200)

    assert set(paired["baseline"]) == {"im_mpc"}


def test_episode_result_row_records_pairing_and_action_overrides():
    requested = np.array([[0.0, 0.5, 0.2], [1.0, -0.5, 0.4]])
    applied = np.array([[0.0, 0.5, 0.2], [1.0, -1.0, 0.8]])
    traj = {
        "actions": requested,
        "applied_actions": applied,
        "episode_return": -10.0,
        "metrics": {"e_grid_mwh": 4.0},
        "reward_terms": {"grid": 2.0},
        "reset_info": {"data_hour": 123},
    }

    row = episode_result_row("JPN", "sac_seed0", 2, 5002, traj)

    assert row["start_hour"] == 123
    assert row["reset_seed"] == 5002
    assert row["safety_override_fraction"] == 0.5
    assert row["reward_grid"] == 2.0


def test_paired_comparisons_orient_lower_metrics_as_improvements():
    rows = []
    for episode, start_hour in enumerate([100, 200, 300]):
        common = dict(country="JPN", episode=episode,
                      reset_seed=5000 + episode, start_hour=start_hour)
        rows.append({**common, "algorithm": "rule_based",
                     "e_grid_mwh": 10.0 + episode,
                     "episode_return": -10.0 - episode})
        rows.append({**common, "algorithm": "sac_seed0",
                     "e_grid_mwh": 9.0 + episode,
                     "episode_return": -9.0 - episode})

    paired = build_paired_comparisons(
        pd.DataFrame(rows), bootstrap_samples=200)
    grid = paired[(paired["baseline"] == "rule_based")
                  & (paired["metric"] == "e_grid_mwh")].iloc[0]
    episode_return = paired[(paired["baseline"] == "rule_based")
                            & (paired["metric"] == "episode_return")].iloc[0]

    assert grid["mean_difference_sac_minus_baseline"] == -1.0
    assert grid["mean_improvement"] == 1.0
    assert episode_return["mean_improvement"] == 1.0


def test_paired_comparisons_reject_mismatched_episode_starts():
    rows = [
        {"country": "JPN", "algorithm": "rule_based", "episode": 0,
         "reset_seed": 5000, "start_hour": 10, "e_grid_mwh": 10.0},
        {"country": "JPN", "algorithm": "sac_seed0", "episode": 0,
         "reset_seed": 5000, "start_hour": 11, "e_grid_mwh": 9.0},
    ]

    with np.testing.assert_raises_regex(ValueError, "Unpaired episode starts"):
        build_paired_comparisons(pd.DataFrame(rows), bootstrap_samples=20)
