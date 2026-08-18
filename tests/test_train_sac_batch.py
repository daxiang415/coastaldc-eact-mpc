"""Batch SAC experiment command construction."""

import sys
from pathlib import Path

from scripts import train_sac_batch


def test_build_jobs_uses_sac_training_defaults(tmp_path):
    args = train_sac_batch.build_parser().parse_args(
        [
            "--countries",
            "JPN",
            "NOR",
            "--seeds",
            "0",
            "1",
            "--timesteps",
            "1680000",
            "--device",
            "cuda",
            "--out-dir",
            str(tmp_path / "models"),
            "--checkpoint-freq",
            "50000",
        ]
    )

    jobs = train_sac_batch.build_jobs(args)

    assert len(jobs) == 4
    first = jobs[0]
    assert first.country == "JPN"
    assert first.seed == 0
    assert first.command[:3] == [
        sys.executable,
        str(Path(train_sac_batch.ROOT) / "scripts" / "train_online_rl.py"),
        "--algo",
    ]
    assert first.command[first.command.index("--algo") + 1] == "sac"
    assert first.command[first.command.index("--device") + 1] == "cuda"
    assert first.command[first.command.index("--checkpoint-freq") + 1] == "50000"
    assert "--no-oracle-workload-projection" in first.command
    assert "--no-oracle-forecast-observations" in first.command
    assert first.command[first.command.index("--sla-penalty-weight") + 1] == "50.0"


def test_build_jobs_skips_existing_models_by_default(tmp_path):
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    (out_dir / "sac_JPN_seed0.zip").write_text("existing", encoding="utf-8")
    args = train_sac_batch.build_parser().parse_args(
        ["--countries", "JPN", "--seeds", "0", "--out-dir", str(out_dir)]
    )

    jobs = train_sac_batch.build_jobs(args)

    assert jobs == []


def test_batch_defaults_do_not_overwrite_legacy_models():
    args = train_sac_batch.build_parser().parse_args([])

    assert args.out_dir.endswith("models_sac_causal_v1")
    assert args.checkpoint_dir.endswith("checkpoints_sac_causal_v1")


def test_batch_defaults_to_three_seeds_and_combined_training_years():
    args = train_sac_batch.build_parser().parse_args([])

    assert args.seeds == [0, 1, 2]
    assert args.timesteps == 1_000_000
    assert args.data_dir.endswith("processed_multiyear\\train_2023_2024")
    assert args.validation_data_dir is None
    assert args.eval_freq == 0
