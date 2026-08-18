"""Training script CLI defaults."""

from scripts import train_online_rl


def test_progress_bar_is_opt_in_by_default():
    args = train_online_rl.build_parser().parse_args([])

    assert args.progress_bar is False


def test_progress_bar_can_be_enabled_explicitly():
    args = train_online_rl.build_parser().parse_args(["--progress-bar"])

    assert args.progress_bar is True


def test_device_can_be_set_to_cuda():
    args = train_online_rl.build_parser().parse_args(["--device", "cuda"])

    assert args.device == "cuda"


def test_checkpointing_is_disabled_by_default():
    args = train_online_rl.build_parser().parse_args([])

    assert args.checkpoint_freq == 0


def test_checkpointing_and_log_interval_can_be_configured():
    args = train_online_rl.build_parser().parse_args(
        ["--checkpoint-freq", "10000", "--log-interval", "5"]
    )

    assert args.checkpoint_freq == 10000
    assert args.log_interval == 5


def test_safe_v1_is_the_default_output_family():
    assert train_online_rl.SAFE_MODEL_DIR.endswith("models_multiyear_v1")
    assert train_online_rl.SAFE_CHECKPOINT_DIR.endswith("checkpoints_multiyear_v1")


def test_multiyear_training_defaults_to_combined_years_without_validation():
    args = train_online_rl.build_parser().parse_args([])

    assert args.data_dir.endswith("processed_multiyear\\train_2023_2024")
    assert args.validation_data_dir is None
    assert args.eval_freq == 0
    assert args.oracle_workload_projection is False
    assert args.oracle_forecast_observations is False
    assert args.sla_penalty_weight == 50.0
