from scripts.fit_causal_forecasts import build_parser


def test_forecast_fit_resume_is_opt_in():
    parser = build_parser()

    assert parser.parse_args([]).resume is False
    assert parser.parse_args([]).out_dir.endswith(
        "causal_forecasts_v3_gated_bias")
    assert parser.parse_args(["--resume"]).resume is True
