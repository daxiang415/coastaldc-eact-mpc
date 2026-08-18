from coastaldc_env import COUNTRIES
from scripts.run_eact_beta_ablation import build_jobs, build_parser


def test_beta_ablation_changes_only_adaptive_floor_treatment(tmp_path):
    args = build_parser().parse_args(["--out-root", str(tmp_path)])

    jobs = build_jobs(args)

    assert len(jobs) == len(COUNTRIES)
    for job in jobs:
        assert job.command[
            job.command.index("--controllers") + 1:
            job.command.index("--data-dir")
        ] == ("eact",)
        assert job.command[job.command.index("--adaptive-beta-floor") + 1] == "0.0"
        assert job.command[job.command.index("--forecast-stress") + 1] == "adverse_bias"
        assert job.command[job.command.index("--forecast-stress-scale") + 1] == "1.0"
        assert job.command[
            job.command.index("--constraint-tolerance") + 1
        ] == "0.0001"
        assert "--no-thermal-safety-shield" in job.command
