from coastaldc_env import COUNTRIES
from scripts.run_eact_final_seasonal import build_jobs, build_parser


def test_final_seasonal_matrix_freezes_e1_and_e2_configuration(tmp_path):
    args = build_parser().parse_args(["--out-root", str(tmp_path)])

    jobs = build_jobs(args)

    assert len(jobs) == 2 * len(COUNTRIES)
    assert sum(job.stage == "e1_no_shift" for job in jobs) == len(COUNTRIES)
    assert sum(job.stage == "e2_shift" for job in jobs) == len(COUNTRIES)
    assert len({job.name for job in jobs}) == len(jobs)
    for job in jobs:
        assert job.command[
            job.command.index("--controllers") + 1:
            job.command.index("--data-dir")
        ] == ("nominal", "static", "eact")
        assert "--adaptive-beta-floor" in job.command
        assert job.command[job.command.index("--adaptive-beta-floor") + 1] == "0.10"
        assert job.command[
            job.command.index("--constraint-tolerance") + 1
        ] == "0.0001"
        assert "--no-thermal-safety-shield" in job.command
        assert "--no-oracle-workload-projection" in job.command
    e1 = next(job for job in jobs if job.stage == "e1_no_shift")
    e2 = next(job for job in jobs if job.stage == "e2_shift")
    assert e1.command[e1.command.index("--forecast-stress-scale") + 1] == "0.0"
    assert e2.command[e2.command.index("--forecast-stress-scale") + 1] == "1.0"
