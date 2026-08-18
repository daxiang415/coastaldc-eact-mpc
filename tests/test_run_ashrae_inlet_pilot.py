from scripts.run_ashrae_inlet_pilot import (
    COUNTRIES,
    SCENARIOS,
    build_jobs,
    build_parser,
)


def test_ashrae_pilot_freezes_the_72_controller_week_matrix(tmp_path):
    args = build_parser().parse_args(["--out-root", str(tmp_path)])

    jobs = build_jobs(args)

    assert len(jobs) == len(COUNTRIES) * len(SCENARIOS) == 6
    assert len({job.name for job in jobs}) == len(jobs)
    assert len(jobs) * 4 * 3 == 72
    for job in jobs:
        command = job.command
        controllers = command[
            command.index("--controllers") + 1:
            command.index("--data-dir")
        ]
        assert controllers == ("nominal", "static", "eact")
        assert command[command.index("--constraint-tolerance") + 1] == "0.0001"
        assert command[command.index("--adaptive-beta-floor") + 1] == "0.10"
        assert command[command.index("--horizon") + 1] == "24"
        assert command[command.index("--maxiter") + 1] == "20"
        assert "--no-thermal-safety-shield" in command
        assert "--no-oracle-workload-projection" in command
    no_shift = next(job for job in jobs if job.stage == "no_shift")
    shifted = next(job for job in jobs if job.stage == "adverse_bias_1sigma")
    assert no_shift.command[
        no_shift.command.index("--forecast-stress-scale") + 1] == "0.0"
    assert shifted.command[
        shifted.command.index("--forecast-stress-scale") + 1] == "1.0"
