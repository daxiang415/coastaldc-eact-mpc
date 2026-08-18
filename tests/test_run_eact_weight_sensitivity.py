from scripts.run_eact_weight_sensitivity import (
    WEIGHT_SETTINGS,
    build_jobs,
    build_parser,
)


def _argument(job, name):
    return job.command[job.command.index(name) + 1]


def test_weight_sensitivity_is_one_factor_at_a_time_and_paired(tmp_path):
    args = build_parser().parse_args(["--out-root", str(tmp_path)])

    jobs = build_jobs(args)

    assert len(jobs) == 3 * 2 * len(WEIGHT_SETTINGS)
    assert len({job.name for job in jobs}) == len(jobs)
    observed = {
        (
            float(_argument(job, "--weight-grid")),
            float(_argument(job, "--weight-co2")),
            float(_argument(job, "--weight-total")),
            float(_argument(job, "--weight-smooth")),
        )
        for job in jobs
    }
    expected = {
        (setting.grid, setting.co2, setting.total, setting.smooth)
        for setting in WEIGHT_SETTINGS
    }
    assert observed == expected
    for job in jobs:
        assert job.command[
            job.command.index("--controllers") + 1:
            job.command.index("--data-dir")
        ] == ("static", "eact")
        assert _argument(job, "--constraint-tolerance") == "0.0001"
        assert "--no-thermal-safety-shield" in job.command
