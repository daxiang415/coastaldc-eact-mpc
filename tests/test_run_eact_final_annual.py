from scripts.run_eact_final_annual import build_jobs, build_parser


def test_final_annual_runner_uses_nine_independent_unreset_trajectories(tmp_path):
    args = build_parser().parse_args(["--out-root", str(tmp_path)])

    jobs = build_jobs(args)

    assert len(jobs) == 9
    assert len({job.tag for job in jobs}) == 9
    assert len({job.out_dir for job in jobs}) == 9
    assert {
        job.command[job.command.index("--countries") + 1]
        for job in jobs
    } == {"JPN", "CHN", "NOR"}
    assert {
        job.command[job.command.index("--controllers") + 1]
        for job in jobs
    } == {"nominal", "static", "eact"}
    for job in jobs:
        assert job.command[
            job.command.index("--controllers") + 1:
            job.command.index("--data-dir")
        ] in (("nominal",), ("static",), ("eact",))
        assert "--continuous-year" in job.command
        assert "--thermal-safety-shield" in job.command
        assert "--no-oracle-workload-projection" in job.command
        assert job.command[job.command.index("--adaptive-beta-floor") + 1] == "0.10"
        assert job.command[
            job.command.index("--constraint-tolerance") + 1
        ] == "0.0001"
