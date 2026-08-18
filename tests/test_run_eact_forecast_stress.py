from scripts.run_eact_forecast_stress import build_jobs, build_parser


def test_default_stress_matrix_is_mpc_only_and_paired(tmp_path):
    args = build_parser().parse_args(["--out-root", str(tmp_path)])

    jobs = build_jobs(args)

    assert len(jobs) == 12
    assert {job.stage for job in jobs} == {"forecast_stress"}
    assert all("sac" not in job.command for job in jobs)
    assert all("--no-thermal-safety-shield" in job.command for job in jobs)
    assert all("--adaptive-beta-floor" in job.command for job in jobs)
    assert all("--weight-grid" in job.command for job in jobs)
    assert all("--weight-co2" in job.command for job in jobs)
    assert all("--weight-total" in job.command for job in jobs)
    assert all("--weight-smooth" in job.command for job in jobs)
    assert all(
        job.command[job.command.index("--controllers") + 1:
                    job.command.index("--data-dir")]
        == ("nominal", "static", "eact")
        for job in jobs
    )
    baseline = next(job for job in jobs if "stress_none_" in job.name)
    assert baseline.command[
        baseline.command.index("--forecast-stress-scale") + 1] == "0.0"
    assert "s0p00" in baseline.out_dir
    assert "b0p10" in baseline.out_dir
    assert "wg1p00_wc2p00_wt0p20_ws0p50" in baseline.out_dir


def test_weight_settings_have_distinct_job_identities(tmp_path):
    parser = build_parser()
    base = parser.parse_args([
        "--out-root", str(tmp_path),
        "--scenarios", "none",
    ])
    changed = parser.parse_args([
        "--out-root", str(tmp_path),
        "--scenarios", "none",
        "--weight-co2", "4.0",
    ])

    base_job = build_jobs(base)[0]
    changed_job = build_jobs(changed)[0]

    assert base_job.name != changed_job.name
    assert base_job.out_dir != changed_job.out_dir
    assert "wc4p00" in changed_job.out_dir


def test_stress_runner_can_schedule_eact_only(tmp_path):
    args = build_parser().parse_args([
        "--out-root", str(tmp_path),
        "--controllers", "eact",
        "--scenarios", "adverse_bias",
    ])

    jobs = build_jobs(args)

    assert len(jobs) == 3
    assert all(
        job.command[job.command.index("--controllers") + 1:
                    job.command.index("--data-dir")] == ("eact",)
        for job in jobs
    )
    assert all("ceact" in job.out_dir for job in jobs)
