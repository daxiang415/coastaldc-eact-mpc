import json

from scripts.run_eact_paper_experiments import (
    Job,
    build_jobs,
    build_parser,
    execution_priority,
    job_is_complete,
)


def test_full_preregistered_matrix_contains_51_resumable_jobs(tmp_path):
    args = build_parser().parse_args([
        "--out-root", str(tmp_path), "--intervention-weight", "0.0"])

    jobs = build_jobs(args)

    assert len(jobs) == 51
    assert sum(job.stage == "calibration" for job in jobs) == 24
    assert sum(job.stage == "seasonal" for job in jobs) == 15
    assert sum(job.stage == "annual" for job in jobs) == 3
    assert sum(job.stage == "ablation" for job in jobs) == 3
    assert sum(job.stage == "sensitivity" for job in jobs) == 6
    seasonal = next(job for job in jobs if job.name == "seasonal_2025_JPN")
    assert "2025-01-15T00:00:00" in seasonal.command
    assert "2025-10-15T00:00:00" in seasonal.command
    annual = next(job for job in jobs if job.name == "annual_2025_JPN")
    assert "sac" not in annual.command
    ablation = next(job for job in jobs if job.name == "ablation_2025_JPN")
    assert "eact_no_intervention" not in ablation.command

    positive_args = build_parser().parse_args([
        "--out-root", str(tmp_path), "--stages", "ablation",
        "--intervention-weight", "0.1"])
    positive_job = build_jobs(positive_args)[0]
    assert "eact_no_intervention" in positive_job.command
    ordered = sorted(jobs, key=execution_priority)
    assert [job.stage for job in ordered[:3]] == ["annual"] * 3


def test_resume_requires_complete_manifest_and_all_outputs(tmp_path):
    job = Job("test", "job", ("python",), str(tmp_path), "tag")
    outputs = {}
    for name in ("summary", "episodes", "weekly", "hourly", "solver"):
        path = tmp_path / f"{name}.csv"
        path.write_text("x\n", encoding="utf-8")
        outputs[name] = {"path": str(path), "rows": 0}
    (tmp_path / "manifest_tag.json").write_text(json.dumps({
        "status": "complete", "command": list(job.command), "outputs": outputs,
    }), encoding="utf-8")

    assert job_is_complete(job)
    (tmp_path / "manifest_tag.json").write_text(json.dumps({
        "status": "complete", "command": ["different"], "outputs": outputs,
    }), encoding="utf-8")
    assert not job_is_complete(job)
    (tmp_path / "manifest_tag.json").write_text(json.dumps({
        "status": "complete", "command": list(job.command), "outputs": outputs,
    }), encoding="utf-8")
    (tmp_path / "solver.csv").unlink()
    assert not job_is_complete(job)
