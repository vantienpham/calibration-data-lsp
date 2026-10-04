"""The plans define exactly the published runs, with their recorded configurations."""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import compress  # noqa: E402
import plans  # noqa: E402

# Settings that have no effect when zero-shot evaluation is disabled (--no-zs).
ZERO_SHOT_ONLY = {"zs_batch", "zs_tasks", "zs_limit"}


def paper_jobs():
    return [job for name in plans.PAPER for job in plans.PLANS[name]()]


def test_job_names_and_outputs_are_unique():
    jobs = paper_jobs() + plans.plan_smoke()
    names = [j["name"] for j in jobs]
    targets = [j.get("run_dir") or j["output"] for j in jobs]
    assert len(names) == len(set(names))
    assert len(targets) == len(set(targets))


def test_every_plan_lists_known_scripts():
    for job in paper_jobs():
        assert os.path.exists(os.path.join(ROOT, job["script"])), job["name"]


def test_records_are_exactly_the_planned_runs():
    planned = {j["run_dir"] for j in paper_jobs() if j.get("run_dir")}
    recorded = {"out/runs/" + os.path.relpath(os.path.dirname(p), os.path.join(ROOT, "records", "runs"))
                for p in glob.glob(os.path.join(ROOT, "records", "runs", "*", "*", "config.json"))}
    assert planned == recorded


def test_recorded_configurations_match_the_plans():
    defaults = vars(compress.parse(["--model", "m", "--run-dir", "d"]))
    for job in paper_jobs():
        if not job.get("run_dir"):
            continue
        record = os.path.join(ROOT, "records", "runs", *job["run_dir"].split("/")[2:])
        with open(os.path.join(record, "config.json")) as f:
            cfg = json.load(f)
        want = vars(compress.parse(job["args"] + [f"--run-dir={job['run_dir']}"]))
        for key, value in want.items():
            if want["no_zs"] and key in ZERO_SHOT_ONLY:
                continue
            # options added to compress.py after a run was made are at their defaults
            assert cfg.get(key, defaults[key]) == value, (job["name"], key, cfg.get(key), value)


def test_records_are_finished():
    for path in glob.glob(os.path.join(ROOT, "records", "runs", "*", "*", "metrics.json")):
        with open(path) as f:
            assert json.load(f).get("done"), path
