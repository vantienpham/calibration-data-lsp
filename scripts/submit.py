#!/usr/bin/env python3
"""Run the jobs of one or more plans, on Slurm or on the local GPU.

    uv run --no-sync python scripts/submit.py PLAN [PLAN ...]            # Slurm
    uv run --no-sync python scripts/submit.py PLAN [PLAN ...] --local    # this machine
    uv run --no-sync python scripts/submit.py all --list                 # status of everything
    uv run --no-sync python scripts/submit.py main --only llama-3.2-1b-proj-c4-r50   # one job

Plans are defined in scripts/plans.py; ``all`` is every plan behind the paper, in
dependency order. A job is skipped when it has finished (its run directory holds
a finished metrics.json, or its declared output exists) or when a file it
requires does not exist yet (e.g. a self-generated calibration set still being
sampled). Re-running is always safe; an interrupted compress.py run resumes from
its last completed phase.

Slurm mode submits through slurm/run.slurm and keeps at most --max-jobs of your
jobs in the queue; run it again to top the queue up. Placement comes from the
FSC_PARTITIONS, FSC_PARTITIONS_80G and FSC_EXCLUDE environment variables (see
scripts/plans.py). Local mode runs the jobs one after another in this process's
environment, in plan order, and stops at the first failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plans  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def queued_names() -> set[str]:
    try:
        out = subprocess.check_output(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
                                      text=True)
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def finished(job: dict) -> bool:
    if job.get("output"):
        return os.path.exists(job["output"])
    m = os.path.join(job["run_dir"], "metrics.json")
    if not os.path.exists(m):
        return False
    try:
        with open(m) as f:
            return bool(json.load(f).get("done"))
    except (OSError, json.JSONDecodeError):
        return False


def job_argv(job: dict) -> list[str]:
    """The job's script and arguments, as run inside slurm/run.slurm or locally."""
    argv = [job["script"]] + list(job["args"])
    if job.get("run_dir"):
        argv.append(f"--run-dir={job['run_dir']}")
    return argv


def sbatch_cmd(job: dict) -> list[str]:
    cmd = ["sbatch", f"--job-name={job['name']}", f"--time={job['time']}",
           f"--gres=gpu:{job.get('gpus', 1)}", f"--cpus-per-task={job.get('cpus', 8)}"]
    if job.get("partitions"):
        cmd.append(f"--partition={job['partitions']}")
    exclude = job.get("exclude", plans.EXCLUDE)
    if exclude:
        cmd.append(f"--exclude={exclude}")
    return cmd + ["slurm/run.slurm"] + job_argv(job)


def run_local(job: dict) -> int:
    os.makedirs("logs", exist_ok=True)
    log = os.path.join("logs", f"{job['name']}.out")
    cmd = [sys.executable] + job_argv(job)
    print(f"running {job['name']} (log: {log})", flush=True)
    with open(log, "a") as f:
        f.write("command   : " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plans", nargs="+", help="plan names from scripts/plans.py, or 'all'")
    ap.add_argument("--local", action="store_true", help="run on this machine instead of Slurm")
    ap.add_argument("--max-jobs", type=int, default=12, help="Slurm: jobs to keep queued at most")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    ap.add_argument("--list", action="store_true", help="print every job's status and exit")
    ap.add_argument("--only", default=None, help="keep the jobs whose name contains this text")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    jobs = [job for name in args.plans for job in plans.get(name)]
    if len({j["name"] for j in jobs}) != len(jobs):
        jobs = list({j["name"]: j for j in jobs}.values())
    if args.only:
        jobs = [j for j in jobs if args.only in j["name"]]
    queued = set() if args.local else queued_names()
    room = args.max_jobs - len(queued)
    n_done = n_queued = n_blocked = n_run = 0
    for job in jobs:
        if finished(job):
            n_done += 1
            status = "done"
        elif job["name"] in queued:
            n_queued += 1
            status = "queued"
        elif any(not os.path.exists(r) for r in job.get("requires", [])):
            n_blocked += 1
            missing = next(r for r in job["requires"] if not os.path.exists(r))
            status = f"blocked (needs {missing})"
        elif args.list:
            status = "to run"
            n_run += 1
        elif args.dry_run:
            cmd = [sys.executable] + job_argv(job) if args.local else sbatch_cmd(job)
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            status = "would run"
            n_run += 1
        elif args.local:
            rc = run_local(job)
            if rc != 0:
                print(f"{job['name']} failed (exit {rc}); see logs/{job['name']}.out", file=sys.stderr)
                return rc
            status = "done"
            n_run += 1
        elif room > 0:
            out = subprocess.run(sbatch_cmd(job), capture_output=True, text=True)
            if out.returncode != 0:
                print(f"sbatch failed for {job['name']}: {out.stderr.strip()}", file=sys.stderr)
                return 1
            status = out.stdout.strip()
            room -= 1
            n_run += 1
        else:
            status = "waiting"
        if args.list or not status.startswith(("done", "queued", "waiting")):
            print(f"{job['name']:<55} {status}")
    verb = "run" if args.local else "submitted"
    if args.dry_run or args.list:
        verb = "to run"
    print(f"{len(jobs)} jobs: {n_done} done, {n_queued} queued, {n_blocked} blocked, {n_run} {verb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
