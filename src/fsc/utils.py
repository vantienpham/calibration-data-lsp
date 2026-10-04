"""Run directories, provenance and small I/O helpers."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time

import torch


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def git_stamp() -> dict:
    """Commit of the code: a .git_commit stamp if the tree was copied without .git
    (lines ``key=value``), else ``git rev-parse`` of the checkout."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stamp = os.path.join(root, ".git_commit")
    if os.path.exists(stamp):
        with open(stamp) as f:
            return dict(line.strip().split("=", 1) for line in f if "=" in line)
    try:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip())
        return {"commit": commit, "dirty": str(dirty).lower()}
    except Exception:
        return {"commit": "unknown"}


def env_info() -> dict:
    import transformers
    info = {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "transformers": str(transformers.__version__),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "git": git_stamp(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip().splitlines()[0]
        except Exception:
            pass
    try:
        import lm_eval
        info["lm_eval"] = lm_eval.__version__
    except Exception:
        pass
    return info


def model_tag(name: str) -> str:
    return name.rstrip("/").split("/")[-1].lower()


class Timer:
    def __init__(self):
        self.t = {}

    def __call__(self, key):
        timer = self

        class _Ctx:
            def __enter__(self_):
                self_.t0 = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

            def __exit__(self_, *a):
                timer.t[key] = {"seconds": round(time.time() - self_.t0, 2)}
                if torch.cuda.is_available():
                    timer.t[key]["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

        return _Ctx()
