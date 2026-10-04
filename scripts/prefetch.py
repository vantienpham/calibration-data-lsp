#!/usr/bin/env python3
"""Cache every model, corpus and benchmark the experiments read. Needs internet.

    uv run --no-sync python scripts/prefetch.py                       # everything
    uv run --no-sync python scripts/prefetch.py --models opt-125m     # one model

Models and corpora are downloaded at the commits pinned in ``fsc.hub``; benchmarks
are loaded through lm-eval's own task definitions, which caches the files the
harness reads. Afterwards jobs can run with HF_HUB_OFFLINE=1 (slurm/run.slurm sets
it), which is what compute nodes without internet access need.

The Llama models are gated: accept their license on huggingface.co and log in
(``uv run --no-sync huggingface-cli login`` or HF_TOKEN) before prefetching them.
Disk: about 24 GB for the five models (15 GB of it Llama-3.1-8B) and under 1 GB for
the corpora.
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

# Weights in other formats, and the original Meta checkpoints, are not needed.
IGNORE = ["original/*", "*.pth", "*.h5", "*.msgpack", "*.ot", "*.onnx", "flax_model*",
          "tf_model*", "rust_model*", "*.gguf"]


def main(argv=None) -> int:
    import plans
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(plans.MODELS),
                    help=f"model tags (default: all of {', '.join(plans.MODELS)})")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--no-corpora", action="store_true")
    ap.add_argument("--no-benchmarks", action="store_true")
    args = ap.parse_args(argv)

    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("HF_DATASETS_OFFLINE") == "1":
        print("error: offline mode is set (HF_HUB_OFFLINE / HF_DATASETS_OFFLINE); unset it",
              file=sys.stderr)
        return 2

    from huggingface_hub import hf_hub_download, snapshot_download
    from fsc import hub
    from fsc.data import FILES

    failures = []

    def attempt(label, fn):
        try:
            print(f"ok   {label}: {fn()}", flush=True)
        except Exception as exc:  # report every failure, then exit non-zero
            failures.append(label)
            print(f"FAIL {label}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if not args.no_models:
        for tag in args.models:
            repo = plans.MODELS[tag]
            attempt(f"model {repo}", lambda: snapshot_download(
                repo, revision=hub.model_revision(repo), ignore_patterns=IGNORE))

    if not args.no_corpora:
        for source, (repo, files) in FILES.items():
            for split, path in files.items():
                attempt(f"corpus {source}/{split}", lambda: hf_hub_download(
                    repo, path, repo_type="dataset", revision=hub.dataset_revision(repo)))

    if not args.no_benchmarks:
        from lm_eval.tasks import TaskManager, get_task_dict
        from fsc.evaluate import ZS_TASKS
        manager = TaskManager()
        for task in ZS_TASKS:
            attempt(f"benchmark {task}",
                    lambda: f"{len(get_task_dict([task], manager)[task].eval_docs)} documents")

    if failures:
        print(f"\n{len(failures)} failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\neverything is cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
