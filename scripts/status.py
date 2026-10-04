#!/usr/bin/env python3
"""One line per run: validation-KL trajectory and, when finished, the headline metrics.

    uv run --no-sync python scripts/status.py [GLOB ...]      # default: out/runs/*/*
"""

from __future__ import annotations

import glob
import json
import os
import sys


def line(d: str) -> str:
    name = "/".join(d.split(os.sep)[-2:])
    out = [f"{name:<58}"]
    log = os.path.join(d, "train_log.jsonl")
    if os.path.exists(log):
        vals = [json.loads(x)["val_kl"] for x in open(log) if x.strip()]
        out.append("valKL " + " ".join(f"{v:.3f}" for v in vals))
    mp = os.path.join(d, "metrics.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        for phase in ("init", "final", "dense"):
            if phase in m:
                ppl = m[phase].get("ppl", {})
                s = " ".join(f"{k}={v:.2f}" for k, v in ppl.items())
                if "zs_mean_acc" in m[phase]:
                    s += f" zs={m[phase]['zs_mean_acc']:.4f}"
                out.append(f"| {phase} {s}")
        if m.get("done"):
            out.append("| done")
    return " ".join(out)


def main() -> int:
    pats = sys.argv[1:] or ["out/runs/*/*"]
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            if os.path.isdir(d):
                print(line(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
