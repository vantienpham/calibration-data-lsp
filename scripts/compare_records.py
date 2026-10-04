#!/usr/bin/env python3
"""Compare your runs with the published records, run by run.

    uv run --no-sync python scripts/compare_records.py [--mine out] [--records records] [RUN ...]

For every run under both <mine>/runs and <records>/runs (or only the given
<model>/<run> names), prints how the re-run differs from the record: the relative
difference of the WikiText-2 and C4 perplexities (BOS-prefixed windows) and of the
SlimPajama geometric-mean perplexity, the difference of the mean zero-shot accuracy
in points, and the relative difference of the best validation KL (trained runs).
The metrics are those of the trained model, or of the evaluated model for dense and
training-free runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os


def _metrics(path: str) -> dict:
    with open(os.path.join(path, "metrics.json")) as f:
        m = json.load(f)
    phase = m.get("final") or m.get("dense") or m.get("init") or {}
    sp = [v for v in (phase.get("sp_ppl") or {}).values() if v and v > 0]
    return {
        "wikitext2": (phase.get("ppl") or {}).get("wikitext2"),
        "c4": (phase.get("ppl") or {}).get("c4"),
        "sp": math.exp(sum(map(math.log, sp)) / len(sp)) if sp else None,
        "acc": phase.get("zs_mean_acc"),
        "val_kl": (m.get("train") or {}).get("best_val_kl"),
    }


def _rel(a, b) -> str:
    return "n/a" if a is None or b is None else f"{100 * (b / a - 1):+.2f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="<model>/<run> names (default: every common run)")
    ap.add_argument("--mine", default="out", help="root of your runs (holds runs/)")
    ap.add_argument("--records", default="records", help="root of the published records")
    args = ap.parse_args(argv)

    def present(root):
        return {os.path.relpath(os.path.dirname(p), os.path.join(root, "runs"))
                for p in glob.glob(os.path.join(root, "runs", "*", "*", "metrics.json"))}

    names = args.runs or sorted(present(args.mine) & present(args.records))
    print(f"{'run':45s} {'WikiText-2':>10s} {'C4':>9s} {'SlimPajama':>10s} {'accuracy':>9s} {'val KL':>8s}")
    for name in names:
        a = _metrics(os.path.join(args.records, "runs", name))
        b = _metrics(os.path.join(args.mine, "runs", name))
        acc = "n/a" if a["acc"] is None or b["acc"] is None else f"{100 * (b['acc'] - a['acc']):+.2f} pt"
        print(f"{name:45s} {_rel(a['wikitext2'], b['wikitext2']):>10s} {_rel(a['c4'], b['c4']):>9s} "
              f"{_rel(a['sp'], b['sp']):>10s} {acc:>9s} {_rel(a['val_kl'], b['val_kl']):>8s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
