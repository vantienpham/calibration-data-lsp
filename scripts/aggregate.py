#!/usr/bin/env python3
"""Collect every run record into two tables.

    uv run --no-sync python scripts/aggregate.py [--records records] [--out results]

Walks <records>/runs/<model>/<run>/{config,metrics,env}.json and writes
<out>/runs.csv, one row per run with the configuration, the training summary,
and every metric of the training-free initialization (prefix ``init_``) and of
the trained model (prefix ``final_``), and <out>/dense.csv for the uncompressed
models. ``--records records`` reads the published records; ``--records out``
reads the records of your own runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def _flat(prefix: str, m: dict) -> dict:
    row = {}
    if not m:
        return row
    for k, v in m.get("ppl", {}).items():
        row[f"{prefix}ppl_{k}"] = v
    for k, v in m.get("ppl_stream", {}).items():
        row[f"{prefix}pplstream_{k}"] = v
    for k, v in m.get("sp_ppl", {}).items():
        row[f"{prefix}sp_{k}"] = v
    sp = [v for v in m.get("sp_ppl", {}).values() if v and v > 0]
    if sp:
        import math
        row[f"{prefix}sp_geo"] = math.exp(sum(math.log(v) for v in sp) / len(sp))
    for task, r in m.get("zs", {}).items():
        for metric in ("acc", "acc_norm"):
            if metric in r:
                row[f"{prefix}zs_{task}_{metric}"] = r[metric]
            if f"{metric}_stderr" in r:
                row[f"{prefix}zs_{task}_{metric}_se"] = r[f"{metric}_stderr"]
    for k in ("zs_mean_acc", "zs_mean_mixed"):
        if k in m:
            row[f"{prefix}{k}"] = m[k]
    for k, v in m.get("kl", {}).items():
        row[f"{prefix}kl_{k}"] = v
    return row


def load(runs: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of every run under <runs>/<model>/<run>/ (compressed, dense)."""
    rows, dense = [], []
    for mpath in sorted(glob.glob(os.path.join(runs, "*", "*", "metrics.json"))):
        d = os.path.dirname(mpath)
        model, run = d.split(os.sep)[-2:]
        with open(mpath) as f:
            m = json.load(f)
        cpath = os.path.join(d, "config.json")
        cfg = json.load(open(cpath)) if os.path.exists(cpath) else {}
        epath = os.path.join(d, "env.json")
        env = json.load(open(epath)) if os.path.exists(epath) else []
        env = env[-1] if isinstance(env, list) and env else (env or {})
        base = {"model": model, "run": run, "done": bool(m.get("done")), "gpu": env.get("gpu")}
        if "dense" in m:
            r = dict(base)
            r.update(_flat("", m["dense"]))
            dense.append(r)
            continue
        r = dict(base)
        for k in ("mode", "source", "ratio", "alloc", "seed", "n_train", "n_val", "lr", "epochs",
                  "batch_size", "seqlen", "drop_p", "lam_ort", "untied", "alloc_from",
                  "selfgen_path"):
            r[k] = cfg.get(k)
        r["realized_ratio"] = m.get("realized_ratio")
        r["trainable_params"] = m.get("trainable_params")
        r["merge_check_kl"] = m.get("merge_check_kl")
        tr = m.get("train") or {}
        r["best_val_kl"] = tr.get("best_val_kl")
        r["best_epoch"] = tr.get("best_epoch")
        r["epochs_run"] = tr.get("epochs_run")
        hist = tr.get("history") or []
        if hist and hist[0].get("epoch") == -1:
            r["init_val_kl"] = hist[0]["val_kl"]
        for phase, t in (m.get("timing") or {}).items():
            r[f"t_{phase}_s"] = t.get("seconds")
            r[f"mem_{phase}_gb"] = t.get("peak_gb")
        r.update(_flat("init_", m.get("init")))
        r.update(_flat("final_", m.get("final")))
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(dense)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="records", help="root holding runs/<model>/<run>/")
    ap.add_argument("--out", default="results", help="directory for runs.csv and dense.csv")
    args = ap.parse_args(argv)
    runs, dense = load(os.path.join(args.records, "runs"))
    os.makedirs(args.out, exist_ok=True)
    sort = ["model", "run"]
    runs.sort_values(sort).to_csv(os.path.join(args.out, "runs.csv"), index=False)
    dense.sort_values(sort).to_csv(os.path.join(args.out, "dense.csv"), index=False)
    print(f"{len(runs)} runs ({int(runs['done'].sum()) if len(runs) else 0} done), "
          f"{len(dense)} dense -> {args.out}/runs.csv, {args.out}/dense.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
