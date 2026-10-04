#!/usr/bin/env python3
"""How much do removed subspaces depend on the calibration source, before and after training?

    uv run --no-sync python scripts/analyze_subspaces.py --model llama-3.2-1b \
        --runs proj-wikitext2-r50-s0 proj-c4-r50-s0 ... --out out/subspaces/subspaces-llama-3.2-1b-r50.csv

For every unit and every pair of runs (a, b), the overlap between removed
subspaces with orthonormal bases U_a, U_b (both d x k) is

    s(a, b) = ||U_a^T U_b||_F^2 / k,

which is 1 for identical subspaces and k / d in expectation for independent
uniformly random ones. Rows are written for init-vs-init (the training-free
whitened bases of the two sources), final-vs-final (the learned ones), and
init-vs-final within a run (how far training moved the subspace).

Needs, per run: bases_init.pt (written with --save-bases) and best_params.pt.
"""

from __future__ import annotations

import argparse
import itertools
import os

import pandas as pd
import torch


def load_run(path: str, device) -> tuple[dict, dict]:
    init = torch.load(os.path.join(path, "bases_init.pt"), map_location="cpu")
    raw = torch.load(os.path.join(path, "best_params.pt"), map_location="cpu")
    init = {k: v.to(device, torch.float32) for k, v in init.items()}
    final = {}
    for k, V in raw.items():
        q, _ = torch.linalg.qr(V.to(device, torch.float64), mode="reduced")
        final[k] = q.float()
    return init, final


def overlap(Ua: torch.Tensor, Ub: torch.Tensor) -> float:
    k = min(Ua.shape[1], Ub.shape[1])
    if k == 0:
        return float("nan")
    return float((Ua.T @ Ub).pow(2).sum() / k)


def unit_meta(name: str) -> tuple[int, str]:
    block, kind = name.split(".", 1)
    return int(block[1:]), kind


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model tag (directory under out/runs)")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--root", default="out/runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {r: load_run(os.path.join(args.root, args.model, r), device) for r in args.runs}
    rows = []
    for r, (init, final) in data.items():
        for u in final:
            b, kind = unit_meta(u)
            d, k = final[u].shape
            rows.append({"pair": "init-final", "a": r, "b": r, "unit": u, "block": b, "kind": kind,
                         "d": d, "k": k, "chance": k / d, "overlap": overlap(init[u], final[u])})
    for ra, rb in itertools.combinations(args.runs, 2):
        (ia, fa), (ib, fb) = data[ra], data[rb]
        for u in fa:
            if u not in fb:
                continue
            b, kind = unit_meta(u)
            d, k = fa[u].shape
            for pair, Ua, Ub in (("init-init", ia[u], ib[u]), ("final-final", fa[u], fb[u])):
                rows.append({"pair": pair, "a": ra, "b": rb, "unit": u, "block": b, "kind": kind,
                             "d": d, "k": k, "chance": k / d, "overlap": overlap(Ua, Ub)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["pair", "kind"])["overlap"].mean().unstack()
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {len(df)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
