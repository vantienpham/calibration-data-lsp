#!/usr/bin/env python3
"""Recompute every quantitative statement of the paper's Results from the records.

    uv run --no-sync python scripts/claims.py [--records records] [--results results]

Prints, per model and ratio, the facts the text asserts (which source is best on
each corpus, accuracy spreads, relative-KL rankings) and the numbers quoted in
the ablations, the appendices and the scale check, so the paper can be checked
against the data mechanically. ``make claims`` writes the output to
results/claims.txt.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

MODELS = ["opt-1.3b", "llama-3.2-1b", "qwen3-1.7b"]
SOURCES = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen", "selfdoc"]
REAL = ["wikitext2", "c4", "slimpajama", "alpaca"]
KL = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen"]


def grid(df, m, r):
    x = df[(df.model == m) & df.run.isin([f"proj-{s}-r{int(r * 100)}-s0" for s in SOURCES])]
    return x.set_index("source")


RECORDS, RESULTS = "records", "results"


def main(argv=None) -> int:
    global RECORDS, RESULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--results", default=RESULTS)
    args = ap.parse_args(argv)
    RECORDS, RESULTS = args.records, args.results
    df = pd.read_csv(os.path.join(RESULTS, "runs.csv"))
    for r in (0.3, 0.5):
        for m in MODELS:
            g = grid(df, m, r)
            if len(g) < 6 or g["final_ppl_c4"].isna().any():
                print(f"{m} r{int(r * 100)}: incomplete ({len(g)} runs)")
                continue
            print(f"== {m} r{int(r * 100)}")
            for k in ("final_ppl_wikitext2", "final_ppl_c4", "final_sp_geo"):
                v = g[k]
                print(f"  {k:22s} best {v.idxmin():10s} {v.min():8.2f} | worst {v.idxmax():10s} {v.max():8.2f}")
            acc = 100 * g["final_zs_mean_acc"]
            real = acc[REAL]
            print(f"  acc best {acc.idxmax()} {acc.max():.1f} | lowest real {real.idxmin()} {real.min():.1f} | "
                  f"spread all {acc.max() - acc.min():.1f}, excl. selfgen {acc.drop('selfgen').max() - acc.drop('selfgen').min():.1f}")
            M = g.loc[SOURCES, [f"final_kl_{e}" for e in KL]].to_numpy(float)
            rel = M / M.min(axis=0, keepdims=True)
            geo = np.exp(np.log(rel).mean(1))
            mx = rel.max(1)
            order = np.argsort(geo)
            print("  geo rank: " + ", ".join(f"{SOURCES[i]} {geo[i]:.3f}" for i in order))
            print("  max rank: " + ", ".join(f"{SOURCES[i]} {mx[i]:.2f}" for i in np.argsort(mx)))
            diag = all(int(np.argmin(M[:, j])) == SOURCES.index(e) for j, e in enumerate(KL))
            print(f"  diagonal is the column minimum on every held-out set: {diag}; "
                  f"self column best: {SOURCES[int(np.argmin(M[:, -1]))]}")
            print(f"  max rel KL wikitext2 {mx[0]:.2f}, alpaca {mx[3]:.2f}")
    protocol(df)
    lr_grid()
    oblique(df)
    accuracy_without_boolq(df)
    ablations(df)
    subspaces()
    scale(df)
    return 0


def protocol(df):
    """Section 3.5: merge exactness and realized compression ratios."""
    import glob
    import json
    proj = df[df.run.str.startswith(("proj", "repro")) & df.done]
    print(f"== protocol: max merge-check KL {proj.merge_check_kl.max():.2e} over {proj.merge_check_kl.notna().sum()} runs")
    dev = []
    for f in sorted(glob.glob(os.path.join(RECORDS, "runs", "*", "*", "alloc.json"))):
        a = json.load(open(f))
        cfg = os.path.join(os.path.dirname(f), "config.json")
        if a.get("alloc") != "uniform" or not os.path.exists(cfg):
            continue
        target = json.load(open(cfg)).get("ratio")
        if target is not None:
            dev.append((abs(a["realized_ratio"] - target), f))
    worst = max(dev)
    print(f"  uniform allocation: {len(dev)} runs, worst |realized - target| = {100 * worst[0]:.3f} points ({worst[1]})")


def lr_grid():
    """Appendices A and B: the learning-rate grids (an -e8 pilot replaces the earlier one at its rate)."""
    import glob
    import json
    import re
    print("== learning-rate grid (best validation KL per rate)")
    for m in MODELS:
        runs = {}
        for d in sorted(glob.glob(os.path.join(RECORDS, "runs", m, "pilot-proj-c4-r50-lr*"))):
            h = re.search(r"-lr([0-9.]+e-[0-9]+)(-e8)?$", d)
            if not h or (float(h.group(1)) in runs and not h.group(2)):
                continue
            runs[float(h.group(1))] = [json.loads(x)["val_kl"] for x in open(f"{d}/train_log.jsonl") if x.strip()]
        best = min(runs, key=lambda lr: min(runs[lr]))
        print(f"  {m:13s} " + " ".join(f"{lr:.0e}: {min(v):.4f} ({len(v)} ep, first {v[0]:.3f})"
                                         for lr, v in sorted(runs.items()))
              + f" | best {best:.0e}, 1e-3 excess {100 * (min(runs[1e-3]) / min(runs[best]) - 1):.1f}%")
    print("  opt-125m, reproduction setting (WikiText-2, 10 epochs)")
    for r in (30, 50, 70):
        kl = {}
        for d in sorted(glob.glob(os.path.join(RECORDS, "runs", "opt-125m", f"pilot-proj-wikitext2-r{r}-lr*"))):
            with open(f"{d}/metrics.json") as f:
                kl[float(re.search(r"-lr([0-9.]+e-[0-9]+)$", d).group(1))] = json.load(f)["train"]["best_val_kl"]
        print(f"    -{r}%: " + " ".join(f"{lr:.0e}: {v:.4f}" for lr, v in sorted(kl.items()))
              + f" | best {min(kl, key=kl.get):.0e}")


def oblique(df):
    """Section 4.1: oblique truncation against the orthogonal recast (Table oblique)."""
    print("== oblique vs recast (C4, SP geo)")
    n, worse, gains = 0, [], []
    for r in (30, 50):
        for m in MODELS:
            rec = grid(df, m, r / 100)
            obl = df[(df.model == m) & df.run.isin([f"none-{s}-r{r}-s0" for s in SOURCES])].set_index("source")
            spreads = []
            for k in ("ppl_c4", "sp_geo"):
                a, b = rec[f"init_{k}"], obl[f"init_{k}"]
                for s in SOURCES:
                    n += 1
                    gains.append(a[s] / b[s])
                    if b[s] > a[s]:
                        worse.append((m, r, s, k))
                spreads.append(b.max() / b.min())
                print(f"  {m:13s} r{r} {k:7s} oblique best {b.idxmin():10s} spread {b.max() / b.min():5.1f}")
    print(f"  oblique better in {n - len(worse)} of {n}; exceptions {worse}; max gain {max(gains):.1f}")
    for r in (30, 50):
        for m in MODELS:
            o = df[(df.model == m) & df.run.isin([f"none-{s}-r{r}-s0" for s in SOURCES])].set_index("source")
            t = grid(df, m, r / 100)
            d = 100 * (t["final_zs_mean_acc"] - o["init_zs_mean_acc"])
            print(f"  acc gain of training over oblique {m} r{r}: {d.min():+.1f} to {d.max():+.1f}")


def accuracy_without_boolq(df):
    """Section 4.2: BoolQ drives the OPT-1.3B accuracy ranking."""
    tasks = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa"]
    lo, hi = 100.0, 0.0
    for r in (30, 50):
        for m in MODELS:
            g = grid(df, m, r / 100)
            a6 = 100 * g[[f"final_zs_{t}_acc" for t in tasks]].mean(axis=1)
            bq = 100 * g["final_zs_boolq_acc"]
            lo, hi = min(lo, bq.min()), max(hi, bq.max())
            top = a6[a6 >= a6.max() - 0.05].index.tolist()
            print(f"  {m:13s} r{r} best without BoolQ: {top} {a6.max():.1f}")
    print(f"  BoolQ accuracy of compressed models: {lo:.1f} to {hi:.1f}")


def _cells(df, run, m="llama-3.2-1b"):
    r = df[(df.model == m) & (df.run == run)].iloc[0]
    return np.array([r.final_ppl_wikitext2, r.final_ppl_c4, r.final_sp_geo, 100 * r.final_zs_mean_acc])


def ablations(df):
    """Sections 4.3 to 4.7 (Llama-3.2-1B at -50% unless stated)."""
    print("== size")
    for s in ("c4", "selfdoc"):
        rows = {n: _cells(df, f"proj-{s}-r50-s0" + ("" if n == 512 else f"-n{n}")) for n in (64, 128, 256, 512, 1024)}
        for n, v in rows.items():
            print(f"  {s:8s} n{n:5d} " + " ".join(f"{x:7.1f}" for x in v))
    print("  c4       n 4096 " + " ".join(f"{x:7.1f}" for x in _cells(df, "proj-c4-r50-s0-n4096")))
    print("  selfdoc  n 4096 " + " ".join(f"{x:7.1f}" for x in _cells(df, "proj-selfdoc-r50-s0-fresh4096")))
    print("== measured-KL allocation (improved cells: lower ppl, higher acc)")
    better, rel = 0, []
    for s in SOURCES:
        u, k = _cells(df, f"proj-{s}-r50-s0"), _cells(df, f"proj-{s}-r50-s0-klalloc")
        better += int((k[:3] < u[:3]).sum() + (k[3] > u[3]))
        rel += list(1 - k[:3] / u[:3])
        print(f"  {s:10s} uniform " + " ".join(f"{x:6.1f}" for x in u) + " | kl " + " ".join(f"{x:6.1f}" for x in k))
    print(f"  improved {better} of 24, max ppl reduction {100 * max(rel):.0f}%")
    print("== free factors (better lr on validation KL) vs projectors")
    for m in ("llama-3.2-1b", "opt-1.3b"):
        for s in ("wikitext2", "c4", "selfgen"):
            p = _cells(df, f"proj-{s}-r50-s0", m)
            fs = df[(df.model == m) & df.run.isin([f"factors-{s}-r50-s0-lr1e-5", f"factors-{s}-r50-s0-lr1e-4"])]
            f = _cells(df, fs.loc[fs.best_val_kl.idxmin(), "run"], m)
            print(f"  {m:13s} {s:10s} rel ppl " + " ".join(f"{100 * (b / a - 1):+5.1f}%" for a, b in zip(p[:3], f[:3]))
                  + f" | acc {f[3] - p[3]:+.1f}")
    print("== seeds")
    for s in SOURCES:
        v = np.array([_cells(df, f"proj-{s}-r50-s{sd}") for sd in (0, 1, 2)])
        mu, sd = v.mean(0), v.std(0, ddof=1)
        print(f"  {s:10s} mean " + " ".join(f"{x:6.1f}" for x in mu) + " | rel sd "
              + " ".join(f"{100 * x:4.1f}%" for x in sd[:3] / mu[:3]) + f" | acc sd {sd[3]:.2f}")
    print("== single-domain (ratio to dense)")
    d = pd.read_csv(os.path.join(RESULTS, "dense.csv")).set_index("model").loc["llama-3.2-1b"]
    doms = ["cc", "c4", "github", "book", "arxiv", "wiki", "stack"]
    for run in ("proj-slimpajama-r50-s0", "proj-sp-github-r50-s0", "proj-sp-arxiv-r50-s0", "proj-sp-book-r50-s0"):
        r = df[(df.model == "llama-3.2-1b") & (df.run == run)].iloc[0]
        print(f"  {run:24s} " + " ".join(f"{x} {r[f'final_sp_{x}'] / d[f'sp_{x}']:6.2f}" for x in doms))


def subspaces():
    """Section 4.4: overlaps of removed subspaces."""
    print("== subspaces")
    for m in MODELS:
        for r in (30, 50):
            d = pd.read_csv(os.path.join(RECORDS, "subspaces", f"subspaces-{m}-r{r}.csv"))
            ii = d[d.pair == "init-init"].groupby(["a", "b"]).overlap.mean()
            ff = d[d.pair == "final-final"].groupby(["a", "b"]).overlap.mean()
            print(f"  {m:13s} r{r} init-final {d[d.pair == 'init-final'].overlap.mean():.4f} "
                  f"across init {ii.mean():.3f} final {ff.mean():.3f} chance {d.chance.mean():.3f} "
                  f"pairs closer after training {int((ff > ii).sum())}/{len(ii)}")



def scale(df, m="llama-3.1-8b", sources=("wikitext2", "c4", "slimpajama", "alpaca", "selfdoc")):
    """Section 4.8: Llama-3.1-8B at -30% (learning-rate pilots, then five sources)."""
    import glob
    import json
    print("== scale")
    for d in sorted(glob.glob(os.path.join(RECORDS, "runs", m, "pilot-proj-c4-r30-lr*"))):
        log = f"{d}/train_log.jsonl"
        if os.path.exists(log):
            v = [json.loads(x)["val_kl"] for x in open(log) if x.strip()]
            print(f"  pilot {d.split('-lr')[-1]:7s} {len(v)} ep, best {min(v):.4f}")
    g = df[(df.model == m) & df.run.isin([f"proj-{s}-r30-s0" for s in sources]) & df.done].set_index("source")
    if len(g) < len(sources):
        print(f"  {len(g)} of {len(sources)} runs done")
        return
    for k in ("init_ppl_wikitext2", "init_ppl_c4", "final_ppl_wikitext2", "final_ppl_c4", "final_sp_geo"):
        v = g[k]
        print(f"  {k:22s} best {v.idxmin():10s} {v.min():8.2f} | worst {v.idxmax():10s} {v.max():9.2f} | spread {v.max() / v.min():.1f}")
    acc = 100 * g["final_zs_mean_acc"]
    print(f"  acc best {acc.idxmax()} {acc.max():.1f} | lowest {acc.idxmin()} {acc.min():.1f}")
    M = g.loc[list(sources), [f"final_kl_{e}" for e in KL]].to_numpy(float)
    rel = M / M.min(axis=0, keepdims=True)
    geo, mx = np.exp(np.log(rel).mean(1)), rel.max(1)
    print("  geo rank: " + ", ".join(f"{sources[i]} {geo[i]:.3f}" for i in np.argsort(geo)))
    print("  max rank: " + ", ".join(f"{sources[i]} {mx[i]:.2f}" for i in np.argsort(mx)))
    print("  column best: " + ", ".join(f"{e}: {sources[int(np.argmin(M[:, j]))]}" for j, e in enumerate(KL)))


if __name__ == "__main__":
    raise SystemExit(main())
