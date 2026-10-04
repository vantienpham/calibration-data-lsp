#!/usr/bin/env python3
"""Tables and figures of the manuscript, from the aggregated run summaries.

    uv run --no-sync python scripts/aggregate.py
    uv run --no-sync python scripts/make_report.py

Reads <results>/{runs,dense}.csv (scripts/aggregate.py) and, from <records>, the
source statistics (stats/), the subspace overlaps (subspaces/) and the training
logs of the learning-rate pilots (runs/). Writes the LaTeX table bodies of the
paper to <results>/tables/ and its figures to <results>/figures/.
Every number in the paper comes from here; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib

# Reproducible PDFs: matplotlib stamps SOURCE_DATE_EPOCH instead of the current time,
# so regenerating an unchanged figure leaves the file, and git, untouched.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODELS = ["opt-1.3b", "llama-3.2-1b", "qwen3-1.7b"]
MODEL_LABEL = {"opt-1.3b": "OPT-1.3B", "llama-3.2-1b": "Llama-3.2-1B", "qwen3-1.7b": "Qwen3-1.7B",
               "opt-125m": "OPT-125M", "llama-3.1-8b": "Llama-3.1-8B"}
SOURCES = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen", "selfdoc"]
SOURCE_LABEL = {"wikitext2": "WikiText-2", "c4": "C4", "slimpajama": "SlimPajama",
                "alpaca": "Alpaca", "selfgen": "Self (seq.)", "selfdoc": "Self (doc.)"}
KL_SETS = ["wikitext2", "c4", "slimpajama", "alpaca", "selfgen"]
KL_LABEL = {"wikitext2": "WikiText-2", "c4": "C4", "slimpajama": "SlimPajama",
            "alpaca": "Alpaca", "selfgen": "Self"}
SP_DOMAINS = ["cc", "c4", "github", "book", "arxiv", "wiki", "stack"]
SP_LABEL = {"cc": "CC", "c4": "C4", "github": "GitHub", "book": "Books",
            "arxiv": "arXiv", "wiki": "Wiki", "stack": "StackEx."}
TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa", "boolq"]

# Validated categorical palette (dataviz reference instance, first six slots, light mode);
# fixed order, one slot per calibration source, never cycled.
PALETTE = {"wikitext2": "#2a78d6", "c4": "#eb6834", "slimpajama": "#1baf7a",
           "alpaca": "#eda100", "selfgen": "#e87ba4", "selfdoc": "#008300"}
MARKER = {"wikitext2": "o", "c4": "s", "slimpajama": "^", "alpaca": "D", "selfgen": "v",
          "selfdoc": "P"}
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "figure.dpi": 150,
})


# ----------------------------------------------------------------------------
# selection helpers
# ----------------------------------------------------------------------------

def rtag(ratio: float) -> str:
    return f"r{int(round(ratio * 100))}"


def pick(df: pd.DataFrame, model: str, run: str) -> pd.Series | None:
    rows = df[(df.model == model) & (df.run == run) & (df.done)]
    return None if rows.empty else rows.iloc[0]


def main_run(df, model, source, ratio, mode="proj", seed=0, tag=""):
    return pick(df, model, f"{mode}-{source}-{rtag(ratio)}-s{seed}{tag}")


def fmt(x, nd=2, bold=False):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "--"
    if abs(x) >= 1000:
        s = f"{x:,.0f}".replace(",", "{,}")
    elif abs(x) >= 100:
        s = f"{x:.0f}"
    else:
        s = f"{x:.{nd}f}"
    return f"\\textbf{{{s}}}" if bold else s


def write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print(f"wrote {path}")


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def table_main(df, dense, out, phase="final", ratios=(0.3, 0.5), mode="proj", init_kind=None):
    """Rows: calibration source; per model: WikiText-2 ppl, C4 ppl, KL to self samples, mean acc."""
    metrics = [("ppl_wikitext2", "Wiki", 1, "min"), ("ppl_c4", "C4", 1, "min"),
               ("sp_geo", "SP", 1, "min"), ("zs_mean_acc", "Acc", 1, "max")]
    lines = []
    head = " & ".join(f"\\multicolumn{{{len(metrics)}}}{{c}}{{\\textbf{{{MODEL_LABEL[m]}}}}}"
                      for m in MODELS)
    lines.append(f"\\textbf{{Calibration}} & {head} \\\\")
    cm = " ".join(f"\\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}"
                  for i in range(len(MODELS)))
    lines.append(cm)
    sub = " & ".join(" & ".join(f"\\textit{{{lab}}}" for _, lab, _, _ in metrics) for _ in MODELS)
    lines.append(f" & {sub} \\\\")
    lines.append("\\midrule")
    drow = ["Dense"]
    for m in MODELS:
        d = dense[dense.model == m]
        for key, _, nd, _ in metrics:
            if d.empty or key not in d:
                drow.append("--")
            else:
                v = d.iloc[0][key]
                drow.append(fmt(100 * v if key.startswith("zs") else v, nd))
    lines.append(" & ".join(drow) + " \\\\")
    for ratio in ratios:
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{1 + len(metrics) * len(MODELS)}}}{{l}}"
                     f"{{\\textit{{${-int(ratio * 100)}\\%$}}}} \\\\")
        # best per column
        vals = {}
        for m in MODELS:
            for key, _, _, best in metrics:
                col = []
                for s in SOURCES:
                    r = main_run(df, m, s, ratio, mode=mode)
                    v = None if r is None else r.get(f"{phase}_{key}")
                    col.append(v if v is not None and not pd.isna(v) else None)
                good = [v for v in col if v is not None]
                tgt = (min(good) if best == "min" else max(good)) if good else None
                vals[(m, key)] = (col, tgt)
        for i, s in enumerate(SOURCES):
            row = [SOURCE_LABEL[s]]
            for m in MODELS:
                for key, _, nd, _ in metrics:
                    col, tgt = vals[(m, key)]
                    v = col[i]
                    shown = None if v is None else (100 * v if key.startswith("zs") else v)
                    row.append(fmt(shown, nd, bold=(v is not None and v == tgt)))
            lines.append(" & ".join(row) + " \\\\")
    write(out, "\n".join(lines) + "\n")


def table_lr(runs_dir: str, out: str):
    """Validation KL after each epoch of the learning-rate pilots (C4, -50%).

    A pilot named ...-e8 (plan pilot-grid: eight epochs, run to the end) replaces the
    earlier pilot at the same rate. Bold: the lowest value of each model, i.e. the
    selected rate at its best epoch. Any epoch without a value is shown as "--" so that
    an incomplete grid is visible, not hidden.
    """
    import re
    runs = {}
    for m in MODELS:
        for d in sorted(glob.glob(os.path.join(runs_dir, m, "pilot-proj-c4-r50-lr*"))):
            hit = re.search(r"-lr([0-9.]+e-[0-9]+)(-e8)?$", d)
            log = os.path.join(d, "train_log.jsonl")
            if not hit or not os.path.exists(log):
                continue
            key = (m, float(hit.group(1)))
            if key in runs and not hit.group(2):
                continue                      # an -e8 pilot already replaces this one
            vals = [json.loads(x)["val_kl"] for x in open(log) if x.strip()]
            runs[key] = vals
    width = max((len(v) for v in runs.values()), default=0)
    best = {m: min(min(v) for (mm, _), v in runs.items() if mm == m) for m, _ in runs}
    lines = ["\\textbf{Model} & \\textbf{LR} & " +
             " & ".join(f"\\textbf{{{e + 1}}}" for e in range(width)) + " \\\\", "\\midrule"]
    for (m, lr), vals in sorted(runs.items(), key=lambda kv: (MODELS.index(kv[0][0]), kv[0][1])):
        cells = [fmt(v, 3, bold=(v == best[m])) for v in vals] + ["--"] * (width - len(vals))
        lines.append(f"{MODEL_LABEL[m]} & {lr:.0e} & " + " & ".join(cells) + " \\\\")
    write(out, "\n".join(lines) + "\n")


# Published numbers of Bini et al. (2026) with uniform allocation, WikiText-2 calibration,
# GPTQ-convention perplexity: untrained whitened initialization (their Table 6) and
# trained LSP (their Table 7 for OPT-125M, Table 9 for OPT-1.3B).
LSP_PAPER = {
    "opt-125m": {"dense": 27.6, "init": {0.3: 582, 0.5: 664, 0.7: 1538},
                 "final": {0.3: 31.7, 0.5: 35.9, 0.7: 44.8}},
    "opt-1.3b": {"dense": 14.6, "init": {0.3: 26.1, 0.5: 74.4, 0.7: 814},
                 "final": {0.3: 16.5, 0.5: 18.7, 0.7: 24.2}},
}


def table_repro(df, dense, out):
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\textbf{Init} & \\textbf{Init, no BOS} & "
             "\\textbf{Oblique} & \\textbf{Init (LSP)} & \\textbf{Trained} & "
             "\\textbf{Trained (LSP)} \\\\",
             "\\midrule"]
    key = "pplstream_wikitext2"

    def g(row, ph):
        if row is None:
            return None
        v = row.get(f"{ph}_{key}")
        return None if pd.isna(v) else v

    for m in ("opt-125m", "opt-1.3b"):
        d = dense[dense.model == m]
        dv = None if d.empty else d.iloc[0]["pplstream_wikitext2"]
        lines.append(f"{MODEL_LABEL[m]} & dense & \\multicolumn{{6}}{{c}}{{{fmt(dv, 1)} (ours), "
                     f"{fmt(LSP_PAPER[m]['dense'], 1)} (LSP)}} \\\\")
        for ratio in (0.3, 0.5, 0.7):
            R = rtag(ratio)
            r = pick(df, m, f"repro-wikitext2-{R}")
            nb = pick(df, m, f"init-nobos-wikitext2-{R}")
            ob = pick(df, m, f"oblique-wikitext2-{R}")
            lines.append(f" & $-{int(ratio * 100)}\\%$ & {fmt(g(r, 'init'), 1)} & {fmt(g(nb, 'init'), 1)} & "
                         f"{fmt(g(ob, 'init'), 1)} & {fmt(LSP_PAPER[m]['init'][ratio], 1)} & "
                         f"{fmt(g(r, 'final'), 1)} & {fmt(LSP_PAPER[m]['final'][ratio], 1)} \\\\")
    write(out, "\n".join(lines) + "\n")


def _cells(row, phase="final"):
    """Wiki ppl, C4 ppl, SlimPajama geo-mean ppl, mean acc (percent) of one run."""
    if row is None:
        return [None] * 4
    def g(k):
        v = row.get(f"{phase}_{k}")
        return None if v is None or pd.isna(v) else v
    acc = g("zs_mean_acc")
    return [g("ppl_wikitext2"), g("ppl_c4"), g("sp_geo"), None if acc is None else 100 * acc]


def _row(label, vals, nds=(1, 1, 1, 1)):
    return label + " & " + " & ".join(fmt(v, nd) for v, nd in zip(vals, nds)) + " \\\\"


def table_ablations(df, out, model="llama-3.2-1b", ratio=0.5):
    """Llama-3.2-1B at -50%: every ablation against its reference run."""
    R = rtag(ratio)
    get = lambda run: pick(df, model, run)
    lines = ["\\textbf{Variant} & \\textbf{Wiki} & \\textbf{C4} & \\textbf{SP} & \\textbf{Acc} \\\\",
             "\\midrule"]
    groups = [
        ("Initialization statistics vs.\\ training windows", [
            ("C4 init, C4 training", f"proj-c4-{R}-s0"),
            ("Self (doc.) init, C4 training", f"proj-c4-{R}-s0-initselfdoc"),
            ("Self (doc.) init, Self (doc.) training", f"proj-selfdoc-{R}-s0"),
            ("C4 init, Self (doc.) training", f"proj-selfdoc-{R}-s0-initc4"),
            ("WikiText-2 init, WikiText-2 training", f"proj-wikitext2-{R}-s0"),
            ("SlimPajama init, WikiText-2 training", f"proj-wikitext2-{R}-s0-initslimpajama"),
            ("SlimPajama init, SlimPajama training", f"proj-slimpajama-{R}-s0"),
            ("WikiText-2 init, SlimPajama training", f"proj-slimpajama-{R}-s0-initwikitext2"),
        ]),
        ("Window format (C4)", [
            ("BOS at window start", f"proj-c4-{R}-s0"),
            ("BOS at every document start", f"proj-c4-{R}-s0-docbos"),
            ("Plain chunks, no BOS", f"proj-c4-{R}-s0-nobos"),
        ]),
        ("Self-generated sequences: temperature", [
            ("$T=0.7$", f"proj-selfgenT0.7-{R}-s0"),
            ("$T=1.0$", f"proj-selfgen-{R}-s0"),
            ("$T=1.3$", f"proj-selfgenT1.3-{R}-s0"),
        ]),
    ]
    for title, items in groups:
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for label, run in items:
            lines.append(_row(label, _cells(get(run))))
        lines.append("\\midrule")
    lines[-1] = lines[-1].replace("\\midrule", "")
    write(out, "\n".join(l for l in lines if l) + "\n")


def table_alloc(df, out, model="llama-3.2-1b", ratio=0.5):
    R = rtag(ratio)
    lines = ["\\textbf{Calibration} & \\multicolumn{4}{c}{\\textbf{Uniform}} & "
             "\\multicolumn{4}{c}{\\textbf{Measured KL}} \\\\",
             "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
             " & Wiki & C4 & SP & Acc & Wiki & C4 & SP & Acc \\\\", "\\midrule"]
    for s in SOURCES:
        a = _cells(pick(df, model, f"proj-{s}-{R}-s0"))
        b = _cells(pick(df, model, f"proj-{s}-{R}-s0-klalloc"))
        nds = (1, 1, 1, 1) * 2
        lines.append(SOURCE_LABEL[s] + " & " + " & ".join(fmt(v, nd) for v, nd in zip(a + b, nds))
                     + " \\\\")
    write(out, "\n".join(lines) + "\n")


def table_factors(df, out, ratio=0.5):
    R = rtag(ratio)
    lines = ["\\textbf{Model} & \\textbf{Calibration} & \\textbf{Learned} & \\textbf{Trainable} & "
             "\\textbf{Wiki} & \\textbf{C4} & \\textbf{SP} & \\textbf{Acc} \\\\", "\\midrule"]
    for m in ("llama-3.2-1b", "opt-1.3b"):
        for s in ("wikitext2", "c4", "selfgen"):
            p = pick(df, m, f"proj-{s}-{R}-s0")
            cands = [pick(df, m, f"factors-{s}-{R}-s0-lr{lr}") for lr in ("1e-5", "1e-4")]
            cands = [c for c in cands if c is not None]
            # free factors: keep the learning rate with the lower validation KL
            f = min(cands, key=lambda c: c["best_val_kl"]) if cands else None
            for label, row in (("projectors", p), ("free factors", f)):
                tp = None if row is None else row.get("trainable_params")
                tps = "--" if tp is None or pd.isna(tp) else f"{tp / 1e6:.0f}\\,M"
                vals = _cells(row)
                first = label == "projectors"
                mlab = f"\\multirow{{6}}{{*}}{{{MODEL_LABEL[m]}}}" if first and s == "wikitext2" else ""
                slab = f"\\multirow{{2}}{{*}}{{{SOURCE_LABEL[s]}}}" if first else ""
                lines.append(f"{mlab} & {slab} & {label} & {tps} & "
                             + " & ".join(fmt(v, nd) for v, nd in zip(vals, (1, 1, 1, 1))) + " \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    write(out, "\n".join(lines) + "\n")


def table_seeds(df, out, model="llama-3.2-1b", ratio=0.5):
    R = rtag(ratio)
    lines = ["\\textbf{Calibration} & \\textbf{Wiki} & \\textbf{C4} & \\textbf{SP} & \\textbf{Acc} \\\\",
             "\\midrule"]
    for s in SOURCES:
        rows = [pick(df, model, f"proj-{s}-{R}-s{sd}") for sd in (0, 1, 2)]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        cells = np.array([[np.nan if v is None else v for v in _cells(r)] for r in rows])
        mu, sd = np.nanmean(cells, 0), np.nanstd(cells, 0, ddof=1) if len(rows) > 1 else [np.nan] * 4
        parts = []
        for j, nd in enumerate((1, 1, 1, 1)):
            parts.append(f"{fmt(mu[j], nd)} $\\pm$ {fmt(sd[j], nd)}" if len(rows) > 1 else fmt(mu[j], nd))
        lines.append(f"{SOURCE_LABEL[s]} ({len(rows)}) & " + " & ".join(parts) + " \\\\")
    write(out, "\n".join(lines) + "\n")


def table_tasks(df, dense, out, ratios=(0.3, 0.5)):
    """Per-task zero-shot accuracy (percent), every model, ratio and source."""
    short = {"arc_easy": "ARC-e", "arc_challenge": "ARC-c", "piqa": "PIQA", "hellaswag": "HellaS.",
             "winogrande": "WinoG.", "openbookqa": "OBQA", "boolq": "BoolQ"}
    lines = ["\\textbf{Model} & \\textbf{Ratio} & \\textbf{Calibration} & " +
             " & ".join(f"\\textbf{{{short[t]}}}" for t in TASKS) + " & \\textbf{Mean} \\\\",
             "\\midrule"]
    n_rows = 1 + len(ratios) * len(SOURCES)
    for m in MODELS:
        d = dense[dense.model == m].iloc[0]
        vals = [100 * d[f"zs_{t}_acc"] for t in TASKS]
        lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{MODEL_LABEL[m]}}} & 0\\% & none (dense) & "
                     + " & ".join(fmt(v, 1) for v in vals) + f" & {fmt(100 * d['zs_mean_acc'], 1)} \\\\")
        for ratio in ratios:
            lines.append(f"\\cmidrule(lr){{2-{3 + len(TASKS) + 1}}}")
            for s in SOURCES:
                r = main_run(df, m, s, ratio)
                vals = [None if r is None else 100 * r.get(f"final_zs_{t}_acc") for t in TASKS]
                mean = None if r is None else 100 * r.get("final_zs_mean_acc")
                tag = (f"\\multirow{{{len(SOURCES)}}}{{*}}{{$-{int(round(ratio * 100))}\\%$}}"
                       if s == SOURCES[0] else "")
                lines.append(f" & {tag} & {SOURCE_LABEL[s]} & " + " & ".join(fmt(v, 1) for v in vals)
                             + f" & {fmt(mean, 1)} \\\\")
        lines.append("\\midrule")
    write(out, "\n".join(lines[:-1]) + "\n")


CONTENT = {"wikitext2": "Wikipedia articles", "c4": "filtered web text",
           "slimpajama": r"web, code, books, arXiv, Wikipedia, Q\&A",
           "alpaca": "instruction--response pairs",
           "selfgen": "2048-token samples of the dense model",
           "selfdoc": r"independent samples, $\le$512 tokens"}


def load_stats(tables_dir="records/stats") -> dict:
    out = {}
    for m in MODELS:
        p = os.path.join(tables_dir, f"sources-{m}.json")
        if os.path.exists(p):
            out[m] = json.load(open(p))
    return out


def table_sources(stats: dict, out: str):
    lines = ["\\textbf{Source} & \\textbf{Content} & " +
             " & ".join(f"\\textbf{{{MODEL_LABEL[m]}}}" for m in MODELS) + " \\\\",
             " & & \\multicolumn{3}{c}{documents per window / dense NLL (nats per token)} \\\\",
             "\\midrule"]
    for s_ in SOURCES:
        cells = []
        for m in MODELS:
            st = stats.get(m, {}).get(s_)
            cells.append("--" if st is None else f"{st['docs_per_window']:.1f} / {st['nll']:.2f}")
        lines.append(f"{SOURCE_LABEL[s_]} & {CONTENT[s_]} & " + " & ".join(cells) + " \\\\")
    write(out, "\n".join(lines) + "\n")


def fig_entropy(stats: dict, out: str):
    """Dense-model entropy along the window, per source: the drift of exact self-samples."""
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.2), sharey=True)
    for ax, m in zip(axes, MODELS):
        for s_ in SOURCES:
            st = stats.get(m, {}).get(s_)
            if not st or "entropy_by_pos_128" not in st:
                continue
            y = st["entropy_by_pos_128"]
            x = [128 * i + 64 for i in range(len(y))]
            ax.plot(x, y, color=PALETTE[s_], marker=MARKER[s_], markersize=3.5, lw=1.4,
                    label=SOURCE_LABEL[s_])
        ax.set_title(MODEL_LABEL[m])
        ax.set_xlabel("position in window")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlim(0, 2048)
    axes[0].set_ylabel("entropy (nats/token)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def heat(ax, M, rows, cols, fmt_s="{:.2f}", vmin=None, vmax=None, log=False, fontsize=6.5):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    data = np.array(M, dtype=float)
    if not np.isfinite(data).any():
        ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
        return None
    norm = (LogNorm(vmin=vmin or np.nanmin(data), vmax=vmax or np.nanmax(data)) if log
            else Normalize(vmin=vmin if vmin is not None else np.nanmin(data),
                           vmax=vmax if vmax is not None else np.nanmax(data)))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            light = norm(v) < 0.55
            ax.text(j, i, fmt_s.format(v), ha="center", va="center", fontsize=fontsize,
                    color=INK if light else "white")
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def fig_kl_matrix(df, out, ratio=0.5):
    """KL(dense || compressed) on each held-out set (columns) per calibration source (rows).

    Cells are colored by the ratio to the best source for that held-out set, on one log
    scale shared by all panels, and annotated with the raw KL; the best is outlined.
    """
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    norm = LogNorm(vmin=1.0, vmax=4.0)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.55), sharey=True)
    im = None
    for ax, m in zip(axes, MODELS):
        M = np.array([[np.nan if (r := main_run(df, m, s_, ratio)) is None
                       else r.get(f"final_kl_{e}", np.nan) for e in KL_SETS] for s_ in SOURCES],
                     dtype=float)
        ax.set_title(MODEL_LABEL[m])
        ax.set_xticks(range(len(KL_SETS)), [KL_LABEL[e] for e in KL_SETS], rotation=30, ha="right")
        ax.set_yticks(range(len(SOURCES)), [SOURCE_LABEL[s_] for s_ in SOURCES])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if not np.isfinite(M).any():
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
            continue
        with np.errstate(invalid="ignore"):
            rel = M / np.nanmin(M, axis=0, keepdims=True)
        im = ax.imshow(np.clip(rel, 1.0, 4.0), cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]):
                    continue
                dark = norm(min(max(rel[i, j], 1.0), 4.0)) > 0.55
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if dark else INK)
        for j in range(M.shape[1]):
            if np.isfinite(M[:, j]).any():
                i = int(np.nanargmin(M[:, j]))
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        ax.set_xlabel("held-out set")
    axes[0].set_ylabel("calibration source")
    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[1, 1.5, 2, 3, 4])
        cb.ax.set_yticklabels(["1", "1.5", "2", "3", "$\\geq$4"])
        cb.set_label("KL / best KL in column", fontsize=7.5)
        cb.outline.set_visible(False)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_domains(df, dense, out, ratio=0.5, models=MODELS, sources=SOURCES, figsize=(7.2, 2.9)):
    """Perplexity on each SlimPajama domain relative to the dense model (shared log scale)."""
    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, models):
        d = dense[dense.model == m]
        M = []
        for s_ in sources:
            r = main_run(df, m, s_, ratio)
            M.append([np.nan if (r is None or d.empty) else
                      r.get(f"final_sp_{dom}", np.nan) / d.iloc[0][f"sp_{dom}"] for dom in SP_DOMAINS])
        heat(ax, M, [SOURCE_LABEL[s_] for s_ in sources], [SP_LABEL[x] for x in SP_DOMAINS],
             fmt_s="{:.1f}", vmin=1.0, vmax=40.0, log=True, fontsize=5.8)
        ax.set_title(MODEL_LABEL.get(m, m))
    axes[0].set_ylabel("calibration source")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def table_bos(df, dense, out, model="llama-3.2-1b", ratio=0.5):
    """WikiText-2 perplexity on BOS-prefixed vs GPTQ-stream windows, before and after training."""
    R = rtag(ratio)
    stats = load_stats().get(model, {})
    lines = ["\\textbf{Calibration} & \\textbf{BOS/window} & \\multicolumn{2}{c}{\\textbf{Untrained}} & "
             "\\multicolumn{2}{c}{\\textbf{Trained}} \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             " & & BOS & stream & BOS & stream \\\\", "\\midrule"]
    # The dense model has no calibration windows and no training: its two perplexities
    # are given in the caption rather than in a row of not-applicable cells.
    rows = [(SOURCE_LABEL[s_], f"proj-{s_}-{R}-s0", s_) for s_ in SOURCES]
    rows += [("C4, BOS per document", f"proj-c4-{R}-s0-docbos", "c4:docbos"),
             ("C4, no BOS", f"proj-c4-{R}-s0-nobos", "c4:nobos")]
    for label, run, src in rows:
        r = pick(df, model, run)
        if r is None:
            continue
        def g(k):
            v = r.get(k)
            return None if v is None or pd.isna(v) else v
        if src == "c4:docbos":     # one BOS per packed document
            nb = stats.get("c4", {}).get("docs_per_window")
        elif src == "c4:nobos":
            nb = 0.0
        else:
            nb = stats.get(src, {}).get("bos_per_window") if src else None
        nbs = "--" if nb is None else f"{nb:.1f}"
        lines.append(f"{label} & {nbs} & {fmt(g('init_ppl_wikitext2'), 1)} & "
                     f"{fmt(g('init_pplstream_wikitext2'), 1)} & {fmt(g('final_ppl_wikitext2'), 1)} & "
                     f"{fmt(g('final_pplstream_wikitext2'), 1)} \\\\")
    write(out, "\n".join(lines) + "\n")


def kl_relative(df, model, ratio):
    """Rows: sources; columns: held-out sets; values: KL / best KL in the column."""
    M = np.array([[np.nan if (r := main_run(df, model, s_, ratio)) is None
                   else r.get(f"final_kl_{e}", np.nan) for e in KL_SETS] for s_ in SOURCES],
                 dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return M / np.nanmin(M, axis=0, keepdims=True)


def table_klsummary(df, out):
    """Geometric mean and maximum of the relative KL over the five held-out sets."""
    cols = [(m, r) for r in (0.3, 0.5) for m in MODELS]
    short = {"opt-1.3b": "OPT", "llama-3.2-1b": "Llama", "qwen3-1.7b": "Qwen3"}
    n = len(MODELS)
    lines = [" & " + " & ".join(f"\\multicolumn{{{2 * n}}}{{c}}{{\\textbf{{$-{int(r * 100)}\\%$}}}}"
                                  for r in (0.3, 0.5)) + " \\\\",
             f"\\cmidrule(lr){{2-{1 + 2 * n}}}\\cmidrule(lr){{{2 + 2 * n}-{1 + 4 * n}}}",
             "\\textbf{Calibration} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{short[m]}}}" for m, _ in cols)
             + " \\\\",
             " & " + " & ".join("geo & max" for _ in cols) + " \\\\", "\\midrule"]
    vals = {}
    for m, r in cols:
        rel = kl_relative(df, m, r)
        complete = np.isfinite(rel).all(axis=1)
        geo = np.where(complete, np.exp(np.nanmean(np.log(rel), axis=1)), np.nan)
        mx = np.where(complete, np.nanmax(rel, axis=1), np.nan)
        vals[(m, r)] = (geo, mx)
    for i, s_ in enumerate(SOURCES):
        cells = []
        for key in cols:
            geo, mx = vals[key]
            best_g = np.nanmin(geo) if np.isfinite(geo).any() else None
            best_m = np.nanmin(mx) if np.isfinite(mx).any() else None
            g, x = geo[i], mx[i]
            cells.append(fmt(None if np.isnan(g) else g, 2, bold=(best_g is not None and g == best_g)))
            cells.append(fmt(None if np.isnan(x) else x, 2, bold=(best_m is not None and x == best_m)))
        lines.append(f"{SOURCE_LABEL[s_]} & " + " & ".join(cells) + " \\\\")
    write(out, "\n".join(lines) + "\n")


def table_oblique(df, out, ratios=(0.3, 0.5)):
    """Training-free: orthogonal recast (LSP init) vs oblique whitened truncation (SVD-LLM)."""
    short = {"opt-1.3b": "OPT-1.3B", "llama-3.2-1b": "Llama-3.2-1B", "qwen3-1.7b": "Qwen3-1.7B"}
    lines = ["\\textbf{Calibration} & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{short[m]}}}}}"
                                                      for m in MODELS) + " \\\\",
             " ".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(MODELS))),
             " & " + " & ".join("\\multicolumn{2}{c}{C4} & \\multicolumn{2}{c}{SP}" for _ in MODELS) + " \\\\",
             " & " + " & ".join("rec. & obl. & rec. & obl." for _ in MODELS) + " \\\\", "\\midrule"]
    for ratio in ratios:
        lines.append(f"\\multicolumn{{{1 + 4 * len(MODELS)}}}{{l}}{{\\textit{{$-{int(ratio * 100)}\\%$}}}} \\\\")
        for s_ in SOURCES:
            cells = []
            for m in MODELS:
                rec = main_run(df, m, s_, ratio)
                obl = main_run(df, m, s_, ratio, mode="none")
                for key in ("ppl_c4", "sp_geo"):
                    for row in (rec, obl):
                        v = None if row is None else row.get(f"init_{key}")
                        cells.append(fmt(None if v is None or pd.isna(v) else v, 1))
            lines.append(f"{SOURCE_LABEL[s_]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    write(out, "\n".join(lines[:-1]) + "\n")


def fig_dumbbell(df, dense, out, ratio=0.3):
    """Perplexity before (hollow) and after (filled) training, per calibration source."""
    metrics = [("ppl_wikitext2", "WikiText-2 perplexity"), ("ppl_c4", "C4 perplexity")]
    fig, axes = plt.subplots(len(metrics), len(MODELS), figsize=(7.0, 3.6), sharex=True)
    for j, m in enumerate(MODELS):
        d = dense[dense.model == m] if not dense.empty else dense
        for i, (key, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for x, s_ in enumerate(SOURCES):
                r = main_run(df, m, s_, ratio)
                if r is None:
                    continue
                a, b = r.get(f"init_{key}"), r.get(f"final_{key}")
                if a is None or pd.isna(a) or b is None or pd.isna(b):
                    continue
                c = PALETTE[s_]
                ax.plot([x, x], [a, b], color=c, lw=1.6, alpha=0.6)
                ax.plot([x], [a], marker=MARKER[s_], ms=5, mfc="white", mec=c, mew=1.4, ls="none")
                ax.plot([x], [b], marker=MARKER[s_], ms=5, color=c, ls="none")
            if not d.empty:
                ax.axhline(d.iloc[0][key], color=INK2, lw=1.0, ls="--")
            ax.set_yscale("log")
            from matplotlib.ticker import FuncFormatter, NullFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(axis="y", color=GRID, lw=0.6, which="major")
            if i == 0:
                ax.set_title(MODEL_LABEL[m])
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(SOURCES)), [SOURCE_LABEL[s_] for s_ in SOURCES],
                          rotation=40, ha="right")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", mfc="white", mec=INK2, ls="none", label="untrained (whitened init.)"),
               Line2D([], [], marker="o", color=INK2, ls="none", label="trained (LSP)"),
               Line2D([], [], color=INK2, ls="--", label="dense")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


KIND_LABEL = {"qkv": "Q/K/V", "o": "output", "gateup": "gate/up", "fc1": "fc1", "down": "down",
              "fc2": "fc2"}


def fig_subspaces(out: str, ratio=0.5, tables_dir="records/subspaces"):
    """Overlap of removed subspaces: across sources (before / after training) and within a run."""
    frames = {}
    for m in MODELS:
        p = os.path.join(tables_dir, f"subspaces-{m}-{rtag(ratio)}.csv")
        if os.path.exists(p):
            frames[m] = pd.read_csv(p)
    if not frames:
        return
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.3), sharey=True)
    # Untrained hollow, trained filled, as in the dumbbell figure: where the two
    # across-source curves coincide, the ring of the untrained marker stays visible.
    styles = {"init-init": ("#2a78d6", "o", "white", "across sources, untrained"),
              "final-final": ("#eb6834", "s", None, "across sources, trained"),
              "init-final": ("#1baf7a", "^", None, "same run, untrained vs trained")}
    lows = []
    for ax, m in zip(axes, MODELS):
        ax.set_title(MODEL_LABEL[m])
        if m not in frames:
            ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center", color=INK2)
            continue
        d = frames[m]
        nb = d.block.max() + 1
        for pair, (col, mk, mfc, lab) in styles.items():
            g = d[d.pair == pair].groupby("block")["overlap"].mean()
            ax.plot((g.index + 0.5) / nb, g.values, color=col, marker=mk, markersize=3.5, lw=1.4,
                    mfc=mfc or col, mew=1.0, label=lab, zorder=3 if mfc else 2)
            lows.append(g.min())
        ch = d.groupby("block")["chance"].mean()
        ax.plot((ch.index + 0.5) / nb, ch.values, color=INK2, lw=1.0, ls="--", label="chance $k/d$")
        lows.append(ch.min())
        ax.set_xlabel("relative depth")
        ax.grid(axis="y", color=GRID, lw=0.6)
    # Overlap is bounded by 1 and read against chance: the axis spans the data, from
    # just below the chance line to 1, rather than from 0.
    lo = math.floor((min(lows) - 0.02) * 10) / 10 if lows else 0.0
    axes[0].set_ylim(lo, 1.0)
    axes[0].set_yticks(np.arange(lo, 1.0001, 0.1))
    axes[0].set_ylabel("subspace overlap")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = next((a.get_legend_handles_labels() for a in axes
                                if a.get_legend_handles_labels()[0]), ([], []))
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def table_domain(df, dense, out, model="llama-3.2-1b", ratio=0.5):
    """Single-domain calibration: perplexity on each SlimPajama domain relative to dense."""
    R = rtag(ratio)
    d = dense[dense.model == model].iloc[0]
    rows = [("SlimPajama (all)", f"proj-slimpajama-{R}-s0"), ("GitHub", f"proj-sp-github-{R}-s0"),
            ("arXiv", f"proj-sp-arxiv-{R}-s0"), ("Books", f"proj-sp-book-{R}-s0")]
    vals = []
    for _, run in rows:
        r = pick(df, model, run)
        rel = [None if r is None else r.get(f"final_sp_{x}") / d[f"sp_{x}"] for x in SP_DOMAINS]
        acc = None if r is None else 100 * r.get("final_zs_mean_acc")
        vals.append(rel + [acc])
    M = np.array([[np.nan if v is None else v for v in row] for row in vals], dtype=float)
    best = [np.nanargmax(M[:, j]) if j == M.shape[1] - 1 else np.nanargmin(M[:, j])
            for j in range(M.shape[1])]
    head = " & ".join(f"\\textbf{{{SP_LABEL[x]}}}" for x in SP_DOMAINS)
    lines = [f"\\textbf{{Calibration}} & {head} & \\textbf{{Acc}} \\\\", "\\midrule"]
    for i, (label, _) in enumerate(rows):
        cells = [fmt(v, 1 if j == M.shape[1] - 1 else 2, bold=(best[j] == i))
                 for j, v in enumerate(M[i])]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    write(out, "\n".join(lines) + "\n")


def fig_size(df, out, model="llama-3.2-1b", ratio=0.5):
    """Calibration-set size at a fixed budget of 512 steps: C4 against self-generated documents."""
    R = rtag(ratio)
    ns = [64, 128, 256, 512, 1024, 4096]
    # 512 is the main run; at 4096 every window is seen once (the self-generated
    # windows come from a larger pool, plan_fresh).
    tags = {512: "", 4096: {"c4": "-n4096", "selfdoc": "-fresh4096"}}
    panels = [("final_ppl_wikitext2", "WikiText-2 perplexity", 1.0, [40, 60, 80, 100]),
              ("final_ppl_c4", "C4 perplexity", 1.0, [40, 50, 60]),
              ("final_sp_geo", "SlimPajama perplexity", 1.0, [30, 50, 100, 200]),
              ("final_zs_mean_acc", "mean accuracy (%)", 100.0, None)]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0))
    for ax, (key, lab, scale, ticks) in zip(axes, panels):
        for s_ in ("c4", "selfdoc"):
            ys = []
            for n in ns:
                tag = tags.get(n, f"-n{n}")
                tag = tag[s_] if isinstance(tag, dict) else tag
                r = pick(df, model, f"proj-{s_}-{R}-s0{tag}")
                ys.append(np.nan if r is None else scale * r.get(key))
            ax.plot(ns, ys, color=PALETTE[s_], marker=MARKER[s_], markersize=4, lw=1.4,
                    label=SOURCE_LABEL[s_])
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 256, 1024, 4096])
        ax.set_xticklabels(["64", "256", "1024", "4096"])
        ax.minorticks_off()
        if ticks is not None:
            ax.set_yscale("log")
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks])
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(lab)
        ax.set_xlabel("distinct windows")
        ax.grid(axis="y", color=GRID, lw=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def table_scale(df, dense, out, model="llama-3.1-8b", ratio=0.3,
                sources=("wikitext2", "c4", "slimpajama", "alpaca", "selfdoc")):
    """Scale check: untrained and trained perplexities, accuracy and relative KL summary."""
    rows = {s_: main_run(df, model, s_, ratio) for s_ in sources}
    if all(r is None for r in rows.values()):
        return
    M = np.array([[np.nan if r is None else r.get(f"final_kl_{e}", np.nan) for e in KL_SETS]
                  for r in rows.values()], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = M / np.nanmin(M, axis=0, keepdims=True)
    complete = np.isfinite(rel).all(axis=1)
    geo = np.where(complete, np.exp(np.nanmean(np.log(rel), axis=1)), np.nan)
    mx = np.where(complete, np.nanmax(rel, axis=1), np.nan)
    keys = [("init", "ppl_wikitext2"), ("init", "ppl_c4"), ("final", "ppl_wikitext2"),
            ("final", "ppl_c4"), ("final", "sp_geo"), ("final", "zs_mean_acc")]
    V = []
    for r in rows.values():
        v = [np.nan if r is None else r.get(f"{ph}_{k}", np.nan) for ph, k in keys]
        v[5] = 100 * v[5]
        V.append(v)
    V = np.column_stack([np.array(V, dtype=float), geo, mx])
    best = [np.nanargmax(V[:, j]) if j == 5 else np.nanargmin(V[:, j]) for j in range(V.shape[1])]
    lines = ["\\textbf{Calibration} & \\multicolumn{2}{c}{\\textbf{Untrained}} & "
             "\\multicolumn{4}{c}{\\textbf{Trained}} & \\multicolumn{2}{c}{\\textbf{Rel. KL}} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-7}\\cmidrule(lr){8-9}",
             " & Wiki & C4 & Wiki & C4 & SP & Acc & geo & max \\\\", "\\midrule"]
    # The dense model's values go in the caption: as a row, most of its cells would be
    # not-applicable dashes (no untrained state, no divergence from itself).
    for i, s_ in enumerate(sources):
        cells = [fmt(None if np.isnan(x) else x, 2 if j >= 6 else 1, bold=(best[j] == i))
                 for j, x in enumerate(V[i])]
        lines.append(f"{SOURCE_LABEL[s_]} & " + " & ".join(cells) + " \\\\")
    write(out, "\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", default="records", help="root of the run records")
    ap.add_argument("--results", default="results", help="runs.csv and dense.csv; output root")
    args = ap.parse_args(argv)
    args.tables = os.path.join(args.results, "tables")
    args.figures = os.path.join(args.results, "figures")
    os.makedirs(args.tables, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)
    df = pd.read_csv(os.path.join(args.results, "runs.csv"))
    dense_csv = os.path.join(args.results, "dense.csv")
    dense = pd.read_csv(dense_csv) if os.path.exists(dense_csv) else pd.DataFrame()
    stats = load_stats(os.path.join(args.records, "stats"))
    table_sources(stats, os.path.join(args.tables, "sources.tex"))
    fig_entropy(stats, os.path.join(args.figures, "entropy.pdf"))
    table_main(df, dense, os.path.join(args.tables, "main.tex"))
    table_klsummary(df, os.path.join(args.tables, "klsummary.tex"))
    table_oblique(df, os.path.join(args.tables, "oblique.tex"))
    if not dense.empty:
        for r in (0.3, 0.5):
            fig_dumbbell(df, dense, os.path.join(args.figures, f"dumbbell_{rtag(r)}.pdf"), ratio=r)
    for r in (0.3, 0.5):
        fig_kl_matrix(df, os.path.join(args.figures, f"kl_matrix_{rtag(r)}.pdf"), ratio=r)
        if not dense.empty:
            fig_domains(df, dense, os.path.join(args.figures, f"domains_{rtag(r)}.pdf"), ratio=r)
    table_lr(os.path.join(args.records, "runs"), os.path.join(args.tables, "lr.tex"))
    if not dense.empty:
        table_bos(df, dense, os.path.join(args.tables, "bos.tex"))
    for r in (0.3, 0.5):
        fig_subspaces(os.path.join(args.figures, f"subspaces_{rtag(r)}.pdf"), ratio=r,
                      tables_dir=os.path.join(args.records, "subspaces"))
    table_repro(df, dense, os.path.join(args.tables, "repro.tex"))
    table_ablations(df, os.path.join(args.tables, "ablations.tex"))
    table_alloc(df, os.path.join(args.tables, "alloc.tex"))
    table_factors(df, os.path.join(args.tables, "factors.tex"))
    table_seeds(df, os.path.join(args.tables, "seeds.tex"))
    if not dense.empty:
        table_tasks(df, dense, os.path.join(args.tables, "tasks.tex"))
        table_domain(df, dense, os.path.join(args.tables, "domain.tex"))
    fig_size(df, os.path.join(args.figures, "size.pdf"))
    table_scale(df, dense, os.path.join(args.tables, "scale.tex"))
    scale_sources = ("wikitext2", "c4", "slimpajama", "alpaca", "selfdoc")
    if all(main_run(df, "llama-3.1-8b", s_, 0.3) is not None for s_ in scale_sources):
        fig_domains(df, dense, os.path.join(args.figures, "domains_8b.pdf"), ratio=0.3,
                    models=["llama-3.1-8b"], sources=scale_sources, figsize=(3.6, 2.4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
