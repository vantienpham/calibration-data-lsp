"""Experiment plans: every job behind the paper's results.

A plan is a list of jobs; ``scripts/submit.py PLAN [PLAN ...]`` runs them, on Slurm
or locally (``--local``). A job is a dict with

    name       unique job name (also the Slurm job name)
    script     the entry point, e.g. scripts/compress.py
    args       its command-line arguments
    run_dir    for compress.py jobs: out/runs/<model>/<run>; or
    output     for other jobs: the file the job produces
    requires   files that must exist before the job can start (optional)
    time       Slurm time limit
    partitions Slurm partitions (see below)

The arguments of every job are exactly those of the published run in
``records/runs/<model>/<run>/config.json``; ``tests/test_plans.py`` checks it.

Slurm placement is read from the environment, so the plans carry no
cluster-specific names:

    FSC_PARTITIONS       partitions for most jobs: bfloat16 GPUs with >= 40 GB
    FSC_PARTITIONS_80G   partitions for Llama-3.1-8B: >= 80 GB (default: FSC_PARTITIONS)
    FSC_EXCLUDE          nodes to avoid (optional)

Empty values let Slurm use its default partition.
"""

from __future__ import annotations

import os

PARTITIONS = os.environ.get("FSC_PARTITIONS", "")
PARTITIONS_80G = os.environ.get("FSC_PARTITIONS_80G", PARTITIONS)
EXCLUDE = os.environ.get("FSC_EXCLUDE", "")

MODELS = {
    "opt-125m": "facebook/opt-125m",
    "opt-1.3b": "facebook/opt-1.3b",
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-Base",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
}

SOURCES = ("wikitext2", "c4", "slimpajama", "alpaca", "selfgen", "selfdoc")
MAIN_MODELS = ("opt-1.3b", "llama-3.2-1b", "qwen3-1.7b")
RATIOS = (0.3, 0.5)
# Projector learning rate per model, selected on the validation KL of C4 calibration
# (plan pilot-lr at -50%, Appendix B; plan scale at -30% for Llama-3.1-8B, Section 4.8).
LR = {"opt-125m": 1e-3, "opt-1.3b": 1e-4, "llama-3.2-1b": 1e-4, "qwen3-1.7b": 1e-4,
      "llama-3.1-8b": 3e-5}
# The training budget of the main experiments: 512 windows of 2048 tokens, batches of
# 8 windows, 8 epochs (512 steps).
COMMON = ["--n-train", "512", "--n-val", "32", "--batch-size", "8", "--micro-batch", "2",
          "--epochs", "8", "--seqlen", "2048"]


def _r(ratio: float) -> str:
    return f"r{int(round(ratio * 100))}"


def _lr(lr: float) -> str:
    return f"lr{lr:.0e}".replace("e-0", "e-")


def run_dir(model: str, name: str) -> str:
    return f"out/runs/{model}/{name}"


def compress_job(model: str, name: str, args: list[str], time: str = "0-04:00:00",
                 partitions: str | None = None, requires: list[str] | None = None) -> dict:
    return {
        "name": f"fsc-{model}-{name}",
        "script": "scripts/compress.py",
        "args": ["--model", MODELS[model]] + args,
        "run_dir": run_dir(model, name),
        "partitions": PARTITIONS if partitions is None else partitions,
        "time": time,
        "requires": requires or [],
    }


# ----------------------------------------------------------------------------
# Self-generated calibration sets
# ----------------------------------------------------------------------------

GEN_BATCH = {"opt-125m": 128, "opt-1.3b": 64, "llama-3.2-1b": 128, "qwen3-1.7b": 64,
             "llama-3.1-8b": 64}
DOC_BATCH = {"opt-125m": 256, "opt-1.3b": 128, "llama-3.2-1b": 256, "qwen3-1.7b": 128,
             "llama-3.1-8b": 128}


def selfgen_path(model: str, temperature: float = 1.0, seed: int = 0, n: int = 1088) -> str:
    """Exact 2048-token samples of the dense model (1088 = 1024 + 64 windows)."""
    tag = f"selfgen-T{temperature:.1f}-s{seed}" + ("" if n == 1088 else f"-n{n}")
    return f"out/calib/{model}/{tag}.pt"


def selfgen_eval_path(model: str) -> str:
    """Exact samples reserved for measuring KL (seed 1000); never used for calibration."""
    return selfgen_path(model, 1.0, 1000, 64)


def selfdoc_path(model: str, seed: int = 0) -> str:
    """Independent self-generated documents of at most 512 tokens (2.3 M tokens)."""
    return f"out/calib/{model}/selfdoc-T1.0-s{seed}.pt"


def selfgen_job(model: str, temperature: float = 1.0, seed: int = 0, n: int = 1088,
                time: str = "0-03:00:00", partitions: str | None = None) -> dict:
    out = selfgen_path(model, temperature, seed, n)
    return {
        "name": f"fsc-gen-{model}-T{temperature:.1f}-s{seed}" + ("" if n == 1088 else f"-n{n}"),
        "script": "scripts/generate_calib.py",
        "args": ["--model", MODELS[model], "--n", str(n), "--seqlen", "2048",
                 "--temperature", str(temperature), "--seed", str(seed),
                 "--batch-size", str(GEN_BATCH[model]), "--out", out],
        "output": out,
        "partitions": PARTITIONS if partitions is None else partitions,
        "time": time,
    }


def selfdoc_job(model: str, seed: int = 0, n_tokens: int = 2_300_000, out: str | None = None,
                partitions: str | None = None) -> dict:
    out = out or selfdoc_path(model, seed)
    return {
        "name": f"fsc-gendoc-{model}-s{seed}" + ("" if n_tokens == 2_300_000 else "-big"),
        "script": "scripts/generate_calib.py",
        "args": ["--model", MODELS[model], "--documents", "--n-tokens", str(n_tokens),
                 "--max-doc-len", "512", "--temperature", "1.0", "--seed", str(seed),
                 "--batch-size", str(DOC_BATCH[model]), "--out", out],
        "output": out,
        "partitions": PARTITIONS if partitions is None else partitions,
        "time": "0-04:00:00",
    }


def plan_selfgen() -> list[dict]:
    """Exact self-samples: evaluation pools, calibration pools, seeds and temperatures."""
    jobs = [selfgen_job(m, 1.0, 1000, n=64, time="0-01:00:00") for m in MAIN_MODELS]
    jobs += [selfgen_job(m) for m in ("opt-125m",) + MAIN_MODELS]
    jobs += [selfgen_job("llama-3.2-1b", 1.0, s) for s in (1, 2)]
    jobs += [selfgen_job("llama-3.2-1b", t, 0) for t in (0.7, 1.3)]
    return jobs


def plan_selfdoc() -> list[dict]:
    """Self-generated documents for every main model, and two more seeds on Llama-3.2-1B."""
    return [selfdoc_job(m) for m in MAIN_MODELS] + [selfdoc_job("llama-3.2-1b", s) for s in (1, 2)]


# ----------------------------------------------------------------------------
# Main study
# ----------------------------------------------------------------------------

def _source_args(model: str, source: str, seed: int = 0, temperature: float = 1.0):
    if source == "selfdoc":
        path = selfdoc_path(model, seed)
        return ["--source", "selfdoc", "--selfgen-path", path], [path]
    if source != "selfgen":
        return ["--source", source], []
    path = selfgen_path(model, temperature, seed)
    return ["--source", "selfgen", "--selfgen-path", path], [path]


def main_job(model, source, ratio, seed=0, mode="proj", extra=(), tag="", temperature=1.0,
             time="0-08:00:00", lr=None, partitions=None) -> dict:
    src, req = _source_args(model, source, seed, temperature)
    sname = source if temperature == 1.0 else f"{source}T{temperature:.1f}"
    name = f"{mode}-{sname}-{_r(ratio)}-s{seed}{tag}"
    kl_pool = selfgen_eval_path(model)
    args = src + ["--ratio", str(ratio), "--seed", str(seed), "--mode", mode,
                  "--kl-selfgen-path", kl_pool] + COMMON
    req = req + [kl_pool]
    if mode == "proj":
        args += ["--lr", str(lr or LR[model])]
    return compress_job(model, name, args + list(extra), time=time, requires=req,
                        partitions=partitions)


def plan_dense() -> list[dict]:
    """The uncompressed models (reference rows of every table)."""
    return [compress_job(m, "dense", ["--mode", "dense"], time="0-02:00:00")
            for m in ("opt-125m",) + MAIN_MODELS]


def plan_main() -> list[dict]:
    """Calibration source x model x ratio, learned projectors (Tables 3-4, Figures 2-4)."""
    return [main_job(m, s, r, extra=["--save-bases"])
            for r in RATIOS for m in MAIN_MODELS for s in SOURCES]


def plan_oblique() -> list[dict]:
    """The training-free oblique whitened truncation of SVD-LLM per source (Table 2)."""
    return [main_job(m, s, r, mode="none", extra=["--init", "oblique"], time="0-03:00:00")
            for r in RATIOS for m in MAIN_MODELS for s in SOURCES]


def plan_subspaces() -> list[dict]:
    """Overlaps of the removed subspaces across sources (Figure A4; needs plan main)."""
    jobs = []
    for m in MAIN_MODELS:
        for r in RATIOS:
            runs = [f"proj-{s}-{_r(r)}-s0" for s in SOURCES]
            out = f"out/subspaces/subspaces-{m}-{_r(r)}.csv"
            jobs.append({"name": f"fsc-subspaces-{m}-{_r(r)}", "script": "scripts/analyze_subspaces.py",
                         "args": ["--model", m, "--runs"] + runs + ["--out", out],
                         "output": out, "partitions": PARTITIONS, "time": "0-01:00:00",
                         "requires": [f"{run_dir(m, x)}/DONE" for x in runs]})
    return jobs


def plan_stats() -> list[dict]:
    """Documents per window, NLL and entropy of every source under each model (Table 1)."""
    jobs = []
    for m in MAIN_MODELS:
        out = f"out/stats/sources-{m}.json"
        sg, sd = selfgen_path(m), selfdoc_path(m)
        jobs.append({"name": f"fsc-stats-{m}", "script": "scripts/source_stats.py",
                     "args": ["--model", MODELS[m], "--selfgen-path", sg, "--selfdoc-path", sd,
                              "--out", out],
                     "output": out, "partitions": PARTITIONS, "time": "0-01:00:00",
                     "requires": [sg, sd]})
    return jobs


# ----------------------------------------------------------------------------
# Ablations (Llama-3.2-1B at -50% unless stated)
# ----------------------------------------------------------------------------

def plan_seeds() -> list[dict]:
    """Seeds 1 and 2 of every source (Table 9; seed 0 is plan main)."""
    return [main_job("llama-3.2-1b", s, 0.5, seed=sd) for sd in (1, 2) for s in SOURCES]


def plan_size() -> list[dict]:
    """Distinct calibration windows at a fixed budget of 512 steps (Figure 5)."""
    jobs = [main_job("llama-3.2-1b", s, 0.5, tag=f"-n{n}",
                     extra=["--n-train", str(n), "--epoch-size", "512"])
            for n in (64, 128, 256, 1024) for s in ("c4", "selfdoc")]
    # 4096 distinct windows, each seen once; the self-generated counterpart needs a
    # larger pool and is plan fresh.
    jobs.append(main_job("llama-3.2-1b", "c4", 0.5, tag="-n4096",
                         extra=["--n-train", "4096", "--epoch-size", "512"]))
    return jobs


def plan_fresh() -> list[dict]:
    """4096 windows of freshly sampled documents, each seen once (Figure 5)."""
    m = "llama-3.2-1b"
    pool = f"out/calib/{m}/selfdoc-T1.0-s0-big.pt"
    gen = selfdoc_job(m, n_tokens=8_500_000, out=pool)
    run = compress_job(m, "proj-selfdoc-r50-s0-fresh4096", [
        "--source", "selfdoc", "--selfgen-path", pool, "--ratio", "0.5", "--seed", "0",
        "--mode", "proj", "--lr", str(LR[m]), "--kl-selfgen-path", selfgen_eval_path(m)]
        + COMMON + ["--n-train", "4096", "--n-val", "32", "--epoch-size", "512"],
        requires=[pool, selfgen_eval_path(m)])
    return [gen, run]


def plan_temperature() -> list[dict]:
    """Exact self-samples drawn at temperatures 0.7 and 1.3 (Table 6)."""
    return [main_job("llama-3.2-1b", "selfgen", 0.5, temperature=t) for t in (0.7, 1.3)]


def plan_factors() -> list[dict]:
    """Free low-rank factors from the same initialization, two learning rates (Table 8)."""
    return [main_job(m, s, 0.5, mode="factors", tag=f"-{_lr(lr)}", extra=["--lr", str(lr)])
            for m in ("llama-3.2-1b", "opt-1.3b") for s in ("wikitext2", "c4", "selfgen")
            for lr in (1e-5, 1e-4)]


def plan_klalloc() -> list[dict]:
    """The measured-KL rank allocation of LSP, computed on each source (Table 7)."""
    return [main_job("llama-3.2-1b", s, 0.5, tag="-klalloc", extra=["--alloc", "kl"])
            for s in SOURCES]


def plan_domain() -> list[dict]:
    """Calibration on single SlimPajama domains (Table 5)."""
    return [main_job("llama-3.2-1b", s, 0.5) for s in ("sp-github", "sp-arxiv", "sp-book")]


def plan_crossover() -> list[dict]:
    """Initialization statistics from one source, training windows from another (Table 6)."""
    jobs = []
    for init_src, train_src in (("c4", "selfdoc"), ("selfdoc", "c4"),
                                ("wikitext2", "slimpajama"), ("slimpajama", "wikitext2")):
        extra = ["--init-source", init_src]
        if init_src == "selfdoc":
            extra += ["--init-selfgen-path", selfdoc_path("llama-3.2-1b")]
        j = main_job("llama-3.2-1b", train_src, 0.5, tag=f"-init{init_src}", extra=extra)
        if init_src == "selfdoc":
            j["requires"].append(selfdoc_path("llama-3.2-1b"))
        jobs.append(j)
    return jobs


def plan_bos() -> list[dict]:
    """Window format: C4 windows without BOS, and with a BOS at every document start
    (Table 6, Appendix C)."""
    jobs = [main_job("llama-3.2-1b", "c4", 0.5, tag="-nobos", extra=["--no-bos-windows"])]
    for r in RATIOS:
        jobs.append(main_job("llama-3.2-1b", "c4", r, tag="-docbos", extra=["--doc-bos"]))
        jobs.append(main_job("llama-3.2-1b", "c4", r, mode="none", tag="-docbos-oblique",
                             extra=["--doc-bos", "--init", "oblique"], time="0-03:00:00"))
    return jobs


# ----------------------------------------------------------------------------
# Learning rates, reproduction check and scale
# ----------------------------------------------------------------------------

def plan_pilot_lr() -> list[dict]:
    """Learning-rate grid on C4 at -50%, eight epochs (Table A2). The "-e8" suffix marks
    the runs added to complete the grid; every run uses the same eight-epoch budget."""
    jobs = []
    for m in ("opt-1.3b", "qwen3-1.7b"):
        for lr in (3e-5, 1e-4, 3e-4):
            jobs.append((m, f"pilot-proj-c4-r50-{_lr(lr)}", lr))
        jobs.append((m, f"pilot-proj-c4-r50-{_lr(1e-3)}-e8", 1e-3))
    for lr in (3e-5, 1e-4, 3e-4, 1e-3):
        jobs.append(("llama-3.2-1b", f"pilot-proj-c4-r50-{_lr(lr)}-e8", lr))
    return [compress_job(m, name, ["--source", "c4", "--ratio", "0.5", "--mode", "proj",
                                   "--lr", str(lr), "--no-zs", "--skip-init-eval"] + COMMON,
                         time="0-06:00:00") for m, name, lr in jobs]


def plan_pilot_opt125m() -> list[dict]:
    """OPT-125M in the reproduction setting of LSP, four learning rates (10 epochs)."""
    return [compress_job("opt-125m", f"pilot-proj-wikitext2-{_r(ratio)}-{_lr(lr)}", [
        "--source", "wikitext2", "--ratio", str(ratio), "--n-train", "1024",
        "--n-val", "64", "--batch-size", "32", "--micro-batch", "8", "--epochs", "10",
        "--lr", str(lr), "--no-zs"], time="0-02:00:00")
        for ratio in (0.3, 0.5, 0.7) for lr in (1e-4, 3e-4, 1e-3, 3e-3)]


def plan_repro() -> list[dict]:
    """Reproduction of LSP with uniform allocation and WikiText-2 calibration (Table A1)."""
    jobs = []
    for model, epochs, lr, time in (("opt-125m", 40, LR["opt-125m"], "0-04:00:00"),
                                    ("opt-1.3b", 20, LR["opt-1.3b"], "0-12:00:00")):
        for ratio in (0.3, 0.5, 0.7):
            jobs.append(compress_job(model, f"repro-wikitext2-{_r(ratio)}", [
                "--source", "wikitext2", "--ratio", str(ratio), "--n-train", "1024",
                "--n-val", "64", "--batch-size", "32", "--micro-batch", "4",
                "--epochs", str(epochs), "--patience", "100", "--lr", str(lr), "--no-zs",
                "--kl-sources", ""], time=time))
    return jobs


def plan_diag_oblique() -> list[dict]:
    """The oblique truncation on OPT-125M in the reproduction setting (Table A1)."""
    return [compress_job("opt-125m", f"oblique-wikitext2-{_r(r)}", [
        "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
        "--mode", "none", "--init", "oblique", "--no-zs"], time="0-01:00:00")
        for r in (0.3, 0.5, 0.7)]


def plan_repro_init() -> list[dict]:
    """Training-free truncations in the reproduction setting: the oblique truncation on
    OPT-1.3B, and the whitened initialization on BOS-less windows (Table A1)."""
    jobs = []
    for r in (0.3, 0.5, 0.7):
        jobs.append(compress_job("opt-1.3b", f"oblique-wikitext2-{_r(r)}", [
            "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
            "--mode", "none", "--init", "oblique", "--no-zs", "--kl-sources", ""],
            time="0-01:00:00"))
        for m in ("opt-125m", "opt-1.3b"):
            jobs.append(compress_job(m, f"init-nobos-wikitext2-{_r(r)}", [
                "--source", "wikitext2", "--ratio", str(r), "--n-train", "1024", "--n-val", "64",
                "--mode", "none", "--no-bos-windows", "--no-zs", "--kl-sources", ""],
                time="0-01:00:00"))
    return jobs


SCALE = "llama-3.1-8b"
SCALE_ARGS = ["--gram-dtype", "fp32", "--micro-batch", "1", "--eval-batch", "2"]


def plan_scale() -> list[dict]:
    """Llama-3.1-8B at -30%: four learning rates on C4, then five sources (Table 10)."""
    big = PARTITIONS_80G
    jobs = [selfgen_job(SCALE, 1.0, 1000, n=64, time="0-02:00:00", partitions=big),
            selfdoc_job(SCALE, partitions=big),
            compress_job(SCALE, "dense", ["--mode", "dense", "--eval-batch", "2"],
                         time="0-04:00:00", partitions=big)]
    for lr in (3e-6, 1e-5, 3e-5, 1e-4):
        jobs.append(compress_job(SCALE, f"pilot-proj-c4-r30-{_lr(lr)}", [
            "--source", "c4", "--ratio", "0.3", "--mode", "proj", "--lr", str(lr), "--no-zs",
            "--skip-init-eval", "--kl-sources", ""] + COMMON + SCALE_ARGS,
            time="0-12:00:00", partitions=big))
    for src in ("wikitext2", "c4", "slimpajama", "alpaca", "selfdoc"):
        jobs.append(main_job(SCALE, src, 0.3, extra=SCALE_ARGS, time="0-16:00:00", partitions=big))
    return jobs


def plan_smoke() -> list[dict]:
    """A two-minute end-to-end check of the installation (OPT-125M); not in the paper."""
    return [compress_job("opt-125m", "smoke", [
        "--source", "wikitext2", "--ratio", "0.5", "--n-train", "64", "--n-val", "8",
        "--epochs", "2", "--zs-limit", "200"], time="0-00:30:00")]


PLANS = {
    "smoke": plan_smoke,
    "dense": plan_dense,
    "selfgen": plan_selfgen,
    "selfdoc": plan_selfdoc,
    "stats": plan_stats,
    "pilot-lr": plan_pilot_lr,
    "pilot-opt125m": plan_pilot_opt125m,
    "main": plan_main,
    "oblique": plan_oblique,
    "subspaces": plan_subspaces,
    "seeds": plan_seeds,
    "size": plan_size,
    "fresh": plan_fresh,
    "temperature": plan_temperature,
    "factors": plan_factors,
    "klalloc": plan_klalloc,
    "domain": plan_domain,
    "crossover": plan_crossover,
    "bos": plan_bos,
    "repro": plan_repro,
    "diag-oblique": plan_diag_oblique,
    "repro-init": plan_repro_init,
    "scale": plan_scale,
}

# Every plan behind the paper, in an order that satisfies the dependencies
# (self-generated pools before the runs that read them, plan main before subspaces).
PAPER = ["dense", "selfgen", "selfdoc", "stats", "pilot-lr", "pilot-opt125m", "main",
         "oblique", "subspaces", "seeds", "size", "fresh", "temperature", "factors",
         "klalloc", "domain", "crossover", "bos", "repro", "diag-oblique", "repro-init",
         "scale"]


def get(name: str) -> list[dict]:
    if name == "all":
        return [job for plan in PAPER for job in PLANS[plan]()]
    if name not in PLANS:
        raise KeyError(f"unknown plan {name!r}; have {sorted(PLANS)} or 'all'")
    return PLANS[name]()
