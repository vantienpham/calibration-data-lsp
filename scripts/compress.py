#!/usr/bin/env python3
"""One compression run, end to end.

    uv run --no-sync python scripts/compress.py --model meta-llama/Llama-3.2-1B \
        --source c4 --ratio 0.3 --run-dir out/runs/llama-3.2-1b/proj-c4-r30-s0

Phases: calibration windows -> Gram matrices -> whitened bases -> allocation ->
evaluation of the training-free initialization (merged) -> training (projectors
or free factors) -> evaluation of the trained model (merged).

Every phase records its result in ``<run-dir>/metrics.json`` (or its own file)
and is skipped when that result exists, so a resubmitted job resumes where the
previous one stopped. Training itself checkpoints every epoch.

Modes:
    dense    evaluate the uncompressed model only
    none     training-free whitened truncation only (the initialization, "NoLSP")
    proj     learned subspace projections (weights frozen)
    factors  free low-rank factors trained from the same initialization
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager, nullcontext

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fsc import alloc, arch, data, evaluate, hub, project, stats, train  # noqa: E402
from fsc import init as finit  # noqa: E402
from fsc.utils import Timer, env_info, log, read_json, write_json  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["dense", "none", "proj", "factors"], default="proj")
    ap.add_argument("--source", default="c4")
    ap.add_argument("--selfgen-path", default=None)
    ap.add_argument("--init-source", default=None,
                    help="source of the Gram statistics (initialization); default: --source")
    ap.add_argument("--init-selfgen-path", default=None)
    ap.add_argument("--doc-bos", action="store_true",
                    help="start every packed document with BOS, not only every window")
    ap.add_argument("--no-bos-windows", action="store_true",
                    help="calibrate on plain chunks without a leading BOS (window-format ablation)")
    ap.add_argument("--ratio", type=float, default=0.3, help="fraction of compressible weights removed")
    ap.add_argument("--alloc", choices=["uniform", "kl"], default="uniform")
    ap.add_argument("--alloc-from", default=None, help="reuse alloc.json of another run")
    ap.add_argument("--init", choices=["recast", "oblique"], default="recast",
                    help="training-free evaluation: orthogonal recast (the LSP initialization) or "
                         "the oblique whitened truncation itself (SVD-LLM); oblique needs --mode none")
    ap.add_argument("--untied", action="store_true", help="no Q/K/V or gate/up tying")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--gram-windows", type=int, default=None, help="default: all training windows")
    ap.add_argument("--gram-batch", type=int, default=4)
    ap.add_argument("--gram-dtype", choices=["fp64", "fp32"], default="fp64",
                    help="Gram accumulator precision; fp32 for 4096-wide models")
    ap.add_argument("--kl-windows", type=int, default=32)
    ap.add_argument("--kl-grid", type=int, default=8, help="removal fractions i/grid, i=1..grid-1")
    ap.add_argument("--kl-batch", type=int, default=4)
    # training
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--drop-p", type=float, default=0.05)
    ap.add_argument("--lam-ort", type=float, default=0.05)
    ap.add_argument("--alpha-epochs", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--epoch-size", type=int, default=None,
                    help="windows per (virtual) epoch; default: --n-train. With a fixed "
                         "epoch size the step budget no longer depends on --n-train")
    ap.add_argument("--seed", type=int, default=0)
    # evaluation
    ap.add_argument("--eval-ppl", default="wikitext2,c4")
    ap.add_argument("--eval-sp", default="cc,c4,github,book,arxiv,wiki,stack")
    ap.add_argument("--sp-windows", type=int, default=32)
    ap.add_argument("--eval-seqlen", type=int, default=2048)
    ap.add_argument("--eval-batch", type=int, default=4)
    ap.add_argument("--zs-tasks", default=",".join(evaluate.ZS_TASKS))
    ap.add_argument("--zs-batch", type=int, default=16)
    ap.add_argument("--zs-limit", type=int, default=None)
    ap.add_argument("--no-zs", action="store_true")
    ap.add_argument("--skip-init-eval", action="store_true")
    ap.add_argument("--save-bases", action="store_true", help="store the initial removed bases (fp16)")
    ap.add_argument("--kl-sources", default="wikitext2,c4,slimpajama,alpaca,selfgen",
                    help="held-out sets on which KL(dense || compressed) is measured")
    ap.add_argument("--kl-eval-windows", type=int, default=32)
    ap.add_argument("--kl-selfgen-path", default=None,
                    help="self-generated pool reserved for evaluation (never used for calibration)")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    ap.add_argument("--reeval-zs", action="store_true",
                    help="re-run only the zero-shot evaluation of a finished run, with "
                         "--zs-batch; every other metric is kept (see reevaluate_zero_shot)")
    return ap.parse_args(argv)


def _csv(s: str) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def load_model(name: str, dtype, device):
    tok = hub.load_tokenizer(name)
    model = hub.load_causal_lm(name, dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def evaluate_all(model, tok, args, device, streams, sp_windows) -> dict:
    """``streams[name] = (stream_without_bos, stream_gptq)``; see evaluate.perplexity."""
    torch.cuda.empty_cache()
    out = {"ppl": {}, "ppl_stream": {}, "sp_ppl": {}}
    bos = tok.bos_token_id
    for name, (plain, gptq) in streams.items():
        out["ppl"][name] = evaluate.perplexity(model, plain, args.eval_seqlen, args.eval_batch,
                                               device, bos=bos)
        if bos is not None:
            out["ppl_stream"][name] = evaluate.perplexity(model, gptq, args.eval_seqlen,
                                                          args.eval_batch, device)
        else:
            out["ppl_stream"][name] = out["ppl"][name]
        log(f"    ppl {name}: {out['ppl'][name]:.3f} (stream {out['ppl_stream'][name]:.3f})")
    for sub, w in sp_windows.items():
        out["sp_ppl"][sub] = evaluate.window_perplexity(model, w, args.eval_batch, device)
    if sp_windows:
        log("    sp ppl " + " ".join(f"{k}={v:.2f}" for k, v in out["sp_ppl"].items()))
    if not args.no_zs:
        out.update(zero_shot_metrics(model, tok, args))
    return out


ZS_KEYS = ("zs", "zs_mean_acc", "zs_mean_mixed")


def zero_shot_metrics(model, tok, args) -> dict:
    """Zero-shot accuracy on --zs-tasks: per task, and the mean (plain accuracy, and
    acc_norm where the task has it)."""
    tasks = _csv(args.zs_tasks)
    zs = evaluate.zero_shot(model, tok, tasks, args.zs_batch, args.zs_limit)
    mixed = [zs[t].get("acc_norm", zs[t]["acc"]) for t in tasks if t in zs]
    out = {"zs": zs, "zs_mean_acc": evaluate.zs_mean(zs, "acc", tasks),
           "zs_mean_mixed": float(sum(mixed) / len(mixed))}
    log(f"    zero-shot mean acc {out['zs_mean_acc']:.4f} (mixed {out['zs_mean_mixed']:.4f})")
    return out


@torch.no_grad()
def merge_check(model, bank, units, bases, windows, device, dtype) -> float:
    """KL between the projected model and its merged factors on a few windows."""
    x = windows[:2].to(device)
    bank.set_active(True)
    a = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
    bank.clear()
    bank.detach(model)
    try:
        with project.merged(model, units, bases, dtype):
            b = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
    finally:
        bank.attach(model)
    return float((a.exp() * (a - b)).sum(-1).mean())


def merge_parts(model, units, bases, dtype=None) -> list:
    """Exact low-rank factors of every unit with a non-empty removed basis, in ``dtype``."""
    parts = []
    for u in units:
        U = bases.get(u.name)
        if U is not None and U.shape[1] > 0:
            parts += project.merge_factors(model, u, U, dtype)[1]
    return parts


def calibration_and_bases(args, model, units, tok, device, timer):
    """Calibration windows, and the whitened bases of every unit from their Gram matrices."""
    with timer("calibration"):
        train_w, val_w = data.calibration_windows(tok, args.source, args.n_train, args.n_val,
                                                  args.seqlen, seed=args.seed,
                                                  selfgen_path=args.selfgen_path,
                                                  use_bos=not args.no_bos_windows,
                                                  doc_bos=args.doc_bos)
    log(f"calibration {args.source}: train {tuple(train_w.shape)}, val {tuple(val_w.shape)}")
    with timer("grams"):
        gw = train_w
        if args.init_source and args.init_source != args.source:
            gw, _ = data.calibration_windows(tok, args.init_source, args.n_train, 1, args.seqlen,
                                             seed=args.seed, selfgen_path=args.init_selfgen_path,
                                             use_bos=not args.no_bos_windows)
            log(f"initialization statistics from {args.init_source}")
        gw = gw if args.gram_windows is None else gw[:args.gram_windows]
        acc = torch.float64 if args.gram_dtype == "fp64" else torch.float32
        grams, n_tok = stats.collect_grams(model, units, gw, args.gram_batch, acc_dtype=acc,
                                           device=device)
        bases = finit.build_bases(model, units, grams)
        del grams
        torch.cuda.empty_cache()
    log(f"Grams on {n_tok} tokens; bases built")
    return train_w, val_w, bases


def oblique_parts(model, units, bases, ks, device, dtype) -> list:
    """Factors of the oblique whitened truncation (SVD-LLM) with the given ranks."""
    parts = []
    for u in units:
        k = ks[u.name]
        if k == 0:
            continue
        if u.side == "in":
            ws = [model.get_submodule(p).weight.detach() for p in u.members]
            A, Bs = bases[u.name].oblique_factors(ws, k, device=device)
            A = A.to(dtype)                  # cast per unit: see project.merge_factors
            parts += [(p, A, B.to(dtype), model.get_submodule(p).bias) for p, B in zip(u.members, Bs)]
        else:
            parts += project.merge_factors(model, u, bases[u.name].removed(k, device=device), dtype)[1]
    return parts


def recast_bases(units, bases, ks, device) -> dict:
    """Removed bases of the orthogonal recast, the initialization of the projectors."""
    return {u.name: bases[u.name].removed(ks[u.name], device=device).float()
            for u in units if ks[u.name] > 0}


@contextmanager
def _active(bank, flag: bool):
    bank.set_active(flag)
    try:
        yield
    finally:
        bank.set_active(True)


@torch.no_grad()
def kl_to_dense(model, kl_sets: dict, teacher, student, batch: int, device) -> dict:
    """Token-averaged KL(dense || compressed) on each held-out set.

    ``teacher`` and ``student`` are zero-argument callables returning context
    managers under which the model computes the dense or the compressed function.
    """
    out = {}
    for name, w in kl_sets.items():
        total, n = 0.0, 0
        for i in range(0, w.shape[0], batch):
            x = w[i:i + batch].to(device)
            with teacher():
                t = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
            with student():
                st = model(input_ids=x, use_cache=False).logits
            total += alloc.kl_sum(t, st)
            n += x.numel()
            del t, st
        out[name] = total / n
    log("    KL to dense " + " ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


def build_kl_sets(tok, args) -> dict:
    sets = {}
    for src in _csv(args.kl_sources):
        if src == "selfgen":
            if not args.kl_selfgen_path or not os.path.exists(args.kl_selfgen_path):
                log("    no self-generated evaluation pool; skipping its KL")
                continue
            sets[src] = data.kl_eval_windows(tok, src, args.kl_eval_windows, args.seqlen,
                                             selfgen_path=args.kl_selfgen_path)
        else:
            sets[src] = data.kl_eval_windows(tok, src, args.kl_eval_windows, args.seqlen)
    return sets


def mark_done(run_dir: str) -> None:
    """Completion marker other jobs can depend on (metrics.json exists before a run ends)."""
    with open(os.path.join(run_dir, "DONE"), "w") as f:
        f.write("done\n")


def main(argv=None) -> int:
    args = parse(argv)
    if args.reeval_zs:
        return reevaluate_zero_shot(args)
    os.makedirs(args.run_dir, exist_ok=True)
    mpath = os.path.join(args.run_dir, "metrics.json")
    metrics = read_json(mpath, {})
    if metrics.get("done"):
        log(f"{args.run_dir}: already done")
        return 0
    write_json(os.path.join(args.run_dir, "config.json"), vars(args))
    envs = read_json(os.path.join(args.run_dir, "env.json"), [])
    envs.append(env_info())
    write_json(os.path.join(args.run_dir, "env.json"), envs)

    device = "cuda"
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    timer = Timer()
    timings = metrics.setdefault("timing", {})

    def save():
        timings.update(timer.t)
        write_json(mpath, metrics)

    log(f"loading {args.model}")
    model, tok = load_model(args.model, dtype, device)
    units = arch.discover_units(model, tie=not args.untied)
    dense_total = arch.total_dense(units)
    log(f"{len(units)} units, {dense_total / 1e6:.1f}M compressible weights")

    streams = {n: (data.eval_stream(tok, n, special=False), data.eval_stream(tok, n))
               for n in _csv(args.eval_ppl)}
    sp_windows = {s: data.eval_windows_sp(tok, s, args.sp_windows, args.eval_seqlen)
                  for s in _csv(args.eval_sp)}

    kl_sets = {} if args.mode == "dense" else build_kl_sets(tok, args)

    if args.mode == "dense":
        if "dense" not in metrics:
            with timer("eval_dense"):
                metrics["dense"] = evaluate_all(model, tok, args, device, streams, sp_windows)
        metrics["done"] = True
        save()
        mark_done(args.run_dir)
        return 0

    train_w, val_w, bases = calibration_and_bases(args, model, units, tok, device, timer)

    # -- allocation -------------------------------------------------------------
    apath = os.path.join(args.run_dir, "alloc.json")
    A = read_json(apath)
    if A is None:
        with timer("alloc"):
            if args.alloc_from:
                A = dict(read_json(args.alloc_from))
                A["copied_from"] = args.alloc_from
            elif args.alloc == "uniform":
                A = {"alloc": "uniform", "ks": alloc.uniform_alloc(units, args.ratio)}
            else:
                cpath = os.path.join(args.run_dir, "kl_curves.json")
                curves = read_json(cpath)
                if curves is None:
                    fracs = [i / args.kl_grid for i in range(1, args.kl_grid)]
                    kw = train_w[: args.kl_windows]
                    raw = alloc.measure_kl_curves(model, units, bases, kw, fracs, args.kl_batch,
                                                  dtype, device, log)
                    curves = {u: {str(k): v for k, v in d.items()} for u, d in raw.items()}
                    write_json(cpath, curves)
                curves_i = {u: {int(k): v for k, v in d.items()} for u, d in curves.items()}
                ks, pred = alloc.greedy_alloc(units, curves_i, args.ratio)
                A = {"alloc": "kl", "ks": ks, "predicted_kl": pred}
        A["realized_ratio"] = arch.realized_ratio(units, A["ks"])
        A["n_dense_units"] = sum(1 for u in units if A["ks"][u.name] == 0)
        write_json(apath, A)
    ks = {k: int(v) for k, v in A["ks"].items()}
    log(f"allocation {A.get('alloc')}: realized ratio {A['realized_ratio']:.4f}, "
        f"{A['n_dense_units']} units dense")
    metrics["realized_ratio"] = A["realized_ratio"]

    if args.init == "oblique":
        if args.mode != "none":
            raise SystemExit("--init oblique is a training-free baseline; use --mode none")
        parts = oblique_parts(model, units, bases, ks, device, dtype)
        del bases
        if "init" not in metrics:
            log("evaluating the oblique whitened truncation")
            with timer("eval_init"):
                with project.swapped(model, parts, dtype):
                    metrics["params"] = {"dense": dense_total,
                                         "stored": project.stored_params(model, units)}
                    metrics["init"] = evaluate_all(model, tok, args, device, streams, sp_windows)
                metrics["init"]["kl"] = kl_to_dense(
                    model, kl_sets, nullcontext, lambda: project.swapped(model, parts, dtype),
                    args.eval_batch, device)
        metrics["done"] = True
        save()
        mark_done(args.run_dir)
        return 0

    init_bases = recast_bases(units, bases, ks, device)
    del bases
    if args.save_bases:
        p = os.path.join(args.run_dir, "bases_init.pt")
        if not os.path.exists(p):
            torch.save({k: v.half().cpu() for k, v in init_bases.items()}, p)

    # -- training-free initialization --------------------------------------------
    if "init" not in metrics and not args.skip_init_eval:
        log("evaluating the initialization (merged)")
        with timer("eval_init"):
            parts = merge_parts(model, units, init_bases, dtype)
            with project.swapped(model, parts, dtype):
                metrics["params"] = {"dense": dense_total,
                                     "stored": project.stored_params(model, units)}
                metrics["init"] = evaluate_all(model, tok, args, device, streams, sp_windows)
            metrics["init"]["kl"] = kl_to_dense(
                model, kl_sets, nullcontext, lambda: project.swapped(model, parts, dtype),
                args.eval_batch, device)
            del parts
        save()

    if args.mode == "none":
        metrics["done"] = True
        save()
        mark_done(args.run_dir)
        return 0

    # -- training ------------------------------------------------------------------
    cfg = train.TrainConfig(mode=args.mode, epochs=args.epochs, lr=args.lr,
                            batch_size=args.batch_size, micro_batch=args.micro_batch,
                            drop_p=args.drop_p, lam_ort=args.lam_ort,
                            alpha_epochs=args.alpha_epochs, warmup_frac=args.warmup_frac,
                            patience=args.patience, seed=args.seed,
                            epoch_size=args.epoch_size)
    if args.mode == "proj":
        bank = project.ProjectorBank(units, init_bases).to(device)
    else:
        bank = project.FactorBank(model, units, init_bases).to(device)
    bank.attach(model)
    n_trainable = sum(p.numel() for p in bank.parameters())
    metrics["trainable_params"] = n_trainable
    log(f"training {args.mode}: {n_trainable / 1e6:.2f}M trainable parameters")

    best_path = os.path.join(args.run_dir, "best_params.pt")
    if "train" not in metrics:
        with timer("train"):
            summary = train.train(model, bank, train_w, val_w, cfg, args.run_dir, device, dtype, log)
        metrics["train"] = summary
        save()
    else:
        bank.load_raw(torch.load(best_path, map_location="cpu"))
        train.eval_mode(bank, dtype)

    # -- trained model, merged ------------------------------------------------------
    if "final" not in metrics:
        log("evaluating the trained model (merged)")
        with timer("eval_final"):
            if args.mode == "proj":
                final_bases = {k: v.to(device) for k, v in bank.export().items()}
                metrics["merge_check_kl"] = merge_check(model, bank, units, final_bases, val_w,
                                                        device, dtype)
                log(f"    merge check KL {metrics['merge_check_kl']:.2e}")
                bank.detach(model)
                parts = merge_parts(model, units, final_bases, dtype)
                with project.swapped(model, parts, dtype):
                    metrics["final"] = evaluate_all(model, tok, args, device, streams, sp_windows)
                metrics["final"]["kl"] = kl_to_dense(
                    model, kl_sets, nullcontext, lambda: project.swapped(model, parts, dtype),
                    args.eval_batch, device)
            else:
                with bank.exported(model, dtype):
                    metrics["params_final"] = project.stored_params(model, units)
                    metrics["final"] = evaluate_all(model, tok, args, device, streams, sp_windows)
                metrics["final"]["kl"] = kl_to_dense(
                    model, kl_sets, lambda: _active(bank, False), lambda: _active(bank, True),
                    args.eval_batch, device)
        save()

    metrics["done"] = True
    save()
    # The optimizer checkpoint only serves a resume; best_params.pt (the trained
    # V or factors) is what reproduces the compressed model.
    state = os.path.join(args.run_dir, "train_state.pt")
    if os.path.exists(state):
        os.remove(state)
    mark_done(args.run_dir)
    log("done")
    return 0


def reevaluate_zero_shot(args) -> int:
    """Re-run only the zero-shot evaluation of a finished run, with --zs-batch.

    Rebuilds every model the run evaluated zero-shot -- the dense model, the
    training-free initialization (from the Gram statistics of the calibration
    windows, as in the run) and the trained model (from best_params.pt) -- and
    replaces their zero-shot results. Every other metric is kept. metrics.json keeps
    the previous results under ``zs_reevaluated``; config.json records the new batch.
    """
    mpath = os.path.join(args.run_dir, "metrics.json")
    cpath = os.path.join(args.run_dir, "config.json")
    metrics, config = read_json(mpath, {}), read_json(cpath, {})
    if not metrics.get("done"):
        raise SystemExit(f"{args.run_dir}: --reeval-zs needs a finished run")
    phases = [p for p in ("dense", "init", "final") if "zs" in (metrics.get(p) or {})]
    if not phases or config.get("zs_batch") == args.zs_batch:
        log(f"{args.run_dir}: nothing to re-evaluate (zero-shot batch {config.get('zs_batch')})")
        return 0
    envs = read_json(os.path.join(args.run_dir, "env.json"), [])
    envs.append(env_info())
    write_json(os.path.join(args.run_dir, "env.json"), envs)

    device, dtype = "cuda", DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    timer = Timer()
    log(f"re-evaluating zero-shot ({', '.join(phases)}) of {args.run_dir} with batch {args.zs_batch}")
    model, tok = load_model(args.model, dtype, device)
    new = {}
    if "dense" in phases:
        with timer("zs_dense"):
            new["dense"] = zero_shot_metrics(model, tok, args)
    else:
        units = arch.discover_units(model, tie=not args.untied)
        _, _, bases = calibration_and_bases(args, model, units, tok, device, timer)
        ks = {k: int(v) for k, v in read_json(os.path.join(args.run_dir, "alloc.json"))["ks"].items()}
        if args.init == "oblique":
            parts = oblique_parts(model, units, bases, ks, device, dtype)
        else:
            init_bases = recast_bases(units, bases, ks, device)
            parts = merge_parts(model, units, init_bases, dtype)
        del bases
        if "init" in phases:
            log("zero-shot of the initialization (merged)")
            with timer("zs_init"), project.swapped(model, parts, dtype):
                new["init"] = zero_shot_metrics(model, tok, args)
        del parts
        if "final" in phases:
            log("zero-shot of the trained model (merged)")
            bank = (project.ProjectorBank(units, init_bases) if args.mode == "proj"
                    else project.FactorBank(model, units, init_bases)).to(device)
            bank.attach(model)
            bank.load_raw(torch.load(os.path.join(args.run_dir, "best_params.pt"), map_location="cpu"))
            train.eval_mode(bank, dtype)
            with timer("zs_final"):
                if args.mode == "proj":
                    final_bases = {k: v.to(device) for k, v in bank.export().items()}
                    bank.detach(model)
                    with project.swapped(model, merge_parts(model, units, final_bases, dtype), dtype):
                        new["final"] = zero_shot_metrics(model, tok, args)
                else:
                    with bank.exported(model, dtype):
                        new["final"] = zero_shot_metrics(model, tok, args)

    previous = {p: {k: metrics[p][k] for k in ZS_KEYS if k in metrics[p]} for p in new}
    for p, values in new.items():
        metrics[p].update(values)
    metrics["zs_reevaluated"] = {"zs_batch": args.zs_batch, "previous_zs_batch": config.get("zs_batch"),
                                 "previous": previous, "timing": timer.t,
                                 "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(mpath, metrics)
    config["zs_batch"] = args.zs_batch
    write_json(cpath, config)
    log("re-evaluated: " + ", ".join(f"{p} {previous[p]['zs_mean_acc']:.4f} -> {new[p]['zs_mean_acc']:.4f}"
                                     for p in new))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
