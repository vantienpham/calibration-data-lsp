"""KL distillation of the projectors (or of free factors) with the weights frozen.

The teacher is the same network with its projectors disabled, run without
gradients, so no second copy of the weights is stored. The loss is the
token-averaged KL(p_dense || p_compressed) over all positions. Its gradient with
respect to the student logits, softmax(s) - p_t, is formed in chunks and pushed
through the network with ``logits.backward(grad)``, so the full-vocabulary
softmax is never kept in the autograd graph.

Schedule (Bini et al., 2026): Adam without weight decay, linear warm-up then
cosine decay, alpha ramped 0 -> 1 over the first epoch, direction dropout,
an orthogonality penalty on V, and early stopping on the validation KL with the
best epoch restored. State is checkpointed every epoch so a wall-time kill
loses at most one epoch.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass
class TrainConfig:
    mode: str = "proj"           # proj | factors
    epochs: int = 8
    lr: float = 1e-3
    batch_size: int = 8          # sequences per optimizer step
    micro_batch: int = 1
    drop_p: float = 0.05
    lam_ort: float = 0.05
    alpha_epochs: float = 1.0
    warmup_frac: float = 0.05
    patience: int = 3
    seed: int = 0
    kl_chunk: int = 512
    epoch_size: int | None = None   # windows per (virtual) epoch; default: all of them

    def to_dict(self):
        return asdict(self)


def kl_and_grad(t_logits: torch.Tensor, s_logits: torch.Tensor, scale: float,
                chunk: int = 512) -> tuple[float, torch.Tensor]:
    """Sum of per-position KL(p_t || p_s), and scale * d/ds of it."""
    grad = torch.empty_like(s_logits)
    total = 0.0
    T = s_logits.shape[1]
    with torch.no_grad():
        for i in range(0, T, chunk):
            lt = t_logits[:, i:i + chunk].float().log_softmax(-1)
            ls = s_logits[:, i:i + chunk].float().log_softmax(-1)
            pt = lt.exp()
            total += float((pt * (lt - ls)).sum())
            grad[:, i:i + chunk] = ((ls.exp() - pt) * scale).to(grad.dtype)
    return total, grad


@torch.no_grad()
def mean_kl(model, bank, windows: torch.Tensor, batch_size: int, device, chunk: int = 512) -> float:
    """Token-averaged KL(dense || compressed) at the evaluation projector."""
    total, n = 0.0, 0
    for i in range(0, windows.shape[0], batch_size):
        x = windows[i:i + batch_size].to(device)
        bank.set_active(False)
        t = model(input_ids=x, use_cache=False).logits
        bank.set_active(True)
        s = model(input_ids=x, use_cache=False).logits
        T = s.shape[1]
        for j in range(0, T, chunk):
            lt = t[:, j:j + chunk].float().log_softmax(-1)
            ls = s[:, j:j + chunk].float().log_softmax(-1)
            total += float((lt.exp() * (lt - ls)).sum())
        n += x.numel()
        if hasattr(bank, "clear"):
            bank.clear()
    return total / n


def _lr_lambda(warmup: int, total: int):
    def f(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return f


def sample_indices(n: int, start: int, count: int, seed: int) -> np.ndarray:
    """Window indices for global sample positions [start, start + count).

    The windows are visited through an endless sequence of permutations of
    range(n), the w-th seeded by (seed, w). With an epoch of n samples this is a
    fresh permutation per epoch; with epochs of a fixed size it cycles through a
    small pool or visits a large one exactly once, deterministically, so a
    resumed run continues the same sequence.
    """
    out = np.empty(count, dtype=np.int64)
    pos = start
    filled = 0
    while filled < count:
        w, off = divmod(pos, n)
        perm = np.random.default_rng([seed, w]).permutation(n)
        take = min(n - off, count - filled)
        out[filled:filled + take] = perm[off:off + take]
        filled += take
        pos += take
    return out


def eval_mode(bank, dtype) -> None:
    """Evaluation operator: alpha = 1, no direction dropout."""
    bank.set_active(True)
    if hasattr(bank, "refresh"):
        bank.set_alpha(1.0)
        with torch.no_grad():
            bank.refresh(dtype, 0.0)
    else:
        bank.alpha = 1.0


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(model, bank, train_w: torch.Tensor, val_w: torch.Tensor, cfg: TrainConfig,
          run_dir: str, device, dtype: torch.dtype, log=print) -> dict:
    """Train ``bank`` (ProjectorBank or FactorBank) in place; returns a summary.

    Resumes from ``run_dir/train_state.pt`` if present. On return the bank holds
    the best-validation parameters.
    """
    is_proj = cfg.mode == "proj"
    n = train_w.shape[0]
    epoch_size = cfg.epoch_size or n
    steps_per_epoch = epoch_size // cfg.batch_size
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    params = [p for p in bank.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(warmup, total_steps))
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    state_path = os.path.join(run_dir, "train_state.pt")
    best_path = os.path.join(run_dir, "best_params.pt")
    log_path = os.path.join(run_dir, "train_log.jsonl")
    start_epoch, best_val, best_epoch, history, step = 0, float("inf"), -1, [], 0
    bad = 0

    if os.path.exists(state_path):
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        bank.load_raw(st["params"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch, best_val, best_epoch = st["epoch"] + 1, st["best_val"], st["best_epoch"]
        history, step, bad = st["history"], st["step"], st["bad"]
        gen.manual_seed(cfg.seed + 1 + 1000 * start_epoch)
        log(f"resumed after epoch {st['epoch']} (best val KL {best_val:.5f} at {best_epoch})")
    else:
        # Validation KL of the initialization: early stopping may keep it.
        eval_mode(bank, dtype)
        v0 = mean_kl(model, bank, val_w, cfg.micro_batch, device, cfg.kl_chunk)
        best_val, best_epoch = v0, -1
        _atomic_save(bank.raw(), best_path)
        history.append({"epoch": -1, "val_kl": v0})
        log(f"init val KL {v0:.5f}")

    # The model stays in eval mode: OPT's config carries dropout 0.1, and the
    # teacher and the student must see the same deterministic network.
    model.eval()

    T = train_w.shape[1]
    for epoch in range(start_epoch, cfg.epochs):
        if bad >= cfg.patience:
            break
        t0 = time.time()
        order = sample_indices(n, epoch * steps_per_epoch * cfg.batch_size,
                               steps_per_epoch * cfg.batch_size, cfg.seed)
        tr_kl, tr_tok = 0.0, 0
        for s in range(steps_per_epoch):
            step += 1
            alpha = min(1.0, step / max(cfg.alpha_epochs * steps_per_epoch, 1))
            idx = order[s * cfg.batch_size:(s + 1) * cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            scale = 1.0 / (len(idx) * T)
            if is_proj:
                bank.set_alpha(alpha)
                bank.begin_step(cfg.drop_p, gen)
            else:
                bank.alpha = alpha
            for j in range(0, len(idx), cfg.micro_batch):
                x = train_w[idx[j:j + cfg.micro_batch]].to(device)
                bank.set_active(False)
                with torch.no_grad():
                    t_logits = model(input_ids=x, use_cache=False).logits
                bank.set_active(True)
                if is_proj:
                    bank.begin_micro(dtype)
                s_logits = model(input_ids=x, use_cache=False).logits
                kl, g = kl_and_grad(t_logits, s_logits, scale, cfg.kl_chunk)
                s_logits.backward(g)
                tr_kl += kl
                tr_tok += x.numel()
                del t_logits, s_logits, g
                if is_proj:
                    bank.clear()
            if is_proj:
                bank.end_step()
                if cfg.lam_ort > 0:
                    (cfg.lam_ort * bank.ortho_penalty()).backward()
            opt.step()
            sched.step()
            if s % max(1, steps_per_epoch // 4) == 0:
                log(f"  epoch {epoch} step {s + 1}/{steps_per_epoch} alpha {alpha:.2f} "
                    f"lr {sched.get_last_lr()[0]:.2e} train KL {tr_kl / max(tr_tok, 1):.5f} "
                    f"({time.time() - t0:.0f}s)")

        # Validation at the evaluation operator: alpha = 1, no dropout.
        eval_mode(bank, dtype)
        v = mean_kl(model, bank, val_w, cfg.micro_batch, device, cfg.kl_chunk)
        rec = {"epoch": epoch, "train_kl": tr_kl / max(tr_tok, 1), "val_kl": v,
               "lr": sched.get_last_lr()[0], "seconds": time.time() - t0}
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if v < best_val:
            best_val, best_epoch, bad = v, epoch, 0
            _atomic_save(bank.raw(), best_path)
        else:
            bad += 1
        log(f"epoch {epoch}: train KL {rec['train_kl']:.5f} val KL {v:.5f} "
            f"(best {best_val:.5f} @ {best_epoch}) {rec['seconds']:.0f}s")
        _atomic_save({"params": bank.raw(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                      "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
                      "history": history, "step": step, "bad": bad}, state_path)

    bank.load_raw(torch.load(best_path, map_location="cpu"))
    eval_mode(bank, dtype)
    epochs_run = max([h["epoch"] for h in history] + [-1]) + 1
    return {"best_val_kl": best_val, "best_epoch": best_epoch, "epochs_run": epochs_run,
            "history": history, "steps": step}
