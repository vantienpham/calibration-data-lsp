"""Rank allocation: how many directions each unit removes under a global budget.

Uniform: every matrix keeps the same fraction rho of its dense parameters,
r = floor(rho * d_in d_out / (d_in + d_out)); a tied group keeps the largest of
its members' ranks; rho is set by bisection on the merged count (shared factors
counted once). This is the uniform control of Bini et al. (2026, Eq. 16).

Measured KL: truncate one unit at a time to each point of a removal grid, with
the rest of the network dense, and record the KL divergence this induces on the
output distribution. Starting dense, a greedy allocator repeatedly takes the
move with the smallest KL increase per parameter saved, then trims the last
move to land on the target.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from .arch import Unit, realized_ratio, total_dense
from .project import ProjectorBank


# ----------------------------------------------------------------------------
# Uniform
# ----------------------------------------------------------------------------

def uniform_ks_at(units: list[Unit], rho: float) -> dict[str, int]:
    ks = {}
    for u in units:
        rs = [int(math.floor(rho * o * u.d_in / (o + u.d_in))) for o in u.d_outs]
        r = max(rs) if u.side == "in" else rs[0]
        r = min(max(r, 1), u.d)
        k = u.d - r
        if k > 0 and u.params(k) >= u.dense_params:
            k = 0
        ks[u.name] = k
    return ks


def uniform_alloc(units: list[Unit], target: float, iters: int = 60) -> dict[str, int]:
    """Largest uniform retained fraction whose realized ratio still meets ``target``."""
    lo, hi = 0.0, 1.0
    best = uniform_ks_at(units, 0.0)
    if realized_ratio(units, best) < target:
        raise ValueError(f"target {target} is not reachable with uniform allocation")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ks = uniform_ks_at(units, mid)
        if realized_ratio(units, ks) >= target:
            lo, best = mid, ks
        else:
            hi = mid
    return best


# ----------------------------------------------------------------------------
# Measured KL
# ----------------------------------------------------------------------------

def grid_ks(u: Unit, fracs: list[float]) -> list[int]:
    return sorted({int(round(f * u.d)) for f in fracs} - {0})


def kl_sum(logp_t: torch.Tensor, s_logits: torch.Tensor, chunk: int = 512) -> float:
    """Sum over positions of KL(p_t || p_s); logp_t are teacher log-probs (fp32)."""
    total = 0.0
    T = s_logits.shape[1]
    for i in range(0, T, chunk):
        lt = logp_t[:, i:i + chunk]
        ls = s_logits[:, i:i + chunk].float().log_softmax(-1)
        total += float((lt.exp() * (lt - ls)).sum())
    return total


@torch.no_grad()
def measure_kl_curves(model, units: list[Unit], bases: dict, windows: torch.Tensor,
                      fracs: list[float], batch_size: int, dtype: torch.dtype,
                      device, log=print) -> dict[str, dict[int, float]]:
    """Per-unit KL(dense || unit truncated to k) on ``windows``, for every grid k."""
    bank = ProjectorBank(units, {u.name: torch.zeros(u.d, 0) for u in units}).to(device)
    bank.attach(model)
    bank.set_active(False)
    try:
        cand = {}
        for u in units:
            cand[u.name] = {k: bases[u.name].removed(k, device=device).to(dtype)
                            for k in grid_ks(u, fracs)}
        sums = {u.name: {k: 0.0 for k in cand[u.name]} for u in units}
        n_tok, t0 = 0, time.time()
        n_batches = math.ceil(windows.shape[0] / batch_size)
        for b, i in enumerate(range(0, windows.shape[0], batch_size)):
            x = windows[i:i + batch_size].to(device)
            logp_t = model(input_ids=x, use_cache=False).logits.float().log_softmax(-1)
            for u in units:
                p = bank[u.name]
                p.active = True
                for k, U in cand[u.name].items():
                    p.set_basis(U, dtype)
                    logits = model(input_ids=x, use_cache=False).logits
                    sums[u.name][k] += kl_sum(logp_t, logits)
                p.active = False
                p.clear()
            n_tok += x.numel()
            del logp_t
            log(f"  kl-measure batch {b + 1}/{n_batches} ({time.time() - t0:.0f}s)")
        return {name: {k: s / n_tok for k, s in d.items()} for name, d in sums.items()}
    finally:
        bank.detach(model)


def _curve(curve: dict[int, float]):
    """Nondecreasing piecewise-linear interpolant through (0, 0) and the grid."""
    ks = [0] + sorted(curve)
    vals = np.maximum.accumulate(np.array([0.0] + [curve[k] for k in sorted(curve)]))
    return ks, (lambda k: float(np.interp(k, ks, vals)))


def greedy_alloc(units: list[Unit], curves: dict[str, dict[int, float]],
                 target: float) -> tuple[dict[str, int], float]:
    """Greedy measured-KL allocation; returns ks and the predicted (additive) KL."""
    need = target * total_dense(units)
    fns = {u.name: _curve(curves[u.name]) for u in units}
    state = {u.name: 0 for u in units}
    saved, last = 0, None
    while saved < need:
        best = None
        for u in units:
            k0 = state[u.name]
            s0 = u.savings(k0)
            grid, f = fns[u.name]
            f0 = f(k0)
            for k1 in grid:
                if k1 <= k0:
                    continue
                s1 = u.savings(k1)
                if s1 <= s0:
                    continue
                cost = (f(k1) - f0) / (s1 - s0)
                if best is None or cost < best[0]:
                    best = (cost, u, k0, k1)
        if best is None:
            raise ValueError(f"target {target} not reachable on the measurement grid")
        _, u, k0, k1 = best
        saved += u.savings(k1) - u.savings(k0)
        state[u.name] = k1
        last = (u, k0, k1, saved - (u.savings(k1) - u.savings(k0)))

    # Trim the last move to the smallest k that still meets the target.
    u, k0, k1, saved_before = last
    required = need - saved_before
    lo, hi = k0 + 1, k1
    while lo < hi:
        mid = (lo + hi) // 2
        if u.savings(mid) - u.savings(k0) >= required:
            hi = mid
        else:
            lo = mid + 1
    state[u.name] = lo
    predicted = sum(fns[v.name][1](state[v.name]) for v in units)
    return state, predicted
