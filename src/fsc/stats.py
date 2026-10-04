"""Second-order statistics of every unit on calibration data.

Input-side units need the Gram matrix of their (shared) input, G = X^T X / n.
Output-side units need the Gram of their bias-free output, W G W^T, which is
accumulated directly from the outputs: it is d_out x d_out, whereas the input
Gram of a down projection would be intermediate x intermediate.

Per-batch products run in float32 (TF32 off) and accumulate in float64.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .arch import Unit


def _base_model(model: nn.Module) -> nn.Module:
    """The decoder stack without the LM head: the head is not needed for Grams."""
    return model.model


@torch.no_grad()
def collect_grams(model: nn.Module, units: list[Unit], windows: torch.Tensor,
                  batch_size: int = 4, acc_dtype: torch.dtype = torch.float64,
                  device: str | torch.device = "cuda") -> tuple[dict[str, torch.Tensor], int]:
    """Gram matrix per unit (normalized by the token count) and the token count."""
    grams = {u.name: torch.zeros(u.d, u.d, dtype=acc_dtype, device=device) for u in units}
    hooks = []

    def in_hook(name):
        def fn(_mod, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            grams[name].add_((x.T @ x).to(acc_dtype))
        return fn

    def out_hook(name):
        def fn(mod, _args, y):
            y = y.reshape(-1, y.shape[-1]).float()
            if mod.bias is not None:
                y = y - mod.bias.float()
            grams[name].add_((y.T @ y).to(acc_dtype))
        return fn

    for u in units:
        mod = model.get_submodule(u.members[0])
        if u.side == "in":
            hooks.append(mod.register_forward_pre_hook(in_hook(u.name)))
        else:
            hooks.append(mod.register_forward_hook(out_hook(u.name)))

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    n_tok = 0
    try:
        base = _base_model(model)
        for i in range(0, windows.shape[0], batch_size):
            batch = windows[i:i + batch_size].to(device)
            base(input_ids=batch, use_cache=False)
            n_tok += batch.numel()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        for h in hooks:
            h.remove()

    for g in grams.values():
        g.div_(n_tok)
    return grams, n_tok
