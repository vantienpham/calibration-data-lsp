#!/usr/bin/env python3
"""Statistics of every calibration source as the dense model sees it.

    uv run --no-sync python scripts/source_stats.py --model meta-llama/Llama-3.2-1B \
        --selfgen-path out/calib/llama-3.2-1b/selfgen-T1.0-s0.pt \
        --selfdoc-path out/calib/llama-3.2-1b/selfdoc-T1.0-s0.pt \
        --out out/stats/sources-llama-3.2-1b.json

For each source: the mean number of EOS-separated documents per training
window, and the dense model's mean next-token NLL and predictive entropy (nats)
on the source's validation windows. NLL - entropy measures how far the source
is from the model's own predictions: it is zero in expectation on exact samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fsc import data  # noqa: E402
from fsc.utils import log  # noqa: E402


@torch.no_grad()
def nll_entropy(model, windows, batch, device, bucket=128):
    """Mean NLL and entropy, and the entropy averaged over position buckets."""
    nll, ent, n = 0.0, 0.0, 0
    by_pos = torch.zeros(windows.shape[1] - 1, dtype=torch.float64)
    for i in range(0, windows.shape[0], batch):
        x = windows[i:i + batch].to(device)
        lp = model(input_ids=x, use_cache=False).logits[:, :-1].float().log_softmax(-1)
        nll -= float(lp.gather(-1, x[:, 1:, None]).sum())
        e = -(lp.exp() * lp).sum(-1)                      # (B, T-1)
        ent += float(e.sum())
        by_pos += e.sum(0).double().cpu()
        n += x[:, 1:].numel()
    by_pos /= windows.shape[0]
    buckets = [float(by_pos[j:j + bucket].mean()) for j in range(0, by_pos.numel(), bucket)]
    return nll / n, ent / n, buckets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--selfgen-path", default=None)
    ap.add_argument("--selfdoc-path", default=None)
    ap.add_argument("--sources", default="wikitext2,c4,slimpajama,alpaca,selfgen,selfdoc")
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from fsc import hub
    tok = hub.load_tokenizer(args.model)
    model = hub.load_causal_lm(args.model, dtype=torch.bfloat16).cuda().eval()
    out = {}
    for src in args.sources.split(","):
        path = {"selfgen": args.selfgen_path, "selfdoc": args.selfdoc_path}.get(src)
        if src in ("selfgen", "selfdoc") and not path:
            continue
        tr, va = data.calibration_windows(tok, src, args.n_train, args.n_val, args.seqlen,
                                          seed=0, selfgen_path=path)
        eos, bos = tok.eos_token_id, tok.bos_token_id
        docs_per_window = float((tr[:, 1:] == eos).sum(1).float().mean()) + 1.0
        bos_per_window = float((tr == bos).sum(1).float().mean()) if bos is not None else 0.0
        nll, ent, by_pos = nll_entropy(model, va, 4, "cuda")
        out[src] = {"docs_per_window": docs_per_window, "bos_per_window": bos_per_window,
                    "nll": nll, "entropy": ent,
                    "nll_minus_entropy": nll - ent, "entropy_by_pos_128": by_pos}
        log(f"{src:<12} docs/window {docs_per_window:6.2f}  NLL {nll:.3f}  entropy {ent:.3f}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
