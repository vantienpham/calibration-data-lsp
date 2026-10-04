#!/usr/bin/env python3
"""Sample calibration text from the dense model (self-generated calibration).

    # exact 2048-token samples
    uv run --no-sync python scripts/generate_calib.py --model meta-llama/Llama-3.2-1B \
        --n 1088 --seqlen 2048 --temperature 1.0 --seed 0 \
        --out out/calib/llama-3.2-1b/selfgen-T1.0-s0.pt
    # independent documents of at most 512 tokens, 2.3 M tokens in total
    uv run --no-sync python scripts/generate_calib.py --model meta-llama/Llama-3.2-1B \
        --documents --n-tokens 2300000 --max-doc-len 512 --seed 0 \
        --out out/calib/llama-3.2-1b/selfdoc-T1.0-s0.pt

Writes {"ids": (n, seqlen) int64, ...} plus diagnostics: the dense model's mean
per-token NLL on its own samples (an estimate of its entropy rate at T = 1) and
a few decoded samples for inspection. Skips if --out exists.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from fsc import evaluate, selfgen  # noqa: E402
from fsc.utils import env_info, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=576)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--documents", action="store_true",
                    help="sample independent documents (<= --max-doc-len tokens, ended by EOS) "
                         "until --n-tokens tokens, instead of --n windows of --seqlen")
    ap.add_argument("--n-tokens", type=int, default=2_300_000)
    ap.add_argument("--max-doc-len", type=int, default=512)
    args = ap.parse_args(argv)

    if os.path.exists(args.out):
        log(f"{args.out} exists; nothing to do")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    from fsc import hub
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    tok = hub.load_tokenizer(args.model)
    model = hub.load_causal_lm(args.model, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    t0 = time.time()
    if args.documents:
        docs = selfgen.sample_documents(model, tok, args.n_tokens, args.max_doc_len,
                                        args.temperature, args.batch_size, args.seed, "cuda", log)
        lengths = torch.tensor([len(d) for d in docs], dtype=torch.long)
        blob = {
            "docs_flat": torch.tensor([t for d in docs for t in d], dtype=torch.long),
            "doc_lengths": lengths,
            "model": args.model, "temperature": args.temperature, "seed": args.seed,
            "max_doc_len": args.max_doc_len, "seconds": time.time() - t0,
            "mean_doc_len": float(lengths.float().mean()),
            "frac_truncated": float((lengths >= args.max_doc_len).float().mean()),
            "samples": [tok.decode(d[:256]) for d in docs[:3]],
            "env": env_info(),
        }
        tmp = args.out + ".tmp"
        torch.save(blob, tmp)
        os.replace(tmp, args.out)
        log(f"wrote {len(docs)} documents ({int(lengths.sum())} tokens, mean length "
            f"{blob['mean_doc_len']:.0f}, {100 * blob['frac_truncated']:.0f}% truncated) to {args.out}")
        return 0
    ids = selfgen.sample(model, tok, args.n, args.seqlen, args.temperature, args.batch_size,
                         args.seed, "cuda", log)
    seconds = time.time() - t0
    nll, cnt = evaluate.nll_windows(model, ids[: min(64, args.n)], 4, "cuda")
    blob = {
        "ids": ids,
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "seqlen": args.seqlen,
        "seconds": seconds,
        "nll_mean_first64": nll / cnt,
        "samples": [tok.decode(ids[i, :256]) for i in range(min(3, args.n))],
        "env": env_info(),
    }
    tmp = args.out + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, args.out)
    log(f"wrote {tuple(ids.shape)} to {args.out} in {seconds:.0f}s; "
        f"self-NLL {nll / cnt:.3f} nats/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
