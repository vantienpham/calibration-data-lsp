"""Perplexity and zero-shot accuracy of a (possibly merged) model."""

from __future__ import annotations

import math

import torch

ZS_TASKS = ["arc_easy", "arc_challenge", "piqa", "hellaswag", "winogrande", "openbookqa", "boolq"]


@torch.no_grad()
def nll_windows(model, windows: torch.Tensor, batch_size: int, device) -> tuple[float, int]:
    """Summed next-token NLL over a batch of windows, and the number of predicted tokens."""
    total, count = 0.0, 0
    for i in range(0, windows.shape[0], batch_size):
        x = windows[i:i + batch_size].to(device)
        logits = model(input_ids=x, use_cache=False).logits
        for b in range(x.shape[0]):
            lp = logits[b, :-1].float().log_softmax(-1)
            total -= float(lp.gather(-1, x[b, 1:, None]).sum())
            count += x.shape[1] - 1
    return total, count


@torch.no_grad()
def perplexity(model, stream: torch.Tensor, seqlen: int, batch_size: int, device,
               max_windows: int | None = None, bos: int | None = None) -> float:
    """Perplexity over non-overlapping windows of a 1-D token stream.

    With ``bos``, every window is BOS followed by seqlen - 1 stream tokens, the
    format the model sees at a document start (and in calibration). Without it,
    windows are plain seqlen-token chunks (the GPTQ/SparseGPT convention, where
    only the first window begins with BOS).
    """
    body = seqlen - 1 if bos is not None else seqlen
    n = stream.numel() // body
    if max_windows is not None:
        n = min(n, max_windows)
    w = stream[: n * body].view(n, body)
    if bos is not None:
        w = torch.cat([torch.full((n, 1), bos, dtype=w.dtype), w], 1)
    total, count = nll_windows(model, w, batch_size, device)
    return math.exp(total / count)


@torch.no_grad()
def window_perplexity(model, windows: torch.Tensor, batch_size: int, device) -> float:
    total, count = nll_windows(model, windows, batch_size, device)
    return math.exp(total / count)


def zero_shot(model, tokenizer, tasks=ZS_TASKS, batch_size: int = 32,
              limit: int | None = None) -> dict:
    """lm-eval-harness, zero-shot; returns {task: {metric: value}} with stderr."""
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model.eval()
    while True:
        # lm-eval takes a float32 log-softmax over (batch, length, vocab); with a
        # 150k vocabulary and long BoolQ/HellaSwag contexts, batch 32 needs ~12 GB.
        # Halve the batch on OOM rather than lose a finished run.
        torch.cuda.empty_cache()
        try:
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
            out = simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=0, limit=limit,
                                  log_samples=False, bootstrap_iters=1000, verbosity="WARNING")
            break
        except torch.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size //= 2
            print(f"zero-shot OOM; retrying with batch size {batch_size}", flush=True)
    res = {}
    for task, r in out["results"].items():
        res[task] = {k.split(",")[0]: v for k, v in r.items()
                     if "," in k and isinstance(v, (int, float))}
    return res


def zs_mean(res: dict, metric: str = "acc", tasks=ZS_TASKS) -> float:
    vals = [res[t][metric] for t in tasks if t in res and metric in res[t]]
    return float(sum(vals) / len(vals)) if vals else float("nan")
