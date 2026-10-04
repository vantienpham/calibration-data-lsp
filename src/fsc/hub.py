"""Pinned Hugging Face Hub revisions of every model and corpus the paper uses.

The experiments ran against these exact commits. Loading by commit makes a re-run
independent of later updates to the Hub repositories, and works offline once the
cache holds them (``scripts/prefetch.py``). Repositories that are not listed are
loaded from their default branch.

Loading at these commits gives bit-identical weights and tokenizers to the
default-branch loads the paper's runs used (checked for OPT-125M, OPT-1.3B,
Llama-3.2-1B and Qwen3-1.7B-Base; Llama-3.1-8B had a single commit).
"""

from __future__ import annotations

MODEL_REVISIONS = {
    "facebook/opt-125m": "27dcfa74d334bc871f3234de431e71c6eeba5dd6",
    "facebook/opt-1.3b": "3f5c25d0bc631cb57ac65913f76e22c2dfb61d62",
    "meta-llama/Llama-3.2-1B": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "meta-llama/Llama-3.1-8B": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
    "Qwen/Qwen3-1.7B-Base": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
}

# Calibration and evaluation corpora (raw files, see fsc.data.FILES).
DATASET_REVISIONS = {
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "allenai/c4": "1588ec454efa1a09f29cd18ddd04fe05fc8653a2",
    "DKYoon/SlimPajama-6B": "b5f90f419b7489cdba26fdbc8c022fcb5562f968",
    "tatsu-lab/alpaca": "dce01c9b08f87459cf36a430d809084718273017",
}

# Zero-shot benchmarks, as read by lm-eval 0.4.9.2's task definitions. lm-eval
# loads them by name, so these are recorded for reference rather than enforced.
BENCHMARK_REVISIONS = {
    "allenai/ai2_arc": "210d026faf9955653af8916fad021475a3f00453",
    "baber/piqa": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
    "Rowan/hellaswag": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    "winogrande": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
    "super_glue": "3de24cf8022e94f4ee4b9d55a6f539891524d646",
}


def model_revision(name: str) -> str | None:
    return MODEL_REVISIONS.get(name)


def dataset_revision(repo: str) -> str | None:
    return DATASET_REVISIONS.get(repo)


def load_tokenizer(name: str, **kwargs):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, revision=model_revision(name), **kwargs)


def load_causal_lm(name: str, **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(name, revision=model_revision(name), **kwargs)
