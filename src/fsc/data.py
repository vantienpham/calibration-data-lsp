"""Calibration sources and evaluation corpora.

Every corpus is read from a raw file of a Hub dataset repository, at the
commit pinned in ``fsc.hub``, via ``hf_hub_download``. Once
``scripts/prefetch.py`` has cached them, jobs run with HF_HUB_OFFLINE=1.

Calibration windows are built the same way for every source: documents are
sampled without replacement (seeded), tokenized without special tokens, packed
into one stream separated by EOS, and cut into windows of ``seqlen - 1`` tokens
preceded by BOS (``seqlen`` tokens, no BOS, for models without one). Training
and validation windows come from disjoint documents of the same source.
"""

from __future__ import annotations

import gzip
import json
import random

import torch

# Raw files, by path; the repository commit is pinned in fsc.hub.DATASET_REVISIONS.
FILES = {
    "wikitext2": ("Salesforce/wikitext", {
        "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
        "validation": "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
        "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    }),
    "c4": ("allenai/c4", {
        "train": "en/c4-train.00000-of-01024.json.gz",
        "validation": "en/c4-validation.00000-of-00008.json.gz",
    }),
    "slimpajama": ("DKYoon/SlimPajama-6B", {
        "train": "data/train-00000-of-00048-ab2b35705f029d94.parquet",
        "validation": "data/validation-00000-of-00001-4fb685c22a3f91ef.parquet",
        "test": "data/test-00000-of-00001-9f769cf7ce219017.parquet",
    }),
    "alpaca": ("tatsu-lab/alpaca", {
        "train": "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet",
    }),
}

# RedPajama subsets inside SlimPajama, by their meta.redpajama_set_name.
SP_SUBSETS = {
    "cc": "RedPajamaCommonCrawl",
    "c4": "RedPajamaC4",
    "github": "RedPajamaGithub",
    "book": "RedPajamaBook",
    "arxiv": "RedPajamaArXiv",
    "wiki": "RedPajamaWikipedia",
    "stack": "RedPajamaStackExchange",
}

ALPACA_HOLDOUT = 2000       # last examples of the train file, reserved for evaluation
C4_PPL_DOCS = 1100          # first validation documents, used for perplexity (GPTQ convention)

ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task{ctx}. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n{input_block}### Response:\n{output}"
)


def raw_file(source: str, split: str) -> str:
    from huggingface_hub import hf_hub_download
    from .hub import dataset_revision
    repo, files = FILES[source]
    return hf_hub_download(repo, files[split], repo_type="dataset",
                           revision=dataset_revision(repo))


def _read_parquet(path: str, columns=None):
    import pyarrow.parquet as pq
    return pq.read_table(path, columns=columns).to_pydict()


def _read_jsonl_gz(path: str, limit: int | None = None) -> list[str]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line)["text"])
            if limit is not None and len(out) >= limit:
                break
    return out


def documents(source: str, split: str) -> list[str]:
    """Plain-text documents of a source split (order as stored)."""
    if source == "c4":
        return _read_jsonl_gz(raw_file("c4", split))
    if source == "alpaca":
        d = _read_parquet(raw_file("alpaca", "train"))
        docs = []
        for ins, inp, out in zip(d["instruction"], d["input"], d["output"]):
            has = bool(inp and inp.strip())
            docs.append(ALPACA_TEMPLATE.format(
                ctx=", paired with an input that provides further context" if has else "",
                instruction=ins.strip(),
                input_block=f"### Input:\n{inp.strip()}\n\n" if has else "",
                output=out.strip()))
        return docs
    if source == "slimpajama" or source.startswith("sp-"):
        d = _read_parquet(raw_file("slimpajama", split), columns=["text", "meta"])
        if source == "slimpajama":
            return d["text"]
        want = SP_SUBSETS[source[3:]]
        return [t for t, m in zip(d["text"], d["meta"]) if _set_name(m) == want]
    raise ValueError(f"no document reader for source {source!r}")


def _set_name(meta) -> str:
    if isinstance(meta, dict):
        return meta.get("redpajama_set_name", "")
    if isinstance(meta, str):
        try:
            return json.loads(meta).get("redpajama_set_name", "")
        except json.JSONDecodeError:
            return ""
    return ""


def wikitext_text(split: str) -> str:
    d = _read_parquet(raw_file("wikitext2", split), columns=["text"])
    return "\n\n".join(d["text"])


# ----------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------

def special_ids(tokenizer) -> tuple[int | None, int]:
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no EOS token to separate documents")
    return bos, eos


def _cut(stream: list[int], n: int, seqlen: int, bos: int | None) -> torch.Tensor:
    body = seqlen - 1 if bos is not None else seqlen
    need = n * body
    if len(stream) < need:
        raise ValueError(f"stream has {len(stream)} tokens, need {need}")
    t = torch.tensor(stream[:need], dtype=torch.long).view(n, body)
    if bos is not None:
        t = torch.cat([torch.full((n, 1), bos, dtype=torch.long), t], 1)
    return t


def _pack(tokenizer, docs: list[str], n_tokens: int, eos: int, start: int = 0,
          batch: int = 256, doc_bos: int | None = None) -> tuple[list[int], int]:
    """Tokenize docs[start:] until ``n_tokens`` tokens are packed; returns (stream, next_index).

    With ``doc_bos``, every document starts with that token (BOS) as well as ending with EOS.
    """
    stream: list[int] = []
    i = start
    while len(stream) < n_tokens:
        if i >= len(docs):
            raise ValueError(f"ran out of documents after {len(stream)} tokens")
        chunk = docs[i:i + batch]
        enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        for ids in enc:
            if doc_bos is not None:
                stream.append(doc_bos)
            stream.extend(ids)
            stream.append(eos)
            i += 1
            if len(stream) >= n_tokens:
                break
    return stream, i


def _pack_ids(docs: list[list[int]], n_tokens: int, eos: int, start: int = 0
              ) -> tuple[list[int], int]:
    """Pack already-tokenized documents, EOS-separated, until n_tokens."""
    stream: list[int] = []
    i = start
    while len(stream) < n_tokens:
        if i >= len(docs):
            raise ValueError(f"ran out of documents after {len(stream)} tokens")
        stream.extend(docs[i])
        stream.append(eos)
        i += 1
    return stream, i


def calibration_windows(tokenizer, source: str, n_train: int, n_val: int, seqlen: int,
                        seed: int = 0, selfgen_path: str | None = None, use_bos: bool = True,
                        doc_bos: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(train, val) windows of shape (n, seqlen) from disjoint documents of ``source``.

    ``use_bos=False`` cuts plain seqlen-token chunks even for models with a BOS
    token (the window-format ablation); self-generated windows always start at
    the model's own document-start token.
    """
    bos, eos = special_ids(tokenizer)
    if not use_bos:
        bos = None
    body = seqlen - 1 if bos is not None else seqlen

    if source == "selfdoc":
        if selfgen_path is None:
            raise ValueError("source 'selfdoc' needs selfgen_path")
        blob = torch.load(selfgen_path, map_location="cpu", weights_only=False)
        flat, lengths = blob["docs_flat"].tolist(), blob["doc_lengths"].tolist()
        docs, pos = [], 0
        for n in lengths:
            docs.append(flat[pos:pos + n])
            pos += n
        random.Random(seed).shuffle(docs)
        s_tr, nxt = _pack_ids(docs, n_train * body, eos)
        s_va, _ = _pack_ids(docs, n_val * body, eos, start=nxt)
        return _cut(s_tr, n_train, seqlen, bos), _cut(s_va, n_val, seqlen, bos)

    if source == "selfgen":
        if selfgen_path is None:
            raise ValueError("source 'selfgen' needs selfgen_path")
        blob = torch.load(selfgen_path, map_location="cpu", weights_only=False)
        ids = blob["ids"] if isinstance(blob, dict) else blob
        if ids.shape[1] < seqlen or ids.shape[0] < n_train + n_val:
            raise ValueError(f"self-generated set {tuple(ids.shape)} too small for "
                             f"{n_train}+{n_val} windows of {seqlen}")
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(ids.shape[0], generator=g)
        ids = ids[perm, :seqlen]
        return ids[:n_train].clone(), ids[n_train:n_train + n_val].clone()

    if source == "wikitext2":
        rng = random.Random(seed)
        tr = tokenizer(wikitext_text("train"), add_special_tokens=False)["input_ids"]
        va = tokenizer(wikitext_text("validation"), add_special_tokens=False)["input_ids"]
        train = _random_chunks(tr, n_train, body, rng)
        val = _random_chunks(va, n_val, body, rng)
        return _add_bos(train, bos), _add_bos(val, bos)

    docs = documents(source, "train")
    if source == "alpaca":
        docs = docs[:-ALPACA_HOLDOUT]
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    docs = [docs[j] for j in order]
    db = bos if doc_bos else None
    s_tr, nxt = _pack(tokenizer, docs, n_train * body, eos, doc_bos=db)
    s_va, _ = _pack(tokenizer, docs, n_val * body, eos, start=nxt, doc_bos=db)
    return _cut(s_tr, n_train, seqlen, bos), _cut(s_va, n_val, seqlen, bos)


def _random_chunks(stream: list[int], n: int, body: int, rng: random.Random) -> torch.Tensor:
    """n non-overlapping chunks of a single stream, chosen at random."""
    n_avail = len(stream) // body
    if n_avail < n:
        raise ValueError(f"stream holds {n_avail} chunks of {body}, need {n}")
    idx = sorted(rng.sample(range(n_avail), n))
    t = torch.tensor(stream[: n_avail * body], dtype=torch.long).view(n_avail, body)
    return t[idx]


def _add_bos(t: torch.Tensor, bos: int | None) -> torch.Tensor:
    if bos is None:
        return t
    return torch.cat([torch.full((t.shape[0], 1), bos, dtype=torch.long), t], 1)


# ----------------------------------------------------------------------------
# Evaluation corpora
# ----------------------------------------------------------------------------

def eval_stream(tokenizer, name: str, special: bool = True) -> torch.Tensor:
    """1-D token stream for perplexity, following the GPTQ/SparseGPT conventions.

    ``special=False`` tokenizes without the leading BOS, for BOS-prefixed windows.
    """
    if name == "wikitext2":
        text = wikitext_text("test")
    elif name == "c4":
        text = " ".join(_read_jsonl_gz(raw_file("c4", "validation"), limit=C4_PPL_DOCS))
    else:
        raise ValueError(name)
    return tokenizer(text, return_tensors="pt", add_special_tokens=special).input_ids[0]


def eval_windows_sp(tokenizer, subset: str, n: int, seqlen: int, seed: int = 0) -> torch.Tensor:
    """Windows from SlimPajama validation documents of one RedPajama subset."""
    bos, eos = special_ids(tokenizer)
    body = seqlen - 1 if bos is not None else seqlen
    d = _read_parquet(raw_file("slimpajama", "validation"), columns=["text", "meta"])
    want = SP_SUBSETS[subset]
    docs = [t for t, m in zip(d["text"], d["meta"]) if _set_name(m) == want]
    random.Random(seed).shuffle(docs)
    stream, _ = _pack(tokenizer, docs, n * body, eos)
    return _cut(stream, n, seqlen, bos)


def kl_eval_windows(tokenizer, source: str, n: int, seqlen: int,
                    selfgen_path: str | None = None, seed: int = 1234) -> torch.Tensor:
    """Held-out windows of a source for measuring KL(dense || compressed).

    Disjoint from every calibration set: WikiText-2 test chunks, C4 validation
    documents after the perplexity set, SlimPajama validation documents, the
    held-out Alpaca examples, and a separately sampled self-generated pool.
    """
    bos, eos = special_ids(tokenizer)
    body = seqlen - 1 if bos is not None else seqlen
    if source == "selfgen":
        blob = torch.load(selfgen_path, map_location="cpu", weights_only=False)
        ids = blob["ids"] if isinstance(blob, dict) else blob
        return ids[:n, :seqlen].clone()
    if source == "wikitext2":
        stream = tokenizer(wikitext_text("test"), add_special_tokens=False)["input_ids"]
        t = torch.tensor(stream[: n * body], dtype=torch.long).view(n, body)
        return _add_bos(t, bos)
    if source == "c4":
        docs = _read_jsonl_gz(raw_file("c4", "validation"))[C4_PPL_DOCS:]
    elif source == "slimpajama":
        docs = _read_parquet(raw_file("slimpajama", "validation"), columns=["text"])["text"]
    elif source == "alpaca":
        docs = documents("alpaca", "train")[-ALPACA_HOLDOUT:]
    else:
        raise ValueError(f"no held-out KL set for {source!r}")
    docs = list(docs)
    random.Random(seed).shuffle(docs)
    stream, _ = _pack(tokenizer, docs, n * body, eos)
    return _cut(stream, n, seqlen, bos)
