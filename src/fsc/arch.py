"""Compression units: which linear layers are projected together, on which side,
and how many parameters they store before and after factorization.

A *unit* is either a single linear layer, projected on its smaller side, or a
tied group of layers that read the same activation (Q/K/V; gate/up), sharing one
input-side projector. Every pretrained linear layer of every decoder block is in
exactly one unit; embeddings, the LM head and normalizations are not compressed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn


@dataclass
class Unit:
    name: str          # e.g. "L3.qkv"
    kind: str          # qkv | o | gateup | down | fc1 | fc2
    block: int
    side: str          # "in": project the shared input; "out": project the output
    members: list[str]  # module paths inside the model
    d_in: int
    d_outs: list[int]  # one per member

    @property
    def d(self) -> int:
        """Dimension of the projected side."""
        return self.d_in if self.side == "in" else self.d_outs[0]

    @property
    def dense_params(self) -> int:
        return sum(o * self.d_in for o in self.d_outs)

    def factor_params(self, r: int) -> int:
        """Weights stored by the merged factors at rank r, shared factors counted once."""
        if self.side == "in":
            return r * (self.d_in + sum(self.d_outs))
        assert len(self.d_outs) == 1, "output-side units have a single member"
        return r * (self.d_outs[0] + self.d_in)

    def params(self, k: int) -> int:
        """Weights stored after removing k directions; k = 0 leaves the unit dense."""
        return self.dense_params if k <= 0 else self.factor_params(self.d - k)

    def savings(self, k: int) -> int:
        return self.dense_params - self.params(k)

    def break_even_k(self) -> int:
        """Smallest k whose factors are strictly smaller than the dense unit."""
        per_rank = self.factor_params(1)
        r_max = (self.dense_params - 1) // per_rank      # largest r with r*per_rank < dense
        return max(self.d - r_max, 1)

    def to_dict(self) -> dict:
        return asdict(self)


def _linear(model: nn.Module, path: str) -> nn.Linear:
    mod = model.get_submodule(path)
    if not isinstance(mod, nn.Linear):
        raise TypeError(f"{path} is {type(mod).__name__}, expected nn.Linear")
    return mod


def _single(model, name, kind, block, path) -> Unit:
    lin = _linear(model, path)
    side = "in" if lin.in_features <= lin.out_features else "out"
    return Unit(name, kind, block, side, [path], lin.in_features, [lin.out_features])


def _tied(model, name, kind, block, paths) -> Unit:
    lins = [_linear(model, p) for p in paths]
    d_in = lins[0].in_features
    assert all(l.in_features == d_in for l in lins), f"{name}: members disagree on d_in"
    return Unit(name, kind, block, "in", list(paths), d_in, [l.out_features for l in lins])


def discover_units(model: nn.Module, tie: bool = True) -> list[Unit]:
    """All compression units of a decoder-only HF model, in forward order.

    With ``tie=False`` every linear layer is its own unit (projected on its
    smaller side), which is the untied control.
    """
    mt = model.config.model_type
    units: list[Unit] = []

    if mt == "opt":
        if not getattr(model.config, "do_layer_norm_before", True):
            raise NotImplementedError("post-LN OPT variants (opt-350m) are not supported")
        prefix = "model.decoder.layers"
        n = len(model.model.decoder.layers)
        for i in range(n):
            a = f"{prefix}.{i}.self_attn"
            qkv = [f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"]
            if tie:
                units.append(_tied(model, f"L{i}.qkv", "qkv", i, qkv))
            else:
                units += [_single(model, f"L{i}.{p.rsplit('.', 1)[1][0]}", "qkv", i, p) for p in qkv]
            units.append(_single(model, f"L{i}.o", "o", i, f"{a}.out_proj"))
            units.append(_single(model, f"L{i}.fc1", "fc1", i, f"{prefix}.{i}.fc1"))
            units.append(_single(model, f"L{i}.fc2", "fc2", i, f"{prefix}.{i}.fc2"))
        return units

    if mt in ("llama", "mistral", "qwen2", "qwen3"):
        prefix = "model.layers"
        n = len(model.model.layers)
        for i in range(n):
            a, m = f"{prefix}.{i}.self_attn", f"{prefix}.{i}.mlp"
            qkv = [f"{a}.q_proj", f"{a}.k_proj", f"{a}.v_proj"]
            gu = [f"{m}.gate_proj", f"{m}.up_proj"]
            if tie:
                units.append(_tied(model, f"L{i}.qkv", "qkv", i, qkv))
            else:
                units += [_single(model, f"L{i}.{p.rsplit('.', 1)[1][0]}", "qkv", i, p) for p in qkv]
            units.append(_single(model, f"L{i}.o", "o", i, f"{a}.o_proj"))
            if tie:
                units.append(_tied(model, f"L{i}.gateup", "gateup", i, gu))
            else:
                units += [_single(model, f"L{i}.{p.rsplit('.', 1)[1][:4]}", "gateup", i, p) for p in gu]
            units.append(_single(model, f"L{i}.down", "down", i, f"{m}.down_proj"))
        return units

    raise NotImplementedError(f"model_type {mt!r} is not supported")


def total_dense(units: list[Unit]) -> int:
    return sum(u.dense_params for u in units)


def realized_ratio(units: list[Unit], ks: dict[str, int]) -> float:
    """Fraction of compressible-linear weights removed by the allocation ``ks``."""
    dense = total_dense(units)
    kept = sum(u.params(ks.get(u.name, 0)) for u in units)
    return 1.0 - kept / dense
