import torch

from conftest import random_orthonormal
from fsc.arch import discover_units
from fsc.project import (FactorBank, Projector, ProjectorBank, complement_basis, merged,
                         stored_params)


def _ids(n=2, t=12, vocab=97, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n, t), generator=g)


def _random_bases(units, frac=0.5, seed=0):
    return {u.name: random_orthonormal(u.d, max(1, int(frac * u.d)), seed + i)
            for i, u in enumerate(units)}


def test_projector_is_orthogonal_projection():
    U = random_orthonormal(16, 5)
    p = Projector(U.float())
    p.refresh(torch.float32)
    x = torch.randn(3, 16)
    y = p.remove(x)
    assert torch.allclose(y @ U.float(), torch.zeros(3, 5), atol=1e-5)   # removed span is gone
    assert torch.allclose(p.remove(y), y, atol=1e-5)                      # idempotent
    p.alpha = 0.0
    assert p.idle()


def test_alpha_zero_is_dense(tiny_model):
    units = discover_units(tiny_model)
    x = _ids()
    ref = tiny_model(input_ids=x).logits
    bank = ProjectorBank(units, {k: v.float() for k, v in _random_bases(units).items()})
    bank.attach(tiny_model)
    bank.set_alpha(0.0)
    bank.refresh(torch.float32)
    out = tiny_model(input_ids=x).logits
    bank.detach(tiny_model)
    assert torch.allclose(out, ref, atol=1e-6)


def test_merge_is_exact(tiny_model):
    """Projected model (alpha=1) and its merged factors compute the same function."""
    units = discover_units(tiny_model)
    bases = _random_bases(units)
    x = _ids()
    bank = ProjectorBank(units, {k: v.float() for k, v in bases.items()})
    bank.attach(tiny_model)
    bank.refresh(torch.float32)
    a = tiny_model(input_ids=x).logits
    bank.detach(tiny_model)
    dense = tiny_model(input_ids=x).logits
    with merged(tiny_model, units, bases, torch.float32):
        b = tiny_model(input_ids=x).logits
        n = stored_params(tiny_model, units)
    assert not torch.allclose(a, dense, atol=1e-3)          # the projection does something
    assert torch.allclose(a, b, atol=1e-4), (a - b).abs().max()
    expected = sum(u.params(bases[u.name].shape[1]) for u in units)
    assert n == expected
    # restored on exit
    assert torch.allclose(tiny_model(input_ids=x).logits, dense)


def test_shared_input_projected_once(tiny_llama):
    units = discover_units(tiny_llama)
    bases = _random_bases(units)
    bank = ProjectorBank(units, {k: v.float() for k, v in bases.items()})
    bank.attach(tiny_llama)
    bank.refresh(torch.float32)
    calls = {"n": 0}
    p = bank["L0.qkv"]
    orig = p.remove

    def counting(x):
        calls["n"] += 1
        return orig(x)

    p.remove = counting
    tiny_llama(input_ids=_ids())
    bank.detach(tiny_llama)
    assert calls["n"] == 1          # Q, K and V share one projection of their input


def test_complement_basis():
    U = random_orthonormal(10, 4)
    Up = complement_basis(U)
    assert Up.shape == (10, 6)
    assert torch.allclose(U.T @ Up, torch.zeros(4, 6, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(Up.T @ Up, torch.eye(6, dtype=torch.float64), atol=1e-10)


def test_factor_bank_starts_at_projection(tiny_model):
    units = discover_units(tiny_model)
    bases = _random_bases(units)
    x = _ids()
    bank = ProjectorBank(units, {k: v.float() for k, v in bases.items()})
    bank.attach(tiny_model)
    for alpha in (1.0, 0.4):
        bank.set_alpha(alpha)
        bank.refresh(torch.float32)
        ref = tiny_model(input_ids=x).logits
        bank.detach(tiny_model)
        fb = FactorBank(tiny_model, units, bases)
        fb.attach(tiny_model)
        fb.alpha = alpha
        out = tiny_model(input_ids=x).logits
        fb.detach(tiny_model)
        assert torch.allclose(out, ref, atol=1e-4), (alpha, (out - ref).abs().max())
        bank.attach(tiny_model)
    bank.detach(tiny_model)


def test_merge_factors_cast_per_unit(tiny_llama):
    """Casting inside merge_factors gives the float64 factors rounded once, and keeps one
    shared A object per tied group (stored once by swapped)."""
    from fsc.project import merge_factors
    units = discover_units(tiny_llama)
    bases = _random_bases(units)
    for u in units:
        A64, p64 = merge_factors(tiny_llama, u, bases[u.name])
        A16, p16 = merge_factors(tiny_llama, u, bases[u.name], torch.bfloat16)
        for (_, a64, b64, _), (_, a16, b16, _) in zip(p64, p16):
            assert a16.dtype == b16.dtype == torch.bfloat16
            assert torch.equal(a16, a64.to(torch.bfloat16)) and torch.equal(b16, b64.to(torch.bfloat16))
        if u.side == "in" and len(u.members) > 1:
            assert all(a is A16 for _, a, _, _ in p16)
