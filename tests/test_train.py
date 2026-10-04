import torch

from conftest import random_orthonormal
from fsc.arch import discover_units
from fsc.project import FactorBank, ProjectorBank
from fsc.train import TrainConfig, mean_kl, sample_indices, train


def _windows(n, t=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 97, (n, t), generator=g)


def _setup(model, frac=0.4):
    units = discover_units(model)
    bases = {u.name: random_orthonormal(u.d, max(1, int(frac * u.d)), i) for i, u in enumerate(units)}
    return units, bases


def test_projector_training_lowers_kl_and_resumes(tiny_llama, tmp_path):
    units, bases = _setup(tiny_llama)
    bank = ProjectorBank(units, {k: v.float() for k, v in bases.items()})
    bank.attach(tiny_llama)
    tr, va = _windows(16), _windows(4, seed=1)
    cfg = TrainConfig(epochs=3, lr=3e-2, batch_size=4, micro_batch=2, patience=5, drop_p=0.0)
    out = train(tiny_llama, bank, tr, va, cfg, str(tmp_path), "cpu", torch.float32, log=lambda *_: None)
    v0 = out["history"][0]["val_kl"]
    assert out["best_val_kl"] < v0
    assert out["epochs_run"] == 3
    # A rerun resumes from the checkpoint and does no further epochs.
    out2 = train(tiny_llama, bank, tr, va, cfg, str(tmp_path), "cpu", torch.float32, log=lambda *_: None)
    assert out2["steps"] == out["steps"]
    assert abs(mean_kl(tiny_llama, bank, va, 2, "cpu") - out["best_val_kl"]) < 1e-6
    bank.detach(tiny_llama)


def test_factor_training_lowers_kl(tiny_opt, tmp_path):
    units, bases = _setup(tiny_opt)
    bank = FactorBank(tiny_opt, units, bases)
    bank.attach(tiny_opt)
    tr, va = _windows(16), _windows(4, seed=1)
    cfg = TrainConfig(mode="factors", epochs=2, lr=1e-2, batch_size=4, micro_batch=4, patience=5)
    out = train(tiny_opt, bank, tr, va, cfg, str(tmp_path), "cpu", torch.float32, log=lambda *_: None)
    assert out["best_val_kl"] < out["history"][0]["val_kl"]
    bank.detach(tiny_opt)


def test_sample_indices_cover_and_resume():
    a = sample_indices(10, 0, 25, seed=3)
    # every block of n consecutive samples is a permutation
    assert sorted(a[:10]) == list(range(10)) and sorted(a[10:20]) == list(range(10))
    # resuming mid-stream reproduces the same sequence
    assert (sample_indices(10, 7, 18, seed=3) == a[7:]).all()
    # a pool as large as the budget is visited exactly once
    b = sample_indices(40, 0, 40, seed=0)
    assert len(set(b.tolist())) == 40


def test_one_qr_per_step_matches_per_micro_batch_gradient(tiny_llama):
    """Accumulating into a detached basis and backpropagating QR once gives the same gradient."""
    from fsc.train import kl_and_grad
    units, bases = _setup(tiny_llama)
    x = _windows(4)
    grads = []
    for once in (False, True):
        bank = ProjectorBank(units, {k: v.float() for k, v in bases.items()})
        bank.attach(tiny_llama)
        bank.set_alpha(0.7)
        if once:
            bank.begin_step(0.0)
        for j in range(0, 4, 2):
            xb = x[j:j + 2]
            bank.set_active(False)
            with torch.no_grad():
                t = tiny_llama(input_ids=xb).logits
            bank.set_active(True)
            if once:
                bank.begin_micro(torch.float32)
            else:
                bank.refresh(torch.float32, 0.0)
            s = tiny_llama(input_ids=xb).logits
            _, g = kl_and_grad(t, s, 1.0 / x.numel())
            s.backward(g)
            bank.clear()
        if once:
            bank.end_step()
        grads.append({k: p.V.grad.clone() for k, p in bank.projs.items()})
        bank.detach(tiny_llama)
    for k in grads[0]:
        assert torch.allclose(grads[0][k], grads[1][k], atol=1e-6, rtol=1e-4), k
