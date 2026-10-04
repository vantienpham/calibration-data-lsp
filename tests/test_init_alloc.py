import torch

from fsc.alloc import greedy_alloc, uniform_alloc
from fsc.arch import Unit, discover_units, realized_ratio
from fsc.init import InBasis, OutBasis, robust_cholesky
from fsc.stats import collect_grams


def _spd(d, seed=0, cond=50.0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    ev = torch.logspace(0, -torch.log10(torch.tensor(cond)).item(), d, dtype=torch.float64)
    return q @ torch.diag(ev) @ q.T


def _err(W, What, G):
    D = (W - What)
    return float(torch.trace(D @ G @ D.T))


def test_input_side_recast_matches_proposition():
    """E(W P_init) = sum_{i>r} s_i^2 + ||S_r C_r^+ C_{>r}||_F^2 (Bini et al., Prop. 1(ii))."""
    torch.manual_seed(0)
    d, m, k = 12, 20, 5
    W = torch.randn(m, d, dtype=torch.float64)
    G = _spd(d)
    b = InBasis([W], G)
    U = b.removed(k)
    P = torch.eye(d, dtype=torch.float64) - U @ U.T
    err = _err(W, W @ P, G)
    r = d - k
    C_r, C_t = b.S @ b.Vw[:, :r], b.S @ b.Vw[:, r:]
    excess = float(((torch.diag(b.sv[:r]) @ torch.linalg.pinv(C_r) @ C_t) ** 2).sum())
    assert abs(err - (b.whitened_error(k) + excess)) < 1e-8 * max(err, 1.0)


def test_isotropic_input_recast_is_optimal():
    torch.manual_seed(1)
    d, m, k = 10, 16, 4
    W = torch.randn(m, d, dtype=torch.float64)
    G = torch.eye(d, dtype=torch.float64) * 3.0
    b = InBasis([W], G)
    U = b.removed(k)
    P = torch.eye(d, dtype=torch.float64) - U @ U.T
    assert abs(_err(W, W @ P, G) - b.whitened_error(k)) < 1e-8


def test_output_side_is_optimal():
    torch.manual_seed(2)
    d_out, d_in, k = 8, 20, 3
    W = torch.randn(d_out, d_in, dtype=torch.float64)
    G = _spd(d_in, 3)
    b = OutBasis(W @ G @ W.T)
    U = b.removed(k)
    P = torch.eye(d_out, dtype=torch.float64) - U @ U.T
    S = torch.linalg.cholesky(G)
    sv = torch.linalg.svdvals(W @ S)
    assert abs(_err(W, P @ W, G) - float((sv[-k:] ** 2).sum())) < 1e-8


def test_cholesky_ridge_on_singular():
    G = torch.zeros(4, 4, dtype=torch.float64)
    G[0, 0] = 1.0
    L, eta = robust_cholesky(G)
    assert eta > 0 and torch.isfinite(L).all()


def test_grams_match_manual(tiny_llama):
    units = discover_units(tiny_llama)
    x = torch.randint(0, 97, (3, 10))
    grams, n = collect_grams(tiny_llama, units, x, batch_size=2, device="cpu")
    assert n == 30
    seen = {}

    def grab(_m, a):
        seen["x"] = a[0].reshape(-1, a[0].shape[-1]).double()   # returns None: input unchanged

    h = tiny_llama.model.layers[0].self_attn.q_proj.register_forward_pre_hook(grab)
    tiny_llama.model(input_ids=x)
    h.remove()
    X = seen["x"]
    assert torch.allclose(grams["L0.qkv"], X.T @ X / n, atol=1e-8)


def test_uniform_alloc_meets_target(tiny_llama):
    units = discover_units(tiny_llama)
    for target in (0.3, 0.5, 0.7):
        ks = uniform_alloc(units, target)
        r = realized_ratio(units, ks)
        assert target <= r < target + 0.08, (target, r)


def test_greedy_prefers_cheap_units():
    a = Unit("a", "qkv", 0, "in", ["a"], 16, [16])
    b = Unit("b", "qkv", 0, "in", ["b"], 16, [16])
    grid = [2, 4, 6, 8, 10, 12, 14]
    curves = {"a": {k: 0.001 * k for k in grid}, "b": {k: 0.1 * k for k in grid}}
    ks, pred = greedy_alloc([a, b], curves, 0.2)
    assert ks["a"] > 0 and ks["b"] == 0
    assert realized_ratio([a, b], ks) >= 0.2
    # trimmed: one rank less on the last unit misses the target
    ks2 = dict(ks)
    ks2["a"] -= 1
    assert realized_ratio([a, b], ks2) < 0.2


def test_oblique_factors_attain_whitened_optimum():
    torch.manual_seed(4)
    d, k = 12, 5
    W1, W2 = torch.randn(10, d, dtype=torch.float64), torch.randn(6, d, dtype=torch.float64)
    G = _spd(d, 5)
    b = InBasis([W1, W2], G)
    A, (B1, B2) = b.oblique_factors([W1, W2], k)
    assert A.shape == (d - k, d) and B1.shape == (10, d - k)
    err = _err(W1, B1 @ A, G) + _err(W2, B2 @ A, G)
    assert abs(err - b.whitened_error(k)) < 1e-8 * max(err, 1.0)
    # the orthogonal recast is never better than the oblique optimum
    U = b.removed(k)
    P = torch.eye(d, dtype=torch.float64) - U @ U.T
    assert _err(W1, W1 @ P, G) + _err(W2, W2 @ P, G) >= err - 1e-9
