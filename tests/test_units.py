from fsc.arch import Unit, discover_units, realized_ratio, total_dense


def test_every_linear_in_exactly_one_unit(tiny_model):
    import torch.nn as nn
    units = discover_units(tiny_model)
    members = [p for u in units for p in u.members]
    assert len(members) == len(set(members))
    linears = {n for n, m in tiny_model.named_modules() if isinstance(m, nn.Linear)}
    head = {n for n in linears if n.endswith("lm_head")}
    other = {n for n in linears if "project_" in n}   # OPT's optional projections
    assert set(members) == linears - head - other


def test_sides_and_ties(tiny_llama):
    units = {u.name: u for u in discover_units(tiny_llama)}
    assert units["L0.qkv"].side == "in" and len(units["L0.qkv"].members) == 3
    assert units["L0.gateup"].side == "in" and len(units["L0.gateup"].members) == 2
    assert units["L0.down"].side == "out"            # 72 -> 32: output is the smaller side
    assert units["L0.o"].side == "in"                # square: input side
    untied = discover_units(tiny_llama, tie=False)
    assert all(len(u.members) == 1 for u in untied)


def test_param_accounting():
    u = Unit("x", "qkv", 0, "in", ["a", "b", "c"], d_in=32, d_outs=[32, 16, 16])
    assert u.dense_params == 32 * 64
    assert u.params(0) == u.dense_params
    assert u.params(10) == (32 - 10) * (32 + 64)
    k = u.break_even_k()
    assert u.savings(k) > 0 and u.savings(k - 1) <= 0
    v = Unit("y", "down", 0, "out", ["d"], d_in=72, d_outs=[32])
    assert v.d == 32 and v.params(8) == 24 * (72 + 32)


def test_realized_ratio(tiny_llama):
    units = discover_units(tiny_llama)
    assert realized_ratio(units, {}) == 0.0
    ks = {u.name: u.d - 1 for u in units}
    assert 0.9 < realized_ratio(units, ks) < 1.0
    assert total_dense(units) > 0
