import os
import sys
from contextlib import nullcontext

import torch

from conftest import random_orthonormal
from fsc.arch import discover_units
from fsc.project import swapped

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import compress  # noqa: E402


def test_kl_to_dense_zero_for_identity_and_positive_for_compression(tiny_model):
    units = discover_units(tiny_model)
    bases = {u.name: random_orthonormal(u.d, max(1, u.d // 2), i) for i, u in enumerate(units)}
    parts = compress.merge_parts(tiny_model, units, bases)
    w = {"x": torch.randint(0, 97, (3, 12))}
    same = compress.kl_to_dense(tiny_model, w, nullcontext, nullcontext, 2, "cpu")
    assert abs(same["x"]) < 1e-6
    kl = compress.kl_to_dense(tiny_model, w, nullcontext,
                              lambda: swapped(tiny_model, parts, torch.float32), 2, "cpu")
    assert kl["x"] > 1e-3
    # the swap is undone after each batch
    dense = tiny_model(input_ids=w["x"]).logits
    assert torch.isfinite(dense).all()
    again = compress.kl_to_dense(tiny_model, w, nullcontext, nullcontext, 2, "cpu")
    assert abs(again["x"]) < 1e-6


def test_reeval_zs_skips_runs_already_at_the_batch_and_refuses_unfinished(tmp_path):
    import json
    import pytest
    import compress
    run = tmp_path / "run"
    run.mkdir()
    metrics = {"done": True, "final": {"zs": {"piqa": {"acc": 0.5}}, "zs_mean_acc": 0.5}}
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "config.json").write_text(json.dumps({"zs_batch": 16}))
    argv = ["--model", "m", "--run-dir", str(run), "--reeval-zs", "--zs-batch", "16"]
    assert compress.main(argv) == 0                       # nothing to do: no GPU needed
    assert json.loads((run / "metrics.json").read_text()) == metrics
    (run / "metrics.json").write_text(json.dumps({"done": False}))
    with pytest.raises(SystemExit):
        compress.main(argv)
