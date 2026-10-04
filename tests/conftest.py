import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def _tiny(kind: str):
    from transformers import (LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM,
                              Qwen3Config, Qwen3ForCausalLM)
    torch.manual_seed(0)
    if kind == "llama":
        cfg = LlamaConfig(vocab_size=97, hidden_size=32, intermediate_size=72, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        m = LlamaForCausalLM(cfg)
    elif kind == "qwen3":
        cfg = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=72, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                          max_position_embeddings=128)
        m = Qwen3ForCausalLM(cfg)
    elif kind == "opt":
        cfg = OPTConfig(vocab_size=97, hidden_size=32, ffn_dim=64, num_hidden_layers=2,
                        num_attention_heads=4, max_position_embeddings=128, word_embed_proj_dim=32,
                        do_layer_norm_before=True, dropout=0.0)
        m = OPTForCausalLM(cfg)
    else:
        raise ValueError(kind)
    # Larger weights than HF's 0.02 init, so the logits are far from uniform and
    # an incorrect projection shows up clearly in the comparisons.
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.2)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture(params=["llama", "qwen3", "opt"])
def tiny_model(request):
    return _tiny(request.param)


@pytest.fixture
def tiny_llama():
    return _tiny("llama")


@pytest.fixture
def tiny_opt():
    return _tiny("opt")


def random_orthonormal(d, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
    return q
