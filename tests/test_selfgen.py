import torch

from fsc import data, selfgen


class StubTok:
    bos_token_id = 1
    eos_token_id = 2

    def __len__(self):
        return 97


def test_sample_documents_end_at_eos_or_max_len(tiny_llama):
    docs = selfgen.sample_documents(tiny_llama, StubTok(), n_tokens=200, max_len=16,
                                    batch_size=8, seed=0, device="cpu", log=lambda *_: None)
    assert sum(len(d) + 1 for d in docs) >= 200
    assert all(0 < len(d) <= 16 for d in docs)
    assert all(StubTok.eos_token_id not in d for d in docs)


def test_sample_windows_start_with_start_token(tiny_llama):
    ids = selfgen.sample(tiny_llama, StubTok(), n=3, seqlen=10, batch_size=2, seed=0,
                         device="cpu", log=lambda *_: None)
    assert ids.shape == (3, 10) and (ids[:, 0] == StubTok.bos_token_id).all()


def test_selfdoc_windows_pack_disjoint_documents(tmp_path):
    docs = [[10 + i] * (3 + i % 5) for i in range(200)]
    blob = {"docs_flat": torch.tensor([t for d in docs for t in d]),
            "doc_lengths": torch.tensor([len(d) for d in docs]),
            "env": {"torch": torch.__version__}}   # a TorchVersion, as real pools store
    path = str(tmp_path / "docs.pt")
    torch.save(blob, path)
    tr, va = data.calibration_windows(StubTok(), "selfdoc", 4, 2, 9, seed=0, selfgen_path=path)
    assert tr.shape == (4, 9) and va.shape == (2, 9)
    assert (tr[:, 0] == 1).all() and (va[:, 0] == 1).all()        # BOS-prefixed
    tr_docs = set(tr[:, 1:].flatten().tolist()) - {2}
    va_docs = set(va[:, 1:].flatten().tolist()) - {2}
    assert not (tr_docs & va_docs)                                  # disjoint documents


class StrTok(StubTok):
    """Tokenizes 'w w w' strings into ids 50.. by word length."""

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[50 + len(w) for w in t.split()] for t in texts]}


def test_doc_bos_packing_starts_each_document_with_bos():
    docs = ["aa bbb", "c dd", "eee f g"]
    stream, _ = data._pack(StrTok(), docs, 10, eos=2, doc_bos=1)
    assert stream[:4] == [1, 52, 53, 2] and stream[4:8] == [1, 51, 52, 2]
    plain, _ = data._pack(StrTok(), docs, 6, eos=2)
    assert 1 not in plain
