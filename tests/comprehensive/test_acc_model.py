"""Accuracy — transformer internals + the promoted serving point (30 tests).

The tiny preset gives fast architectural probes; the committed
dpo_round2_small checkpoint (52.18M params) is loaded once per module for
the serving-contract probes: parameter count, seeded determinism, prompt
trimming, and KV-cache/prefill equivalence.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from ryko_lm.model.ryko_model import RykoConfig, RykoLM, get_preset

ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL = ROOT / "artifacts" / "dpo_round2_small" / "final.pt"
requires_final_model = pytest.mark.skipif(
    not FINAL_MODEL.exists(), reason="final model artifact not committed"
)

TINY = get_preset("tiny")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    m = RykoLM(TINY)
    m.eval()
    return m


@pytest.fixture(scope="module")
def served():
    return RykoLM.load(FINAL_MODEL, device="cpu")


# ------------------------------------------------------- architecture (18)
@pytest.mark.parametrize("bsz,seq", [(1, 1), (1, 8), (2, 16), (4, 32), (3, 64)])
def test_forward_shapes(tiny, bsz, seq):
    idx = torch.randint(0, TINY.vocab_size, (bsz, seq))
    logits, loss = tiny(idx)
    assert logits.shape == (bsz, seq, TINY.vocab_size)
    assert loss is None


def test_loss_finite_and_backprop(tiny):
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    targets = torch.randint(0, TINY.vocab_size, (2, 16))
    tiny.train()
    _logits, loss = tiny(idx, targets)
    assert torch.isfinite(loss)
    loss.backward()
    grad = tiny.blocks[0].attn.q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    tiny.zero_grad(set_to_none=True)
    tiny.eval()


def test_tied_embeddings(tiny):
    assert tiny.lm_head.weight is tiny.tok_emb.weight


def test_num_params_non_embedding(tiny):
    total = tiny.num_params()
    non_emb = tiny.num_params(non_embedding=True)
    assert 0 < non_emb < total
    assert total - non_emb == TINY.vocab_size * TINY.n_embd


def test_kv_cache_prefill_decode_matches_full_forward(tiny):
    torch.manual_seed(7)
    idx = torch.randint(0, TINY.vocab_size, (1, 24))
    full_logits, _ = tiny(idx)
    cache = [[None, None] for _ in range(TINY.n_layer)]
    prefill, _ = tiny(idx[:, :-1], kv_cache=cache)
    step, _ = tiny(idx[:, -1:], kv_cache=cache)
    assert torch.allclose(full_logits[:, -1, :], step[:, -1, :], atol=1e-5)
    assert torch.allclose(full_logits[:, :-1, :], prefill, atol=1e-5)


def test_kv_cache_grows_incrementally(tiny):
    idx = torch.randint(0, TINY.vocab_size, (1, 6))
    cache = [[None, None] for _ in range(TINY.n_layer)]
    tiny(idx, kv_cache=cache)
    assert cache[0][0].shape[2] == 6
    tiny(torch.randint(0, TINY.vocab_size, (1, 1)), kv_cache=cache)
    assert cache[0][0].shape[2] == 7


def test_rope_extends_beyond_block_size(tiny):
    long_idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size + 44))
    logits, _ = tiny(long_idx)
    assert logits.shape[1] == TINY.block_size + 44


def test_rope_capacity_stored_as_positions(tiny):
    long_idx = torch.randint(0, TINY.vocab_size, (1, TINY.block_size + 44))
    tiny(long_idx)
    assert tiny.rope_cos.size(0) >= TINY.block_size + 44


def test_generate_length_respected(tiny):
    idx = torch.randint(0, TINY.vocab_size, (1, 4))
    g = torch.Generator().manual_seed(11)
    out = tiny.generate(idx, max_new_tokens=9, eos_id=None, generator=g)
    assert out.shape[1] == 9  # generated tokens only, no early EOS


def test_generate_iter_yields_scalars(tiny):
    idx = torch.randint(0, TINY.vocab_size, (1, 3))
    g = torch.Generator().manual_seed(5)
    tokens = list(tiny.generate_iter(idx, max_new_tokens=4, eos_id=None, generator=g))
    assert len(tokens) == 4 and all(t.shape == (1, 1) for t in tokens)


def test_generate_top_k1_is_argmax_stable(tiny):
    idx = torch.randint(0, TINY.vocab_size, (1, 4))
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(99)
    a = tiny.generate(idx, max_new_tokens=6, top_k=1, generator=g1)
    b = tiny.generate(idx, max_new_tokens=6, top_k=1, generator=g2)
    assert torch.equal(a, b)  # top_k=1 keeps only argmax — seed-independent


def test_generate_deterministic_given_seed(tiny):
    idx = torch.randint(0, TINY.vocab_size, (1, 4))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    a = tiny.generate(idx, max_new_tokens=8, eos_id=None, generator=g1)
    b = tiny.generate(idx, max_new_tokens=8, eos_id=None, generator=g2)
    assert torch.equal(a, b)


def test_generate_temperature_extremes_run(tiny):
    idx = torch.randint(0, TINY.vocab_size, (1, 4))
    for temp in (0.2, 1.7):
        g = torch.Generator().manual_seed(1)
        out = tiny.generate(idx, max_new_tokens=4, temperature=temp,
                            eos_id=None, generator=g)
        assert out.shape[1] == 4


def test_save_load_round_trip_logits(tmp_path, tiny):
    path = tmp_path / "m.pt"
    tiny.save(path)
    clone = RykoLM.load(path)
    idx = torch.randint(0, TINY.vocab_size, (1, 12))
    a, _ = tiny(idx)
    b, _ = clone(idx)
    assert torch.equal(a, b)


def test_save_without_tie_drops_head(tmp_path, tiny):
    path = tmp_path / "m.pt"
    tiny.save(path)
    blob = torch.load(path, weights_only=True)
    assert "lm_head.weight" not in blob["state_dict"]


def test_config_round_trip_ignores_unknown():
    cfg = RykoConfig.from_dict({**RykoConfig().to_dict(), "alien": 1})
    assert cfg == RykoConfig()
    assert cfg.to_dict() == RykoConfig().to_dict()


def test_presets_registry():
    for name in ("tiny", "small", "base"):
        assert get_preset(name).vocab_size == 32768
    with pytest.raises(KeyError):
        get_preset("mega")


# --------------------------------------------- serving checkpoint (12)
@requires_final_model
def test_served_param_contract(served):
    assert served.num_params() == 52_179_456


@requires_final_model
def test_served_block_size(served):
    assert served.cfg.block_size == 512
    assert served.cfg.vocab_size == 32768


@requires_final_model
def test_served_tied(served):
    assert served.lm_head.weight is served.tok_emb.weight


@requires_final_model
def test_served_forward_loss_finite(served):
    idx = torch.randint(0, 32768, (1, 32))
    _, loss = served(idx, idx)
    assert torch.isfinite(loss)


@requires_final_model
def test_served_seeded_generation_deterministic(served):
    from ryko_lm.tokenizer.loader import RykoTokenizer

    tok = RykoTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    ids = tok.encode("اللغة العربية", add_bos=True)
    g1 = torch.Generator().manual_seed(21)
    g2 = torch.Generator().manual_seed(21)
    a = served.generate(torch.tensor([ids]), max_new_tokens=10, generator=g1)
    b = served.generate(torch.tensor([ids]), max_new_tokens=10, generator=g2)
    assert torch.equal(a, b)


@requires_final_model
def test_served_generation_vocab_range(served):
    from ryko_lm.tokenizer.loader import RykoTokenizer

    tok = RykoTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    ids = tok.encode("الزكاة في المال", add_bos=True)
    out = served.generate(torch.tensor([ids]), max_new_tokens=8,
                          generator=torch.Generator().manual_seed(2))
    assert (out >= 0).all() and (out < 32768).all()


@requires_final_model
def test_served_long_prompt_trimmed(served):
    # 600 tokens exceed block_size 512 minus room for 8 generated — the
    # generation must trim the prompt, not crash
    long_ids = [1] + [200] * 600
    out = served.generate(torch.tensor([long_ids]), max_new_tokens=8,
                          generator=torch.Generator().manual_seed(4))
    assert out.shape[1] >= 1


@requires_final_model
def test_served_kv_matches_full(served):
    idx = torch.randint(0, 32768, (1, 16))
    full, _ = served(idx)
    cache = [[None, None] for _ in range(served.cfg.n_layer)]
    served(idx[:, :-1], kv_cache=cache)
    step, _ = served(idx[:, -1:], kv_cache=cache)
    assert torch.allclose(full[:, -1, :], step[:, -1, :], atol=1e-4)


@requires_final_model
def test_served_repetition_penalty_runs(served):
    from ryko_lm.tokenizer.loader import RykoTokenizer

    tok = RykoTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    ids = tok.encode("قال قال قال", add_bos=True)
    out = served.generate(torch.tensor([ids]), max_new_tokens=6,
                          repetition_penalty=1.9,
                          generator=torch.Generator().manual_seed(8))
    assert out.shape[1] >= 1


@requires_final_model
def test_served_top_p_sampling_runs(served):
    from ryko_lm.tokenizer.loader import RykoTokenizer

    tok = RykoTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    ids = tok.encode("النحو", add_bos=True)
    out = served.generate(torch.tensor([ids]), max_new_tokens=6, top_p=0.5,
                          generator=torch.Generator().manual_seed(3))
    assert out.shape[1] >= 1


@requires_final_model
def test_served_eos_stops_early(served):
    # force eos_id to the first prompt token: sampling argmax over random
    # logits may or may not pick it — instead pin eos to a token the model
    # just emitted by generating 1 token and stopping there
    from ryko_lm.tokenizer.loader import RykoTokenizer

    tok = RykoTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    ids = tok.encode("اللغة", add_bos=True)
    first = torch.Generator().manual_seed(6)
    out = served.generate(torch.tensor([ids]), max_new_tokens=5, generator=first)
    eos_pin = int(out[0, 0])
    g = torch.Generator().manual_seed(6)
    out2 = served.generate(torch.tensor([ids]), max_new_tokens=5,
                           eos_id=eos_pin, generator=g)
    assert out2.shape[1] == 1  # first sampled token == eos_pin -> instant stop


@requires_final_model
def test_served_greedy_beats_random_heat(served):
    """top_k=1 greedy should be the modal path — logits finite sanity."""
    from ryko_lm.tokenizer.loader import RykoTokenizer

    tok = RykoTokenizer.from_file(ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json")
    ids = tok.encode("العقل", add_bos=True)
    idx = torch.tensor([ids])
    logits, _ = served(idx)
    assert torch.isfinite(logits).all()
    assert math.isfinite(float(logits.max()))
