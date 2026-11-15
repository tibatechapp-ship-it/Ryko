"""Architecture tests for RykoLM (random weights — no artifacts required)."""

import torch

from ryko_lm.model.ryko_model import RykoConfig, RykoLM, get_preset


def tiny(**overrides) -> RykoLM:
    cfg = RykoConfig(vocab_size=512, block_size=64, n_layer=2, n_head=4,
                     n_kv_head=2, n_embd=64, ffn_hidden=128, **overrides)
    torch.manual_seed(0)
    return RykoLM(cfg)


def test_forward_shapes_and_loss():
    model = tiny()
    idx = torch.randint(0, 512, (2, 32))
    logits, loss = model(idx, torch.randint(0, 512, (2, 32)))
    assert logits.shape == (2, 32, 512)
    assert loss.item() > 0.0
    assert abs(loss.item() - torch.log(torch.tensor(512.0)).item()) < 1.0


def test_causal_masking_hides_the_future():
    model = tiny().eval()
    idx = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        full, _ = model(idx)
        past_logits, _ = model(idx[:, :8])
    delta = (full[0, :8] - past_logits[0]).abs().max()
    assert delta.item() < 1e-4, "past logits must not depend on future tokens"


def test_kv_cache_matches_full_forward():
    model = tiny().eval()
    idx = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        full, _ = model(idx)  # (1, 12, V)

        cache = [[None, None] for _ in range(model.cfg.n_layer)]
        pieces = []
        for t in range(12):  # incremental decode
            logits, _ = model(idx[:, t: t + 1], kv_cache=cache)
            pieces.append(logits)
        incremental = torch.cat(pieces, dim=1)

    assert (full - incremental).abs().max().item() < 1e-4


def test_tied_embeddings():
    model = tiny(tie_embeddings=True)
    assert model.lm_head.weight is model.tok_emb.weight


def test_save_load_roundtrip_deterministic(tmp_path):
    model = tiny()
    path = tmp_path / "m.pt"
    model.save(path, half=True)
    loaded = RykoLM.load(path)
    idx = torch.randint(0, 512, (1, 8))
    generator = torch.Generator().manual_seed(3)
    out_a = loaded.generate(idx, max_new_tokens=10, temperature=1.0,
                            top_k=None, top_p=None, repetition_penalty=1.0,
                            generator=generator)
    generator = torch.Generator().manual_seed(3)
    out_b = loaded.generate(idx, max_new_tokens=10, temperature=1.0,
                            top_k=None, top_p=None, repetition_penalty=1.0,
                            generator=generator)
    assert torch.equal(out_a, out_b)


def test_generation_shape_and_stops_at_eos():
    model = tiny()
    prompt = torch.tensor([[1, 5, 6]])
    out = model.generate(prompt, max_new_tokens=20, temperature=1.0, top_k=10)
    assert out.dim() == 2 and out.size(0) == 1 and 1 <= out.size(1) <= 20


def test_presets_scale_monotonically():
    counts = {name: RykoLM(get_preset(name)).num_params()
              for name in ("tiny", "small", "base")}
    assert counts["tiny"] < counts["small"] < counts["base"]


def test_config_roundtrip_ignores_unknown_keys():
    cfg = RykoConfig.from_dict({"vocab_size": 100, "alien_key": 1})
    assert cfg.vocab_size == 100
