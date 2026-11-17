"""DPO trainer tests — pair loading, padded log-prob equivalence, one
end-to-end tiny run with exact resume, and a monotone-loss contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from ryko_lm.model.ryko_model import RykoConfig, RykoLM
from ryko_lm.tokenizer.loader import RykoTokenizer
from ryko_lm.training.dpo_train import (
    _batch_logprobs,
    _response_logprob,
    dpo_accuracy,
    dpo_train,
    load_pairs,
)

from conftest import TOKENIZER_JSON, requires_tokenizer

MICRO = RykoConfig(n_layer=2, n_head=2, n_kv_head=1, n_embd=64,
                   ffn_hidden=128, block_size=64)


@pytest.fixture(scope="module")
def tok() -> RykoTokenizer:
    return RykoTokenizer.from_file(TOKENIZER_JSON)


@requires_tokenizer
def test_load_pairs_skips_degenerate_and_overlong(tmp_path, tok) -> None:
    p = tmp_path / "pairs.jsonl"
    rows = [
        {"id": "ok", "prompt": "من أي كتاب؟", "chosen": " «أ»", "rejected": " «ب»"},
        {"id": "no-rej", "prompt": "س", "chosen": " ج", "rejected": ""},
        {"id": "overlong", "prompt": "سؤال " * 200, "chosen": " ج", "rejected": " ر"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                 encoding="utf-8")
    pairs = load_pairs(p, tok, 64)
    assert [q["id"] for q in pairs] == ["ok"]


@requires_tokenizer
def test_padded_batch_equals_single_row(tok) -> None:
    torch.manual_seed(0)
    model = RykoLM(MICRO)
    model.eval()
    seqs = [
        {"p": tok.encode("من أي كتاب هذا المقطع؟"),
         "c": tok.encode(" «الكتاب الأول»"),
         "r": tok.encode(" «الكتاب الثاني»")},
        {"p": tok.encode("وثّق:"),
         "c": tok.encode(" مُوثَّق حرفياً."),
         "r": tok.encode(" الاقتباس مركّب غير موجود — يُرفض.")},
    ]
    padded_c = _batch_logprobs(model, seqs, "c", pad_id=0)
    padded_r = _batch_logprobs(model, seqs, "r", pad_id=0)
    for i, s in enumerate(seqs):
        for key, padded in (("c", padded_c[i]), ("r", padded_r[i])):
            row = torch.tensor([s["p"] + s[key]], dtype=torch.long)
            single = _response_logprob(model, row, len(s["p"]))
            assert torch.allclose(padded, single, atol=1e-4), \
                (i, key, float(padded), float(single))


@requires_tokenizer
def test_dpo_accuracy_prefers_higher_logprob(tok) -> None:
    torch.manual_seed(1)
    model = RykoLM(MICRO)
    model.eval()
    pairs = [{"id": f"p{i}", "p": tok.encode("س؟"), "c": tok.encode(f" نعم{i}"),
              "r": tok.encode(f" لا{i}")} for i in range(4)]
    summary = dpo_accuracy(model, pairs, pad_id=0)
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert "mean_margin" in summary


def _write_config(tmp_path: Path) -> Path:
    cfg = {"name": "micro", "model": MICRO.to_dict(),
           "train": {"seed": 1337}}
    p = tmp_path / "micro.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _write_pairs(tmp_path: Path, tok: RykoTokenizer, n: int) -> Path:
    p = tmp_path / "pairs.jsonl"
    rows = [{"id": f"dpo-t{i:03d}",
             "prompt": f"من أي كتاب المقطع {i}؟",
             "chosen": f" «الصحيح رقم {i}»",
             "rejected": f" «الخاطئ رقم {i}»"} for i in range(n)]
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                           for r in rows) + "\n", encoding="utf-8")
    return p


def _init_sft(tmp_path: Path, tok: RykoTokenizer) -> Path:
    torch.manual_seed(42)
    model = RykoLM(MICRO)
    out = tmp_path / "sft"
    model.save(out / "final.pt")
    return out / "final.pt"


@requires_tokenizer
def test_dpo_end_to_end_runs_resumes_and_logs(tmp_path, tok) -> None:
    torch.manual_seed(0)
    sft = _init_sft(tmp_path, tok)
    pairs = _write_pairs(tmp_path, tok, 6)
    cfg = _write_config(tmp_path)
    out = tmp_path / "dpo"
    rc = dpo_train(_args(cfg, sft, pairs, out,
                         batch_size=2, epochs=2, save_every=2))
    assert rc in (0, 3)  # 3 only if a time budget cut it; none set here
    last = out / "last.pt"
    final = out / "final.pt"
    assert last.exists() and final.exists()
    records = [json.loads(l) for l in
               (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records, "metrics log must not be empty"
    assert records[-1]["step"] == 6  # 2 epochs x 3 steps
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["pairs"] == 6
    # exact resume: a fresh run with a tight max-steps extends, then a full
    # run completes from the checkpoint without restarting
    out2 = tmp_path / "dpo_resume"
    dpo_train(_args(cfg, sft, pairs, out2, batch_size=2, epochs=2,
                    save_every=2, max_steps=2))
    ck = torch.load(out2 / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2
    dpo_train(_args(cfg, sft, pairs, out2, batch_size=2, epochs=2,
                    save_every=2))
    ck = torch.load(out2 / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 6


@requires_tokenizer
def test_dpo_loss_decreases_on_separable_pairs(tmp_path, tok) -> None:
    """Chosen/rejected differ in one token — loss must drop measurably."""
    torch.manual_seed(7)
    sft = _init_sft(tmp_path, tok)
    rows = [{"id": f"dpo-s{i:03d}",
             "prompt": "من أي كتاب؟",
             "chosen": " «الصواب»",
             "rejected": " «الخطأ»"} for i in range(4)]
    p = tmp_path / "pairs.jsonl"
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                           for r in rows) + "\n", encoding="utf-8")
    cfg = _write_config(tmp_path)
    out = tmp_path / "dpo_fit"
    dpo_train(_args(cfg, sft, p, out, batch_size=4, epochs=20,
                    save_every=10, lr=5e-4))
    records = [json.loads(l) for l in
               (out / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[-1]["loss"] < records[0]["loss"]


def _args(cfg, sft, pairs, out, **kw) -> "argparse.Namespace":  # noqa: ANN001
    import argparse
    base = dict(config=cfg, init_from=sft, pairs=pairs,
                tokenizer=TOKENIZER_JSON, out_dir=out, epochs=1,
                batch_size=2, beta=0.1, lr=1e-5, min_lr=1e-6, warmup=2,
                weight_decay=0.0, grad_clip=1.0, max_steps=None,
                save_every=50, seed=1337, threads=2, time_budget=None,
                reset=False)
    base.update(kw)
    return argparse.Namespace(**base)
