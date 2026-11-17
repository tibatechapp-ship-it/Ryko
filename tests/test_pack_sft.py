"""pack_sft tests — bin layout, byte-rebuild determinism, train.py contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import TOKENIZER_JSON, requires_tokenizer
from ryko_lm.training.pack_sft import build, pack_file
from ryko_lm.tokenizer.loader import EOS_ID, RykoTokenizer


@pytest.fixture()
def tok() -> RykoTokenizer:
    return RykoTokenizer.from_file(TOKENIZER_JSON)


def _write_pairs(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


@requires_tokenizer
def test_pack_layout_eos_separated(tmp_path, tok) -> None:
    src = tmp_path / "sft_train.jsonl"
    _write_pairs(src, [
        {"id": "sft-a", "prompt": "السؤال: س؟\nالجواب:", "completion": " ج."},
        {"id": "sft-b", "prompt": "السؤال: س2؟\nالجواب:", "completion": " ج2."},
    ])
    out = tmp_path / "train.bin"
    stats = pack_file(tok, src, out)
    assert stats["pairs"] == 2 and stats["skipped"] == []
    ids = np.fromfile(out, dtype=np.uint16)
    a = tok.encode("السؤال: س؟\nالجواب:") + tok.encode(" ج.")
    b = tok.encode("السؤال: س2؟\nالجواب:") + tok.encode(" ج2.")
    assert list(ids) == a + [EOS_ID] + b + [EOS_ID]
    assert stats["tokens"] == len(a) + 1 + len(b) + 1


@requires_tokenizer
def test_pack_skips_degenerate_pairs(tmp_path, tok) -> None:
    src = tmp_path / "sft_train.jsonl"
    _write_pairs(src, [
        {"id": "sft-ok", "prompt": "س؟", "completion": " ج."},
        {"id": "sft-empty", "prompt": "س؟", "completion": ""},
    ])
    out = tmp_path / "train.bin"
    stats = pack_file(tok, src, out)
    assert stats["pairs"] == 1
    assert stats["skipped"] == ["sft-empty"]


@requires_tokenizer
def test_build_byte_rebuildable_and_val_contract(tmp_path, tok) -> None:
    train, dev = tmp_path / "t.jsonl", tmp_path / "d.jsonl"
    _write_pairs(train, [{"id": "a", "prompt": "س", "completion": " ج"}])
    _write_pairs(dev, [{"id": "b", "prompt": "س2", "completion": " ج2"}])
    out1 = tmp_path / "run1"
    manifest = build(train, dev, TOKENIZER_JSON, out1)
    # the filename contract of ryko_lm.training.train
    assert (out1 / "train.bin").exists() and (out1 / "val.bin").exists()
    assert manifest["files"]["train"]["pairs"] == 1
    assert manifest["files"]["val"]["pairs"] == 1
    assert manifest["tokens_total"] == (manifest["files"]["train"]["tokens"]
                                        + manifest["files"]["val"]["tokens"])
    # determinism: a second independent build is byte-identical
    out2 = tmp_path / "run2"
    manifest2 = build(train, dev, TOKENIZER_JSON, out2)
    for name in ("train", "val"):
        assert manifest["files"][name]["sha256"] == \
            manifest2["files"][name]["sha256"]
