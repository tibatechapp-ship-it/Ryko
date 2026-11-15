"""Shared pytest fixtures: repository root + artifact paths."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json"
FINAL_MODEL = ROOT / "artifacts" / "training" / "final.pt"
SAMPLE_DIR = ROOT / "data" / "sample"

requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(), reason="tokenizer artifact not committed"
)
requires_final_model = pytest.mark.skipif(
    not FINAL_MODEL.exists(), reason="final model artifact not committed"
)
