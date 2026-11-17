"""Shared pytest fixtures: repository root + artifact paths."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json"
# v1.17.0 fix: the serving point is the receipted v5c checkpoint (held-out
# PPL 221.38), the same default the API serves (see
# ryko_lm.serving.api.MODEL_PATH).  The old artifacts/training/final.pt is a
# v1.1-era archive (PPL 431.78) that the first PalmX-IC runs measured by
# mistake — guarded against regression by
# test_guard_provenance.test_serving_default_checkpoint_is_receipted.
FINAL_MODEL = ROOT / "artifacts" / "training_v5c" / "final.pt"
SAMPLE_DIR = ROOT / "data" / "sample"

requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(), reason="tokenizer artifact not committed"
)
requires_final_model = pytest.mark.skipif(
    not FINAL_MODEL.exists(), reason="final model artifact not committed"
)
