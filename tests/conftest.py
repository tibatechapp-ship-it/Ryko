"""Shared pytest fixtures: repository root + artifact paths."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "ryko-tokenizer.json"
# v1.24.0 promotion: the serving point is the receipted round-2 checkpoint
# (artifacts/dpo_round2_small/final.pt, 52.18M params — the v1.23.0 point:
# SFT+DPO + ling merge + DPO round 2), the same default the API serves (see
# ryko_lm.serving.api.MODEL_PATH).  Promotion ledger: artifacts/training/
# final.pt (v1.1 archive, PPL 431.78) -> artifacts/training_v5c/final.pt
# (11.5M, PPL 221.38, since v1.17.0) -> artifacts/dpo_round2_small/final.pt
# (since v1.24.0) — the serving chain and the measurement chain are pinned
# together by
# test_guard_provenance.test_serving_default_is_the_promoted_round2_point.
FINAL_MODEL = ROOT / "artifacts" / "dpo_round2_small" / "final.pt"
SAMPLE_DIR = ROOT / "data" / "sample"

requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(), reason="tokenizer artifact not committed"
)
requires_final_model = pytest.mark.skipif(
    not FINAL_MODEL.exists(), reason="final model artifact not committed"
)
