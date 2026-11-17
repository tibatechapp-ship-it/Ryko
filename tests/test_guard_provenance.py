"""Tests for the ZYKO-adopted modules: provenance receipts + generation guard."""

from __future__ import annotations

import json

import pytest

from ryko_lm.provenance import make_receipt, verify_receipt
from ryko_lm.serving.guard import guard_output, seal


# ---------------------------------------------------------------------------
# provenance receipts (ZYKO pattern)
# ---------------------------------------------------------------------------
def test_receipt_roundtrip(tmp_path):
    target = tmp_path / "artifacts"
    target.mkdir()
    (target / "train.bin").write_bytes(b"\x01\x02\x03\x04")
    (target / "stats.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "receipt.json"
    receipt = make_receipt(target, kind="corpus", meta={"tokens": 42})
    out.write_text(json.dumps(receipt), encoding="utf-8")

    ok, report = verify_receipt(out)
    assert ok
    assert all(r["ok"] for r in report)
    assert len(report) == 3  # receipt hash + 2 artifacts


def test_receipt_detects_tampering(tmp_path):
    target = tmp_path / "artifacts"
    target.mkdir()
    (target / "train.bin").write_bytes(b"\x01\x02\x03\x04")
    out = tmp_path / "receipt.json"
    out.write_text(json.dumps(make_receipt(target)), encoding="utf-8")

    (target / "train.bin").write_bytes(b"\xff\xff\xff\xff")  # tamper
    ok, report = verify_receipt(out)
    assert not ok
    assert any(not r["ok"] for r in report)


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    from ryko_lm.provenance import sha256_file

    f = tmp_path / "x.bin"
    f.write_bytes(b"ryko")
    assert sha256_file(f) == hashlib.sha256(b"ryko").hexdigest()


def test_serving_default_checkpoint_is_receipted():
    """v1.17.0 regression guard: the checkpoint the API serves by default
    must exist AND carry a verified receipt whose sha256 matches the actual
    bytes.  The old artifacts/training/final.pt default silently served the
    stale v1.2-era checkpoint (PPL 431.78) — both published PalmX-IC runs
    measured it by mistake (random-level score) — so an unreceipted or stale
    serving default must now fail loudly."""
    from ryko_lm.provenance import verify_receipt
    from ryko_lm.serving.api import MODEL_PATH

    assert MODEL_PATH.exists(), f"serving checkpoint missing: {MODEL_PATH}"
    receipt_path = MODEL_PATH.with_name("final_receipt.json")
    assert receipt_path.exists(), (
        f"serving checkpoint has no receipt: {MODEL_PATH} — every served "
        "artifact must be receipted (ZYKO pattern)")
    ok, report = verify_receipt(receipt_path)
    assert ok, f"receipt self-check failed: {report}"
    artifact_checks = [r for r in report if r["check"] != "receipt_sha256"]
    assert artifact_checks, f"receipt lists no artifacts: {report}"
    assert all(r["ok"] for r in artifact_checks), (
        f"receipted bytes do not match the file: {report}")


# ---------------------------------------------------------------------------
# generation guard (ZYKO pattern)
# ---------------------------------------------------------------------------
def test_guard_allows_good_arabic():
    text = "قال رسول الله صلى الله عليه وسلم إن الصدق يهدي إلى البر، وإن الكذب يهدي إلى الفجور."
    verdict = guard_output(text, prompt="قال رسول الله")
    assert verdict["decision"] == "ALLOW"
    assert verdict["reasons"] == []


def test_guard_reviews_short_output():
    verdict = guard_output("قال")
    assert verdict["decision"] == "REVIEW"
    assert "too_short" in verdict["reasons"]


def test_guard_reviews_repetition_loop():
    text = "قال قال قال " * 20
    verdict = guard_output(text)
    assert verdict["decision"] == "REVIEW"
    assert any(r.startswith("repetition_loop") for r in verdict["reasons"])


def test_guard_reviews_residual_markup():
    text = "هذا نص عربي طويل بما يكفي ليمرر الفحص الأساسي دون مشاكل %~12# ثم يكمل."
    verdict = guard_output(text)
    assert verdict["decision"] == "REVIEW"
    assert "residual_markup" in verdict["reasons"]


def test_guard_seal_deterministic():
    assert seal("م", "ن") == seal("م", "ن")
    assert seal("م", "ن") != seal("م", "س")


# ---------------------------------------------------------------------------
# eval helpers
# ---------------------------------------------------------------------------
def test_eval_module_imports():
    from ryko_lm.eval import evaluate  # noqa: F401
    from ryko_lm.training import export  # noqa: F401
