# -*- coding: utf-8 -*-
"""اختبارات سجل القطع — الحوكمة هنا تصبح كوداً يُختبر."""
import copy

import pytest

from ryko.audit import AuditLog
from ryko.registry import (
    MIN_COMPOSITIONAL_DELTA,
    Registry,
    RegistryError,
    validate_piece,
)

BASE_PIECE = {
    "id": "provider-test-v1",
    "type": "provider",
    "role": "implementer",
    "family": "test-family",
    "version": "1.0.0",
    "source": "test://piece",
    "license": "internal",
    "interface_contract": {"inputs": ["a"], "outputs": ["b"], "constraints": []},
    "holdout_scores": {"standalone": 0.9, "compositional_delta": 0.15},
    "cost": {"inference": "0$"},
    "status": "approved",
    "independent_review": "approved",
    "changelog": ["1.0.0: اختبار"],
}


@pytest.fixture
def registry_path(tmp_path):
    return str(tmp_path / "registry.yaml")


@pytest.fixture
def audit_path(tmp_path):
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def registry(registry_path, audit_path):
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write("schema: ryko.registry/v1\npieces: []\n")
    return Registry(registry_path, audit=AuditLog(audit_path))


def _valid(**overrides):
    piece = copy.deepcopy(BASE_PIECE)
    piece.update(overrides)
    return piece


# ---------- المخطط الصارم ----------

def test_valid_piece_passes():
    assert validate_piece(_valid()) == []


@pytest.mark.parametrize("missing", ["id", "type", "version", "source", "license",
                                     "interface_contract", "holdout_scores", "cost",
                                     "status", "changelog"])
def test_missing_required_field_rejected(missing):
    piece = _valid()
    del piece[missing]
    assert any(missing in e for e in validate_piece(piece))


def test_unknown_field_rejected_strict_schema():
    errs = validate_piece(_valid(secret_field="x"))
    assert any("غير معروفة" in e for e in errs)


def test_bad_semver_rejected():
    assert any("SemVer" in e for e in validate_piece(_valid(version="1.0")))


def test_bad_type_rejected():
    assert any("type" in e for e in validate_piece(_valid(type="magic")))


def test_bad_status_rejected():
    assert any("status" in e for e in validate_piece(_valid(status="flying")))


def test_bad_id_pattern_rejected():
    assert any("id" in e for e in validate_piece(_valid(id="Bad_ID!")))


def test_empty_interface_contract_rejected():
    piece = _valid()
    piece["interface_contract"] = {"inputs": [], "outputs": ["x"]}
    assert any("inputs" in e for e in validate_piece(piece))


def test_standalone_out_of_range_rejected():
    piece = _valid(holdout_scores={"standalone": 1.5, "compositional_delta": 0.2})
    assert any("standalone" in e for e in validate_piece(piece))


# ---------- قاعدة الاعتماد الصارمة ----------

def test_approval_requires_delta_threshold():
    piece = _valid(holdout_scores={"standalone": 0.9, "compositional_delta": MIN_COMPOSITIONAL_DELTA - 0.01})
    errs = validate_piece(piece)
    assert any("رفض تلقائي" in e for e in errs)


def test_approval_without_independent_review_rejected():
    piece = _valid(independent_review="pending")
    assert any("مراجعة مستقلة" in e for e in validate_piece(piece))


def test_approval_without_standalone_rejected():
    piece = _valid(holdout_scores={"compositional_delta": 0.5})
    assert any("standalone" in e for e in validate_piece(piece))


def test_baseline_piece_exempt_from_delta():
    piece = _valid(baseline=True, holdout_scores={"standalone": 1.0, "compositional_delta": 0.0})
    assert validate_piece(piece) == []


def test_candidate_piece_needs_no_scores():
    piece = _valid(status="candidate", holdout_scores=None, independent_review="pending")
    assert validate_piece(piece) == []


# ---------- السجل: إضافة وقراءة ----------

def test_add_and_get_roundtrip(registry):
    registry.add(_valid(), actor="test")
    assert registry.get("provider-test-v1")["id"] == "provider-test-v1"
    assert registry.has("provider-test-v1") and len(registry) == 1


def test_add_invalid_raises(registry):
    with pytest.raises(RegistryError):
        registry.add(_valid(status="nope"), actor="test")


def test_duplicate_id_rejected(registry):
    registry.add(_valid(), actor="test")
    with pytest.raises(RegistryError, match="مكرر|موجود"):
        registry.add(_valid(), actor="test")


def test_get_missing_raises(registry):
    with pytest.raises(RegistryError, match="غير مسجلة"):
        registry.get("nope-v1")


def test_registry_file_must_exist(tmp_path):
    with pytest.raises(RegistryError):
        Registry(str(tmp_path / "missing.yaml"))


def test_persisted_registry_reloads(tmp_path, audit_path):
    path = str(tmp_path / "reg.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("schema: ryko.registry/v1\npieces: []\n")
    reg = Registry(path, audit=AuditLog(audit_path))
    reg.add(_valid(), actor="test")
    reg2 = Registry(path)
    assert reg2.get("provider-test-v1")["version"] == "1.0.0"


# ---------- انتقالات الحالة ----------

def test_illegal_transition_rejected(registry):
    registry.add(_valid(), actor="test")  # approved
    with pytest.raises(RegistryError, match="ممنوع"):
        registry.set_status("provider-test-v1", "candidate", actor="test", reason="x")


def test_reject_then_back_to_draft_allowed(registry):
    registry.add(_valid(), actor="test")
    registry.set_status("provider-test-v1", "rejected", actor="test", reason="سيء")
    registry.set_status("provider-test-v1", "draft", actor="test", reason="إعادة")
    assert registry.get("provider-test-v1")["status"] == "draft"


def test_set_status_unknown_rejected(registry):
    registry.add(_valid(), actor="test")
    with pytest.raises(RegistryError, match="غير معروفة"):
        registry.set_status("provider-test-v1", "martian", actor="test")


def test_set_status_same_rejected(registry):
    registry.add(_valid(), actor="test")
    with pytest.raises(RegistryError):
        registry.set_status("provider-test-v1", "approved", actor="test")


def test_suspend_approved_and_resume_via_tested(registry):
    registry.add(_valid(), actor="test")
    registry.set_status("provider-test-v1", "suspended", actor="test", reason="انخفاض مقياس")
    registry.set_status("provider-test-v1", "tested", actor="test", reason="بعد إصلاح")
    assert registry.get("provider-test-v1")["status"] == "tested"


# ---------- التدقيق والكتابة الذرية ----------

def test_mutations_are_audited(registry, audit_path):
    registry.add(_valid(), actor="auditor")
    log = AuditLog(audit_path)
    assert [e["action"] for e in log.entries()] == ["registry.add"]
    assert log.entries()[0]["actor"] == "auditor"
    registry.set_status("provider-test-v1", "rejected", actor="auditor", reason="اختبار")
    actions = [e["action"] for e in AuditLog(audit_path).entries()]
    assert actions[-1] == "registry.set_status"


def test_audit_chain_valid_after_mutations(registry, audit_path):
    registry.add(_valid(), actor="a")
    registry.set_status("provider-test-v1", "rejected", actor="a", reason="r")
    ok, bad = AuditLog(audit_path).verify()
    assert ok and bad is None


def test_get_returns_deep_copy(registry):
    registry.add(_valid(), actor="a")
    piece = registry.get("provider-test-v1")
    piece["status"] = "suspended"
    assert registry.get("provider-test-v1")["status"] == "approved"


def test_real_seed_registry_loads():
    import os
    seed = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "registry", "registry.yaml")
    reg = Registry(seed)
    assert len(reg) >= 6
    assert reg.get("implementer-mock-v1")["status"] == "approved"
    assert reg.get("provider-openai-compat-v1")["status"] == "candidate"
