# -*- coding: utf-8 -*-
"""اختبارات المزودات — العقد الواحد، حتمية الموك، وشكل طلب OpenAI-Compat."""
import json
import os
import urllib.error

import pytest

from ryko.providers.base import Provider, build_provider
from ryko.providers.factory import build_pair
from ryko.providers.mock import MockImplementer, MockReviewer
from ryko.providers.openai_compat import OpenAICompatProvider


# ---------- العقد ----------

def test_base_provider_raises_by_default():
    with pytest.raises(NotImplementedError):
        Provider().generate("s", "p")


def test_fingerprints_hide_secrets():
    p = OpenAICompatProvider("x", "fam", "http://localhost:1", "SECRET", "m")
    fp = p.fingerprint()
    assert "SECRET" not in json.dumps(fp) and fp["model"] == "m"


def test_unknown_kind_rejected():
    with pytest.raises(Exception, match="غير مدعوم"):
        build_provider({"id": "p-x-v1", "config": {"kind": "quantum"}})


# ---------- المزود الحتمي ----------

def test_mock_implementer_known_task_correct():
    out = MockImplementer().generate("s", "TASK_ID: t-001-fib\nATTEMPT: 1")
    assert "def fib" in out and "TODO" not in out


def test_mock_implementer_unknown_task_fails_safe_stub():
    out = MockImplementer().generate("s", "TASK_ID: t-999-unknown\nATTEMPT: 1")
    assert "TODO" in out  # يفشل الاختبارات لاحقاً — لا تخمين


def test_mock_implementer_injected_failure_then_repair():
    impl = MockImplementer(fail_until_attempt=2)
    broken = impl.generate("s", "TASK_ID: t-003-brackets\nATTEMPT: 1")
    broken2 = impl.generate("s", "TASK_ID: t-003-brackets\nATTEMPT: 2")
    fixed = impl.generate("s", "TASK_ID: t-003-brackets\nATTEMPT: 3")
    assert "TODO" in broken and "TODO" in broken2
    assert "TODO" not in fixed and "def is_balanced" in fixed


def test_mock_reviewer_rejects_banned_patterns():
    review = MockReviewer().generate(
        "s", "FILE:\n# TODO ناقص\ndef f():\n    pass  #\n")
    assert "VERDICT: REJECT" in review


def test_mock_reviewer_approves_clean_output():
    from ryko.providers.mock import SOLUTIONS
    review = MockReviewer().generate("s", f"FILE:\n{SOLUTIONS['t-001-fib']}")
    assert "VERDICT: APPROVE" in review


def test_mock_reviewer_reject_all_mode():
    review = MockReviewer(reject_all=True).generate("s", "FILE:\nclean")
    assert "VERDICT: REJECT" in review


def test_families_are_distinct_by_design():
    assert MockImplementer().family != MockReviewer().family


# ---------- المصنع والاستقلالية ----------

def _registry_with_pair(tmp_path):
    from ryko.audit import AuditLog
    from ryko.registry import Registry
    reg_path = str(tmp_path / "r.yaml")
    with open(reg_path, "w", encoding="utf-8") as f:
        f.write("schema: ryko.registry/v1\npieces: []\n")
    reg = Registry(reg_path, audit=AuditLog(str(tmp_path / "a.jsonl")))
    reg.add({
        "id": "impl-x-v1", "type": "provider", "role": "implementer", "family": "fam-a",
        "version": "1.0.0", "source": "mock", "license": "internal",
        "interface_contract": {"inputs": ["t"], "outputs": ["f"], "constraints": []},
        "holdout_scores": {"standalone": 1.0, "compositional_delta": 0.0},
        "cost": {}, "status": "approved", "independent_review": "approved", "baseline": True,
        "changelog": ["1.0.0: x"], "config": {"kind": "mock", "mode": "standard"},
    })
    reg.add({
        "id": "rev-x-v1", "type": "reviewer", "role": "reviewer", "family": "fam-b",
        "version": "1.0.0", "source": "mock", "license": "internal",
        "interface_contract": {"inputs": ["t"], "outputs": ["r"], "constraints": []},
        "holdout_scores": {"standalone": 1.0, "compositional_delta": 0.0},
        "cost": {}, "status": "approved", "independent_review": "approved", "baseline": True,
        "changelog": ["1.0.0: x"], "config": {"kind": "mock", "mode": "standard"},
    })
    return reg


def test_build_pair_from_registry(tmp_path):
    reg = _registry_with_pair(tmp_path)
    impl, rev, _, _ = build_pair(reg, "impl-x-v1", "rev-x-v1")
    assert impl.family != rev.family


def test_build_pair_enforces_independence(tmp_path):
    reg = _registry_with_pair(tmp_path)
    # مراجع بنفس عائلة المنفذ = خطر حرج — يُرفض قبل أي توليد
    piece = dict(reg.get("rev-x-v1"), family="fam-a")
    reg._pieces = [p for p in reg._pieces if p["id"] != "rev-x-v1"]
    reg._pieces.append(piece)
    from ryko.providers.base import ProviderError
    with pytest.raises(ProviderError, match="استقلالية"):
        build_pair(reg, "impl-x-v1", "rev-x-v1")


def test_build_pair_rejects_unapproved_piece(tmp_path):
    reg = _registry_with_pair(tmp_path)
    reg.set_status("rev-x-v1", "rejected", actor="t", reason="t")
    from ryko.registry import RegistryError
    with pytest.raises(RegistryError, match="قبل الاعتماد"):
        build_pair(reg, "impl-x-v1", "rev-x-v1")


# ---------- OpenAI-Compatible ----------

def test_from_env_missing_vars_raise(monkeypatch):
    for var in ("RYKO_LLM_BASE_URL", "RYKO_LLM_API_KEY", "RYKO_LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(Exception, match="مفقودة"):
        OpenAICompatProvider.from_env("x", "fam")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_generate_posts_contract_and_parses(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"choices": [{"message": {"content": "مرحباً من النموذج"}}]})

    monkeypatch.setattr("ryko.providers.openai_compat.urllib.request.urlopen", fake_urlopen)
    p = OpenAICompatProvider("prov", "fam", "http://localhost:8000/v1", "k", "mymodel")
    out = p.generate("نظام", "طلب", temperature=0.1, max_tokens=64)
    assert out == "مرحباً من النموذج"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["body"]["model"] == "mymodel"
    assert captured["body"]["messages"][0] == {"role": "system", "content": "نظام"}
    assert captured["body"]["temperature"] == 0.1
    assert captured["body"]["max_tokens"] == 64


def test_generate_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def boom(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.URLError("رفض الاتصال")

    monkeypatch.setattr("ryko.providers.openai_compat.urllib.request.urlopen", boom)
    p = OpenAICompatProvider("prov", "fam", "http://localhost:9", "k", "m", retries=2)
    with pytest.raises(Exception, match="محاولات"):
        p.generate("s", "p")
    assert calls["n"] == 3  # محاولة أصلية + إعادتان
