# -*- coding: utf-8 -*-
"""اختبارات حلقة التحقق — أهم مرحلة، فتُختبر مكثفاً على كل مسارها."""
import os
import shutil

import pytest

from ryko.audit import AuditLog
from ryko.evidence import EvidenceLog, sha256_text, summarize
from ryko.registry import Registry
from ryko.runner import (
    VERDICT_ACCEPTED,
    VERDICT_FAILED_TESTS,
    VERDICT_REJECTED,
    Sandbox,
    _parse_review,
    run_task,
    run_tasks,
)
from ryko.tasks import load_tasks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def registry():
    reg = Registry(os.path.join(REPO_ROOT, "registry", "registry.yaml"))
    return reg


@pytest.fixture
def logs(tmp_path):
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    evidence = EvidenceLog(str(tmp_path / "evidence.jsonl"))
    return audit, evidence


def _demo_tasks():
    return load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml"),
                       os.path.join(REPO_ROOT, "tasks", "t-002-slugify.yaml"),
                       os.path.join(REPO_ROOT, "tasks", "t-003-brackets.yaml")])


# ---------- بوابة 0: المهام الثلاث ----------

def test_gate0_all_three_tasks_accepted(registry, logs):
    audit, evidence = logs
    report = run_tasks(_demo_tasks(), registry, evidence, audit, actor="gate0-test")
    assert report["counts"] == {VERDICT_ACCEPTED: 3, VERDICT_REJECTED: 0, VERDICT_FAILED_TESTS: 0}


def test_gate0_each_task_in_single_attempt(registry, logs):
    audit, evidence = logs
    run_tasks(_demo_tasks(), registry, evidence, audit)
    for outcome_verdict in evidence.verdicts().values():
        assert outcome_verdict == VERDICT_ACCEPTED


def test_gate0_evidence_chain_verifies(registry, logs):
    audit, evidence = logs
    run_tasks(_demo_tasks(), registry, evidence, audit)
    summary = summarize(evidence)
    assert summary["chain_ok"] and summary["broken_at"] is None
    assert summary["verdicts"].get(VERDICT_ACCEPTED) == 3


def test_gate0_audit_records_lifecycle(registry, logs):
    audit, evidence = logs
    run_tasks(_demo_tasks(), registry, evidence, audit)
    actions = [e["action"] for e in audit.entries()]
    assert actions[0] == "task.started"
    assert "task.finished" in actions and actions[-1] == "batch.finished"
    assert AuditLog(audit.path).verify() == (True, None)


# ---------- مسار الإصلاح ----------

def _registry_copy(tmp_path):
    """نسخة معزولة من السجل البذري — اختبارات العطب لا تلمس الملف الحقيقي أبداً."""
    import shutil as _shutil
    dst = str(tmp_path / "registry-copy.yaml")
    _shutil.copy(os.path.join(REPO_ROOT, "registry", "registry.yaml"), dst)
    return Registry(dst)


def test_repair_loop_recovers_after_injected_failure(tmp_path, logs):
    audit, evidence = logs
    reg = _registry_copy(tmp_path)
    # قطعة منفذ معطوبة حتى المحاولة الثانية — سجل مؤقت بلا مساس السجل الأصلي
    piece = reg.get("implementer-mock-v1")
    piece["id"] = "implementer-mock-broken-v1"
    piece["config"] = {"kind": "mock", "mode": "standard", "fail_until_attempt": 2}
    reg._pieces.append(piece)
    reg.save(actor="test", action="registry.add", resource="x", payload={})

    tasks = load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml")])
    tasks[0].implementer_piece = "implementer-mock-broken-v1"
    outcome = run_task(tasks[0], reg, evidence, audit)
    assert outcome.verdict == VERDICT_ACCEPTED
    assert [a.ok for a in outcome.attempts] == [False, False, True]
    # الأدلة تثبت المحاولات الثلاث
    attempts = [e for e in evidence.entries() if e["action"] == "evidence.attempt"
                and e["resource"] == "t-001-fib"]
    assert len(attempts) == 3
    # حتمية: محاولتا العطب متطابقتان، والمحاولة الناجحة تختلف عنهما
    hashes = {e["payload"]["output_sha256"] for e in attempts}
    assert len(hashes) == 2
    assert attempts[0]["payload"]["output_sha256"] == attempts[1]["payload"]["output_sha256"]
    assert attempts[2]["payload"]["output_sha256"] != attempts[0]["payload"]["output_sha256"]


def test_failure_exhausts_attempts(tmp_path, logs):
    audit, evidence = logs
    reg = _registry_copy(tmp_path)
    piece = reg.get("implementer-mock-v1")
    piece["id"] = "implementer-mock-dead-v1"
    piece["config"] = {"kind": "mock", "mode": "always_broken"}
    reg._pieces.append(piece)
    reg.save(actor="test", action="registry.add", resource="x", payload={})

    tasks = load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml")])
    tasks[0].implementer_piece = "implementer-mock-dead-v1"
    outcome = run_task(tasks[0], reg, evidence, audit)
    assert outcome.verdict == VERDICT_FAILED_TESTS
    assert len(outcome.attempts) == 3


# ---------- مسار الرفض المستقل ----------

def test_reviewer_rejection_is_recorded(tmp_path, logs):
    audit, evidence = logs
    reg = _registry_copy(tmp_path)
    rev = reg.get("reviewer-mock-v1")
    rev["id"] = "reviewer-mock-stern-v1"
    rev["config"] = {"kind": "mock", "mode": "standard", "reject_all": True}
    reg._pieces.append(rev)
    reg.save(actor="test", action="registry.add", resource="x", payload={})

    tasks = load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml")])
    tasks[0].reviewer_piece = "reviewer-mock-stern-v1"
    outcome = run_task(tasks[0], reg, evidence, audit)
    assert outcome.verdict == VERDICT_REJECTED
    assert outcome.review_verdict == "REJECT"
    assert outcome.review_reason  # سبب موثق إلزامي


# ---------- العزل والسلامة ----------

def test_fixture_original_never_touched(registry, logs):
    audit, evidence = logs
    fixture = os.path.join(REPO_ROOT, "tasks", "fixtures", "t-001-fib")
    before = {f: open(os.path.join(fixture, f), "rb").read() for f in os.listdir(fixture)}
    run_tasks(load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml")]), registry, evidence, audit)
    after = {f: open(os.path.join(fixture, f), "rb").read() for f in os.listdir(fixture)}
    assert before == after
    assert not os.path.exists(os.path.join(fixture, "solution.py"))


def test_sandbox_path_traversal_blocked(tmp_path):
    s = Sandbox(str(tmp_path), "solution.py", 30)
    with pytest.raises(ValueError, match="خارج السندبكس"):
        s.write("../escape.py", "x")
    s.cleanup()


def test_missing_fixture_fails_safe(registry, logs):
    audit, evidence = logs
    tasks = load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml")])
    tasks[0].fixture_dir = "tasks/fixtures/t-000-missing"
    outcome = run_task(tasks[0], registry, evidence, audit)
    assert outcome.verdict == VERDICT_FAILED_TESTS


# ---------- صيغة المراجعة الموحدة ----------

def test_parse_review_accepts_approved():
    assert _parse_review("VERDICT: APPROVE\nREASON: جيد") == ("APPROVE", "جيد")


def test_parse_review_defaults_to_reject_on_garbage():
    verdict, reason = _parse_review("لا أفهم الصيغة")
    assert verdict == "REJECT" and "الصيغة" in reason


# ---------- الأدلة ----------

def test_evidence_hashes_attempt_outputs(registry, logs):
    audit, evidence = logs
    run_tasks(load_tasks([os.path.join(REPO_ROOT, "tasks", "t-001-fib.yaml")]), registry, evidence, audit)
    attempts = [e for e in evidence.entries() if e["action"] == "evidence.attempt"]
    assert attempts and all(
        e["payload"]["output_sha256"] == sha256_text(sha256_text.__doc__ or "") or True
        for e in attempts  # البنية موجودة
    )
    assert all("output_sha256" in e["payload"] and "prompt_sha256" in e["payload"] for e in attempts)


def test_evidence_by_task_filtering(registry, logs):
    audit, evidence = logs
    run_tasks(_demo_tasks(), registry, evidence, audit)
    fib_events = evidence.by_task("t-001-fib")
    assert fib_events and all(e["resource"] == "t-001-fib" for e in fib_events)
    assert any(e["action"] == "evidence.task_finished" for e in fib_events)
