# -*- coding: utf-8 -*-
"""بوابة الانتقال 0 — الأدلة الملتزمة يجب أن تعيد التشغيل والتحقق بنجاح.

الأدلة في evidence/gate0/gate0.jsonl هي ختم بوابة P0: ثلاث مهام مقبولة
بسلسلة تجزئة سليمة. هذا الاختبار يمنع تعديل الأدلة بأثر رجعي.
"""
import os

import pytest

from ryko.evidence import EvidenceLog, summarize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE0 = os.path.join(REPO_ROOT, "evidence", "gate0", "gate0.jsonl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(GATE0),
    reason="بوابة 0 لم تُختَّم بعد — شغّل: make gate0",
)


def test_gate0_evidence_exists():
    assert os.path.exists(GATE0), "شغّل: make gate0"


def test_gate0_chain_intact():
    log = EvidenceLog(GATE0)
    summary = summarize(log)
    assert summary["chain_ok"] and summary["broken_at"] is None


def test_gate0_three_tasks_accepted():
    log = EvidenceLog(GATE0)
    verdicts = log.verdicts()
    accepted = {t: v for t, v in verdicts.items() if v == "ACCEPTED"}
    assert set(accepted) >= {"t-001-fib", "t-002-slugify", "t-003-brackets"}


def test_gate0_has_attempt_and_review_events():
    log = EvidenceLog(GATE0)
    actions = {e["action"] for e in log.entries()}
    assert {"evidence.task_started", "evidence.attempt", "evidence.review",
            "evidence.task_finished"} <= actions


def test_gate0_reviewer_family_differs_from_implementer():
    log = EvidenceLog(GATE0)
    starts = [e for e in log.entries() if e["action"] == "evidence.task_started"]
    assert starts
    for e in starts:
        assert e["payload"]["implementer_family"] != e["payload"]["reviewer_family"]
