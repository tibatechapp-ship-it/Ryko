"""حلقة التحقق MVP — قلب المرحلة 1 (PLAN.md): أهم مرحلة.

    قطعة (أ) تكتب → اختبارات → فشل؟ إصلاح حتى max_attempts → نجاح؟
    قطعة (ب) مستقلة تراجع → قبول/رفض بأدلة

ضمانات هيكلية (لا مجرد اتفاقيات):
- الاستقلالية: منفذ ومراجع بنفس `family` ⇒ رفض قبل أي توليد.
- السندبكس: كل كتابة/تنفيذ في نسخة مؤقتة من fixture — الأصل لا يُلمس.
- الأدلة: كل محاولة (بصمة المدخل، بصمة المخرج، نتيجة الاختبار) وكل قرار
  تُلحق بـ Evidence Log بسلسلة تجزئة، وكل مهمة تُلحق بسجل التدقيق.
- لا أرقام تُخمَّن: القرار مبني على رموز خروج الاختبارات وصيغة VERDICT الموحدة.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .audit import AuditLog
from .evidence import EvidenceLog, sha256_text
from .providers.base import Provider, ProviderError
from .providers.factory import build_pair
from .registry import Registry
from .tasks import TaskSpec

VERDICT_ACCEPTED = "ACCEPTED"
VERDICT_REJECTED = "REJECTED"
VERDICT_FAILED_TESTS = "FAILED_TESTS"
VERDICTS = (VERDICT_ACCEPTED, VERDICT_REJECTED, VERDICT_FAILED_TESTS)

REVIEW_FORMAT = (
    "راجع المخرج التالي بمعايير corpus/style. اجب بهذه الصيغة حصراً:\n"
    "VERDICT: APPROVE\n"
    "أو\n"
    "VERDICT: REJECT\n"
    "REASON: <سبب قصير>\n"
)
VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVE|REJECT)", re.IGNORECASE)
REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


@dataclass
class Attempt:
    index: int
    output_sha256: str
    test_exit: int
    test_output_tail: str
    ok: bool


@dataclass
class TaskOutcome:
    task_id: str
    verdict: str
    attempts: List[Attempt] = field(default_factory=list)
    review_verdict: Optional[str] = None
    review_reason: Optional[str] = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_ACCEPTED


class Sandbox:
    """نسخة عمل معزولة من fixture — كتابة وتنفيذ دون لمس الأصل."""

    def __init__(self, fixture_dir: str, target_path: str, timeout_sec: int):
        self._tmp = tempfile.mkdtemp(prefix="ryko-sandbox-")
        self.fixture_dir = fixture_dir
        self.target_path = target_path
        self.timeout_sec = timeout_sec
        shutil.copytree(fixture_dir, self._tmp, dirs_exist_ok=True)

    def write(self, rel_path: str, content: str) -> None:
        full = os.path.join(self._tmp, rel_path)
        if not os.path.abspath(full).startswith(os.path.abspath(self._tmp) + os.sep):
            raise ValueError(f"مسار هدف خارج السندبكس: {rel_path}")
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    def run(self, command: str) -> Tuple[int, str]:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self._tmp,
                capture_output=True, text=True, timeout=self.timeout_sec,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, output[-4000:]
        except subprocess.TimeoutExpired:
            return 124, f"[TIMEOUT] تجاوز الأمر المهلة {self.timeout_sec}s"

    def read(self, rel_path: str) -> str:
        with open(os.path.join(self._tmp, rel_path), "r", encoding="utf-8") as f:
            return f.read()

    def cleanup(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)


def _implementer_prompt(task: TaskSpec, attempt: int, feedback: Optional[str]) -> str:
    parts = [
        "أنت قطعة منفذة في منظومة Ryko. أعد الملف المطلوب كنص كامل فقط، بلا شرح خارجي.",
        f"TASK_ID: {task.id}",
        f"ATTEMPT: {attempt}",
        f"TARGET: {task.target_path}",
        f"TITLE: {task.title}",
        "INSTRUCTION:",
        task.instruction.strip(),
    ]
    if feedback:
        parts.append("FEEDBACK من المحاولة السابقة (أصلح هذه الأعطال تحديداً):")
        parts.append(feedback.strip())
    return "\n".join(parts)


def _reviewer_prompt(task: TaskSpec, content: str, test_tail: str) -> str:
    return "\n".join([
        REVIEW_FORMAT,
        f"TASK_ID: {task.id}",
        f"TARGET: {task.target_path}",
        "TEST_TAIL:",
        test_tail.strip()[-1500:],
        "FILE:",
        content,
    ])


def _parse_review(text: str) -> Tuple[str, str]:
    m = VERDICT_RE.search(text or "")
    verdict = (m.group(1).upper() if m else "REJECT")
    reason_m = REASON_RE.search(text or "")
    reason = reason_m.group(1).strip() if reason_m else "بلا سبب موثق — الصيغة الموحدة إلزامية"
    return verdict, reason


def run_task(
    task: TaskSpec,
    registry: Registry,
    evidence: EvidenceLog,
    audit: AuditLog,
    actor: str = "runner",
    fixture_root: str = ".",
    sandbox_dir_override: Optional[str] = None,
) -> TaskOutcome:
    """ينفذ مهمة واحدة كاملة بالحلقة والتدقيق. يعيد TaskOutcome بأدلة مختومة."""
    implementer, reviewer, impl_piece, rev_piece = build_pair(
        registry, task.implementer_piece, task.reviewer_piece
    )
    fixture_dir = task.fixture_dir
    if not os.path.isdir(fixture_dir):
        fixture_dir = os.path.join(fixture_root, task.fixture_dir)
    if not os.path.isdir(fixture_dir):
        outcome = TaskOutcome(task.id, VERDICT_FAILED_TESTS, detail=f"fixture مفقود: {task.fixture_dir}")
        evidence.record("task_finished", task.id, {"verdict": outcome.verdict, "detail": outcome.detail})
        audit.append(action="task.finished", actor=actor, resource=task.id,
                     payload={"verdict": outcome.verdict})
        return outcome

    audit.append(action="task.started", actor=actor, resource=task.id, payload={
        "implementer": impl_piece["id"], "reviewer": rev_piece["id"],
        "max_attempts": task.max_attempts,
    })
    evidence.record("task_started", task.id, {
        "implementer": impl_piece["id"], "implementer_family": implementer.family,
        "reviewer": rev_piece["id"], "reviewer_family": reviewer.family,
        "target": task.target_path,
    })

    sandbox = Sandbox(fixture_dir, task.target_path, task.timeout_sec)
    outcome = TaskOutcome(task_id=task.id, verdict=VERDICT_FAILED_TESTS)
    try:
        feedback: Optional[str] = None
        for attempt_no in range(1, task.max_attempts + 1):
            prompt = _implementer_prompt(task, attempt_no, feedback)
            content = implementer.generate(
                system="أنت قطعة منفذة حتمية تعمل داخل منظومة Ryko.",
                prompt=prompt,
            )
            sandbox.write(task.target_path, content)
            exit_code, tail = sandbox.run(task.test_command)
            attempt = Attempt(
                index=attempt_no, output_sha256=sha256_text(content),
                test_exit=exit_code, test_output_tail=tail, ok=(exit_code == 0),
            )
            outcome.attempts.append(attempt)
            evidence.record("attempt", task.id, {
                "attempt": attempt_no,
                "output_sha256": attempt.output_sha256,
                "test_command": task.test_command,
                "test_exit": exit_code,
                "prompt_sha256": sha256_text(prompt),
            })
            if attempt.ok:
                break
            feedback = tail

        passed = outcome.attempts and outcome.attempts[-1].ok
        if not passed:
            outcome.verdict = VERDICT_FAILED_TESTS
            outcome.detail = f"فشلت الاختبارات بعد {len(outcome.attempts)} محاولة"
            evidence.record("tests_failed", task.id, {
                "attempts": len(outcome.attempts),
                "last_tail": outcome.attempts[-1].test_output_tail[-800:] if outcome.attempts else "",
            })
        else:
            content = sandbox.read(task.target_path)
            review_text = reviewer.generate(
                system="أنت مراجع مستقل — لا ترى سجل المنفذ، تحكم على المخرج والاختبارات فقط.",
                prompt=_reviewer_prompt(task, content, outcome.attempts[-1].test_output_tail),
            )
            review_verdict, review_reason = _parse_review(review_text)
            outcome.review_verdict = review_verdict
            outcome.review_reason = review_reason
            evidence.record("review", task.id, {
                "verdict": review_verdict,
                "reason": review_reason,
                "reviewer": rev_piece["id"],
                "reviewer_family": reviewer.family,
                "review_sha256": sha256_text(review_text),
            })
            if review_verdict == "APPROVE":
                outcome.verdict = VERDICT_ACCEPTED
                outcome.detail = f"قُبل بعد {len(outcome.attempts)} محاولة"
            else:
                outcome.verdict = VERDICT_REJECTED
                outcome.detail = f"المراجع المستقل رفض: {review_reason}"
    finally:
        sandbox.cleanup()

    evidence.record("task_finished", task.id, {
        "verdict": outcome.verdict, "detail": outcome.detail,
        "attempts": [a.index for a in outcome.attempts],
    })
    audit.append(action="task.finished", actor=actor, resource=task.id, payload={
        "verdict": outcome.verdict, "attempts": len(outcome.attempts),
    })
    return outcome


def run_tasks(
    task_specs: List[TaskSpec],
    registry: Registry,
    evidence: EvidenceLog,
    audit: AuditLog,
    actor: str = "runner",
) -> Dict[str, Any]:
    outcomes = [run_task(t, registry, evidence, audit, actor=actor) for t in task_specs]
    report = {
        "tasks": [
            {
                "task": o.task_id, "verdict": o.verdict, "attempts": len(o.attempts),
                "review": o.review_verdict, "detail": o.detail,
            }
            for o in outcomes
        ],
        "counts": {
            v: sum(1 for o in outcomes if o.verdict == v) for v in VERDICTS
        },
        "total": len(outcomes),
    }
    audit.append(action="batch.finished", actor=actor, resource="tasks",
                 payload={"total": report["total"], "counts": report["counts"]})
    return report


def _main(argv: List[str]) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="حلقة التحقق Ryko")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("--tasks", nargs="+", required=True, help="ملفات/مجلدات مهام")
    p_run.add_argument("--registry", default="registry/registry.yaml")
    p_run.add_argument("--evidence", default="var/evidence/run.jsonl")
    p_run.add_argument("--audit", default="var/audit.jsonl")
    p_run.add_argument("--actor", default="cli")
    p_run.add_argument("--strict", action="store_true", help="فشل إن لم تُقبل كل المهام")
    args = parser.parse_args(argv)

    registry = Registry(args.registry, audit=AuditLog(args.audit))
    evidence = EvidenceLog(args.evidence)
    audit = AuditLog(args.audit)
    specs = __import__("ryko.tasks", fromlist=["load_tasks"]).load_tasks(args.tasks)
    report = run_tasks(specs, registry, evidence, audit, actor=args.actor)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    chain_ok, bad = evidence.verify()
    if not chain_ok:
        print(f"[FAIL] كسر في سلسلة الأدلة عند {bad}")
        return 1
    if args.strict:
        if report["counts"][VERDICT_ACCEPTED] != report["total"]:
            print("[BLOCKED] بوابة صارمة: ليست كل المهام مقبولة")
            return 1
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
