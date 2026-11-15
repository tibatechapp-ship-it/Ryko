"""بوابة الأدلة (Evidence Gate) — لا قبول بلا أدلة قابلة لإعادة التشغيل.

كل محاولة وكل قرار يُختم في Evidence Log بسلسلة تجزئة (يشارك محرك
`ryko.audit`) بحيث يمكن إعادة بناء مسار أي مهمة من السجل وحده: البصمات
(sha256 للمدخلات والمخرجات) تثبت ماذا جرى، دون الاعتماد على ذاكرة أي طرف.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

from .audit import AuditLog


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EvidenceLog(AuditLog):
    """Evidence Log — نفس ضمانات سلسلة التدقيق، بدلالة قرارات الحلقة."""

    def record(self, event_type: str, task_id: str, payload: Dict[str, Any],
               actor: str = "runner") -> Dict[str, Any]:
        return self.append(
            action=f"evidence.{event_type}",
            actor=actor,
            resource=task_id,
            payload=payload,
        )

    def by_task(self, task_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries() if e.get("resource") == task_id]

    def verdicts(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for e in self.entries():
            if e["action"] == "evidence.task_finished":
                out[e["resource"]] = (e.get("payload") or {}).get("verdict", "UNKNOWN")
        return out


def summarize(log: EvidenceLog) -> Dict[str, Any]:
    """ملخص بوابة — يُستخدم في تقارير البوابات وتقارير CI."""
    counts: Dict[str, int] = {}
    for verdict in log.verdicts().values():
        counts[verdict] = counts.get(verdict, 0) + 1
    ok, bad = log.verify()
    return {
        "events": len(log.entries()),
        "chain_ok": ok,
        "broken_at": bad,
        "verdicts": counts,
    }
