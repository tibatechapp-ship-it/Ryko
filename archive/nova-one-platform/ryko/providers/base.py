"""عقد مزود القطعة — الواجهة الوحيدة التي تعرفها الحلقة (ADR-0004).

كل قطعة توليد (منفذ/مراجع/مخطط…) هي Provider بخصائص إلزامية:
- `name`   معرف القطعة كما في السجل
- `family` عائلة الاستقلال — الحلقة ترفض منفذاً ومراجعاً بنفس العائلة
- `generate(system, prompt)` نص حتمي الحتمية الممكنة وقابل للأدلة

لا مفاتيح ولا نقاط نهاية في الملفات — تُقرأ من البيئة عند البناء فقط.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..audit import canonical


class ProviderError(RuntimeError):
    pass


class Provider:
    name: str = "base"
    family: str = "base"
    role: str = "generic"

    def generate(self, system: str, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        raise NotImplementedError

    def fingerprint(self) -> Dict[str, Any]:
        """بصمة تُختم في الأدلة — تكفي لتتبع القطعة دون تسريب إعداداتها."""
        return {"name": self.name, "family": self.family, "role": self.role}


def build_provider(piece: Dict[str, Any]) -> Provider:
    """يبني المزود من قطعة سجل — السجل هو مصدر الحقيقة الوحيد للتهيئة.

    المطلوب في القطعة: config.kind. الأنواع المدعومة:
      - mock:             مزود حتمي offline (انظر mock.py)
      - openai_compat:    نقطة نهاية OpenAI-Compatible من البيئة
    """
    from .factory import build_from_config  # استيراد متأخر يمنع الدورانية

    return build_from_config(piece)


def canonical_fingerprint(providers: list) -> str:
    return canonical([p.fingerprint() for p in providers])


__all__ = ["Provider", "ProviderError", "build_provider", "canonical_fingerprint", "Optional"]
