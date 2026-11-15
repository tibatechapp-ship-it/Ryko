"""مزود حتمي offline — حاملة بوابة الانتقال 0 والاختبارات المكثفة (ADR-0004).

سلوك موثق بالكامل:
- منفذ: يعيد الحل الصحيح للمهمة من `SOLUTIONS`، أو حللاً معيباً حتى المحاولة N
  (`fail_until_attempt`) — يسمح باختبار حلقة الإصلاح بدقة حتمية.
- مراجع: يرفض أنماطاً محظورة ويطالب بحكم في صيغة موحدة (VERDICT/REASON).
"""
from __future__ import annotations

from typing import Dict, Optional

from .base import Provider, ProviderError

BANNED_PATTERNS = ("assert False", "print(", "TODO", "FIXME", "pass  #", "eval(", "exec(")

SOLUTIONS: Dict[str, str] = {
    "t-001-fib": '''"""حساب فيبوناتشي تكرارياً — مقدار موحد."""
def fib(n: int) -> int:
    """يعيد العنصر n في متتالية فيبوناتشي (fib(0)=0, fib(1)=1)."""
    if not isinstance(n, int):
        raise TypeError("n يجب أن يكون عدداً صحيحاً")
    if n < 0:
        raise ValueError("n لا يقبل السالب")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    "t-002-slugify": '''"""تحويل نص إلى رابط قصير — قواعد corpus/style/01-naming."""
import re

_NON_WORD = re.compile(r"[^\\w]+", re.UNICODE)

def slugify(text: str) -> str:
    """يحوّل النص إلى slug: حروف/أرقام/شرطات، حروف صغيرة، بلا شرطات طرفية."""
    lowered = (text or "").lower()
    slug = _NON_WORD.sub("-", lowered).strip("-")
    return slug
''',
    "t-003-brackets": '''"""مدقق توازن الأقواس — مكدس واحد، تعقيد خطي."""
_PAIRS = {")": "(", "]": "[", "}": "{"}

def is_balanced(text: str) -> bool:
    """True إذا كانت كل الأقواس ()[]{} متوازنة ومتداخلة صحيحاً."""
    stack = []
    for ch in text or "":
        if ch in "([{":
            stack.append(ch)
        elif ch in _PAIRS:
            if not stack or stack.pop() != _PAIRS[ch]:
                return False
    return not stack
''',
}

_BROKEN: Dict[str, str] = {
    "t-001-fib": 'def fib(n):\n    # TODO ناقص\n    if n < 0:\n        return -1\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n',
    "t-002-slugify": 'def slugify(text):\n    # TODO ناقص\n    return (text or "").strip().lower().replace(" ", "-")\n',
    "t-003-brackets": 'def is_balanced(text):\n    # TODO ناقص\n    depth = 0\n    for ch in text or "":\n        if ch in "([{":\n            depth += 1\n        elif ch in ")]}":\n            depth -= 1\n    return depth == 0\n',
}


class MockImplementer(Provider):
    name = "implementer-mock"
    family = "mock-impl-family"
    role = "implementer"

    def __init__(self, name: str = "implementer-mock", fail_until_attempt: int = 0,
                 family: Optional[str] = None):
        self.name = name
        self.family = family or self.family  # العائلة من السجل — مصدر الحوكمة
        self.fail_until_attempt = max(0, fail_until_attempt)

    def generate(self, system: str, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        task_id, attempt = self._parse(prompt)
        solution = SOLUTIONS.get(task_id)
        if solution is None:
            # مهمة غير معروفة: هيكل أدنى فاشل عمداً — الحلقة يجب أن تمتنع لا تخمّن
            return f'# TODO: لا حل معروف للمهمة {task_id}\ndef main():\n    pass  #\n'
        if attempt <= self.fail_until_attempt:
            return _BROKEN[task_id]
        return solution

    @staticmethod
    def _parse(prompt: str):
        task_id = "unknown"
        attempt = 1
        for line in prompt.splitlines():
            if line.startswith("TASK_ID:"):
                task_id = line.split(":", 1)[1].strip()
            elif line.startswith("ATTEMPT:"):
                try:
                    attempt = int(line.split(":", 1)[1].strip())
                except ValueError:
                    attempt = 1
        return task_id, attempt


class MockReviewer(Provider):
    """مراجع مستقل حتمي — يرفض الأنماط المحظورة ويطالب بصيغة حكم موحدة."""

    name = "reviewer-mock"
    family = "mock-rev-family"
    role = "reviewer"

    def __init__(self, name: str = "reviewer-mock", banned=BANNED_PATTERNS,
                 reject_all: bool = False, family: Optional[str] = None):
        self.name = name
        self.banned = tuple(banned)
        self.reject_all = reject_all
        self.family = family or self.family  # العائلة من السجل — مصدر الحوكمة

    def generate(self, system: str, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        if self.reject_all:
            return "VERDICT: REJECT\nREASON: رفض صريح للاختبار — مراجعة غير مطمئنة"
        hits = [p for p in self.banned if p in prompt]
        if hits:
            return f"VERDICT: REJECT\nREASON: أنماط محظورة في المخرج: {hits}"
        return "VERDICT: APPROVE\nREASON: لا أنماط محظورة، يوجد توثيق، الاختبارات خضراء"


def build_mock(piece: Dict, config: Dict) -> Provider:  # noqa: ANN001
    family = piece.get("family")
    role = piece.get("role", "implementer")
    if role == "reviewer":
        return MockReviewer(name=piece["id"], family=family,
                            reject_all=bool(config.get("reject_all", False)))
    if role == "implementer":
        fail_until = int(config.get("fail_until_attempt", 0))
        if config.get("mode") == "always_broken":
            fail_until = 10**9
        return MockImplementer(name=piece["id"], family=family,
                               fail_until_attempt=fail_until)
    raise ProviderError(f"دور mock غير مدعوم: {role!r}")
