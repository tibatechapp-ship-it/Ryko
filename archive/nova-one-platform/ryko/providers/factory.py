"""مصنع المزودات — السجل مصدر الحقيقة الوحيد للتهيئة (حوكمة، لا أفعال حرة)."""
from __future__ import annotations

from typing import Any, Dict

from ..registry import RegistryError
from .base import Provider, ProviderError
from .mock import build_mock
from .openai_compat import OpenAICompatProvider


def build_from_config(piece: Dict[str, Any]) -> Provider:
    config = piece.get("config") or {}
    kind = config.get("kind")
    if kind == "mock":
        return build_mock(piece, config)
    if kind == "openai_compat":
        return OpenAICompatProvider.from_env(
            name=piece["id"],
            family=piece.get("family") or piece["id"],
            role=piece.get("role", "generic"),
        )
    raise ProviderError(
        f"قطعة {piece['id']}: config.kind غير مدعوم: {kind!r} — المسموح: mock, openai_compat"
    )


def build_pair(registry: Registry_like, implementer_id: str, reviewer_id: str):
    """يبني ثنائي (منفذ، مراجع) من السجل ويفرض الاستقلالية هيكلياً.

    القاعدة الذهبية 1: المراجع مستقل تماماً — عائلة مختلفة إلزاماً، والفشل
    يحدث قبل أي توليد.
    """
    implementer_piece = registry.get(implementer_id)
    reviewer_piece = registry.get(reviewer_id)
    for p, expected_role in ((implementer_piece, "implementer"), (reviewer_piece, "reviewer")):
        if p["status"] != "approved":
            raise RegistryError(
                f"قطعة {p['id']} حالتها {p['status']!r} — لا تُستخدم في الحلقة قبل الاعتماد (قاعدة الانضباط 1)"
            )
        if p.get("role") != expected_role:
            raise RegistryError(f"قطعة {p['id']} دورها {p.get('role')!r} والمطلوب {expected_role!r}")
    implementer = build_from_config(implementer_piece)
    reviewer = build_from_config(reviewer_piece)
    if implementer.family == reviewer.family:
        raise ProviderError(
            f"استقلالية مكسورة: المنفذ والمراجع بنفس العائلة ({implementer.family!r}) — "
            "خطر حرج موثق في PLAN.md (المرحلة 1)"
        )
    return implementer, reviewer, implementer_piece, reviewer_piece


class Registry_like:
    """تعريف بنيوي فقط — Registry يطابقه."""
    def get(self, piece_id: str) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError
