"""سجل القطع (Piece Registry) — قلب حوكمة NOVA-ONE (docs/ARCHITECTURE.md §5).

قواعد غير قابلة للتفاوض:
- لا قطعة تدخل الإنتاج (`approved`) بلا: عقد واجهة، ترخيص، تقييم Holdout، مراجعة مستقلة.
- `compositional_delta < +0.10` → رفض تلقائي للاعتماد حتى لو كانت القطعة ممتازة منفردة.
  (استثناء موثق: القطع الأساسية `baseline: true` — بنية تحتية بلا تركيب سابق تقاس عليه.)
- كل تعديل = كتابة ذرية + سجل تدقيق. لا حذف قطعة من السجل — فقط `suspended/rejected`.
"""
from __future__ import annotations

import copy
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import yaml

from .audit import AuditLog

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]*[a-z0-9]$")

STATUS_DRAFT = "draft"
STATUS_CANDIDATE = "candidate"
STATUS_TESTED = "tested"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_SUSPENDED = "suspended"
STATUSES = (STATUS_DRAFT, STATUS_CANDIDATE, STATUS_TESTED, STATUS_APPROVED, STATUS_REJECTED, STATUS_SUSPENDED)

TYPES = ("base_weights", "merged_model", "moe", "lora_adapter", "tool", "corpus", "provider", "reviewer", "dataset", "eval_harness")

REQUIRED_FIELDS = (
    "id", "type", "version", "source", "license",
    "interface_contract", "holdout_scores", "cost", "status", "changelog",
)
OPTIONAL_FIELDS = ("description", "owner", "role", "family", "baseline", "independent_review", "tags", "config")

MIN_COMPOSITIONAL_DELTA = 0.10

# انتقالات حالة مقيّدة — الحوكمة لا تثق بالأفعال الحرة
ALLOWED_TRANSITIONS: Dict[str, set] = {
    STATUS_DRAFT: {STATUS_CANDIDATE, STATUS_REJECTED},
    STATUS_CANDIDATE: {STATUS_TESTED, STATUS_REJECTED},
    STATUS_TESTED: {STATUS_APPROVED, STATUS_REJECTED, STATUS_SUSPENDED},
    STATUS_APPROVED: {STATUS_SUSPENDED, STATUS_REJECTED},
    STATUS_REJECTED: {STATUS_DRAFT},
    STATUS_SUSPENDED: {STATUS_TESTED, STATUS_REJECTED},
}


class RegistryError(ValueError):
    """خطأ حوكمة — الرسالة تصف القاعدة المكسورة."""


def _validate_field_types(piece: Dict[str, Any], errors: List[str]) -> None:
    ic = piece.get("interface_contract")
    if ic is not None:
        if not isinstance(ic, dict):
            errors.append("interface_contract يجب أن يكون خريطة")
        else:
            for key in ("inputs", "outputs"):
                if not isinstance(ic.get(key), list) or not ic.get(key):
                    errors.append(f"interface_contract.{key} قائمة غير فارغة إلزامية")
            if "constraints" in ic and not isinstance(ic.get("constraints"), list):
                errors.append("interface_contract.constraints قائمة")
    hs = piece.get("holdout_scores")
    if hs is not None:
        if not isinstance(hs, dict):
            errors.append("holdout_scores يجب أن يكون خريطة أو null")
        else:
            for key in ("standalone", "compositional_delta"):
                if key in hs and not isinstance(hs[key], (int, float)):
                    errors.append(f"holdout_scores.{key} رقم")
            standalone = hs.get("standalone")
            if isinstance(standalone, (int, float)) and not 0.0 <= standalone <= 1.0:
                errors.append("holdout_scores.standalone بين 0 و1")
    if "changelog" in piece and not isinstance(piece["changelog"], list):
        errors.append("changelog قائمة")
    if "baseline" in piece and not isinstance(piece["baseline"], bool):
        errors.append("baseline منطقي")
    review = piece.get("independent_review")
    if review is not None and review not in ("pending", "approved", "rejected"):
        errors.append("independent_review واحدة من pending/approved/rejected")
    config = piece.get("config")
    if config is not None and not isinstance(config, dict):
        errors.append("config خريطة")


def validate_piece(piece: Dict[str, Any]) -> List[str]:
    """يعيد قائمة أخطاء الحوكمة (فارغة = صالح). فحص صارم: حقل غير معروف = خطأ."""
    errors: List[str] = []
    known = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    unknown = sorted(set(piece) - known)
    if unknown:
        errors.append(f"حقول غير معروفة (المخطط صارم): {unknown}")

    for field in REQUIRED_FIELDS:
        if field not in piece:
            errors.append(f"حقل إلزامي مفقود: {field}")
    if errors:
        return errors

    if not ID_PATTERN.match(piece["id"]) or not (4 <= len(piece["id"]) <= 64):
        errors.append(f"id غير مطابق للنمط: {piece['id']!r}")
    if not SEMVER.match(str(piece["version"])):
        errors.append(f"version يجب أن يكون SemVer: {piece['version']!r}")
    if piece["type"] not in TYPES:
        errors.append(f"type غير معروف: {piece['type']!r} — المسموح {TYPES}")
    if piece["status"] not in STATUSES:
        errors.append(f"status غير معروف: {piece['status']!r} — المسموح {STATUSES}")
    if not str(piece["source"]).strip():
        errors.append("source لا يجوز أن يكون فارغاً")
    if not str(piece["license"]).strip():
        errors.append("license لا يجوز أن يكون فارغاً")
    _validate_field_types(piece, errors)

    if piece["status"] == STATUS_APPROVED:
        if piece.get("independent_review") != "approved":
            errors.append("قطعة معتمدة بلا مراجعة مستقلة موافقة — يخالف قاعدة الانضباط 1")
        if piece.get("baseline", False):
            # القطع الأساسية هي المسطرة نفسها (بنية تحتية قياس) — تُعفى من نتيجة Holdout
            # بالبذرة الأولية، لكن بقاءها معتمدة يتطلب بقاء اختباراتها خضراء في CI.
            return errors
        hs = piece.get("holdout_scores") or {}
        if hs.get("standalone") is None:
            errors.append("قطعة معتمدة بلا holdout_scores.standalone — يخالف قاعدة الانضباط 1")
        delta = hs.get("compositional_delta")
        if delta is None or delta < MIN_COMPOSITIONAL_DELTA:
            errors.append(
                f"رفض تلقائي: compositional_delta={delta} < +{MIN_COMPOSITIONAL_DELTA} "
                "(القطع الجيدة التي لا تحسّن التركيب تُرفض)"
            )
    return errors


class Registry:
    """سجل قطع YAML-مدعوم — قراءة/كتابة/تحقق/انتقال حالة + تدقيق إلزامي."""

    def __init__(self, path: str, audit: Optional[AuditLog] = None):
        self.path = path
        self.audit = audit
        self._pieces: List[Dict[str, Any]] = []
        self.reload()

    # ---------- تحميل وحفظ ----------

    def reload(self) -> None:
        if not os.path.exists(self.path):
            raise RegistryError(f"ملف السجل غير موجود: {self.path}")
        with open(self.path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if not isinstance(doc, dict) or not isinstance(doc.get("pieces"), list):
            raise RegistryError("السجل يجب أن يكون خريطة بمفتاح pieces قائمة")
        seen = set()
        for i, piece in enumerate(doc["pieces"]):
            errs = validate_piece(piece)
            if errs:
                raise RegistryError(f"قطعة #{i} غير صالحة: {'; '.join(errs)}")
            pid = piece["id"]
            if pid in seen:
                raise RegistryError(f"معرف مكرر في السجل: {pid}")
            seen.add(pid)
        self._pieces = doc["pieces"]

    def save(self, actor: str, action: str, resource: str, payload: Dict[str, Any]) -> None:
        """كتابة ذرية + سجل تدقيق — إما الاثنان أو لا شيء يُعتبر منجزاً منطقياً."""
        doc = {
            "schema": "ryko.registry/v1",
            "pieces": sorted(copy.deepcopy(self._pieces), key=lambda p: p["id"]),
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(self.path)) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if self.audit is not None:
            self.audit.append(action=action, actor=actor, resource=resource, payload=payload)

    # ---------- قراءة ----------

    def get(self, piece_id: str) -> Dict[str, Any]:
        for piece in self._pieces:
            if piece["id"] == piece_id:
                return copy.deepcopy(piece)
        raise RegistryError(f"قطعة غير مسجلة: {piece_id}")

    def has(self, piece_id: str) -> bool:
        return any(p["id"] == piece_id for p in self._pieces)

    def list(self, status: Optional[str] = None, type_: Optional[str] = None) -> List[Dict[str, Any]]:
        out = [p for p in self._pieces
               if (status is None or p["status"] == status)
               and (type_ is None or p["type"] == type_)]
        return sorted(out, key=lambda p: p["id"])

    def __len__(self) -> int:
        return len(self._pieces)

    # ---------- تعديل ----------

    def add(self, piece: Dict[str, Any], actor: str = "system") -> Dict[str, Any]:
        errs = validate_piece(piece)
        if errs:
            raise RegistryError(f"رفض إضافة قطعة غير صالحة: {'; '.join(errs)}")
        if self.has(piece["id"]):
            raise RegistryError(f"معرف موجود مسبقاً: {piece['id']} — لا استنساخ في السجل")
        self._pieces.append(copy.deepcopy(piece))
        self.save(actor=actor, action="registry.add", resource=piece["id"],
                  payload={"status": piece["status"], "version": piece["version"]})
        return self.get(piece["id"])

    def set_status(self, piece_id: str, new_status: str, actor: str = "system",
                   reason: str = "") -> Dict[str, Any]:
        if new_status not in STATUSES:
            raise RegistryError(f"حالة غير معروفة: {new_status}")
        piece = self.get(piece_id)
        current = piece["status"]
        if new_status == current:
            raise RegistryError(f"القطعة أصلاً بحالة {current}")
        if new_status not in ALLOWED_TRANSITIONS[current]:
            raise RegistryError(
                f"انتقال ممنوع: {current} → {new_status} — المسموح من {current}: {sorted(ALLOWED_TRANSITIONS[current])}"
            )
        if new_status == STATUS_APPROVED:
            errs = [e for e in validate_piece(piece) if "holdout_scores.standalone" in e or "compositional_delta" in e
                    or "مراجعة مستقلة" in e]
            if errs:
                raise RegistryError(f"شروط الاعتماد غير مستوفاة: {'; '.join(errs)}")
        for p in self._pieces:
            if p["id"] == piece_id:
                p["status"] = new_status
        self.save(actor=actor, action="registry.set_status", resource=piece_id,
                  payload={"from": current, "to": new_status, "reason": reason})
        return self.get(piece_id)


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="سجل قطع Ryko — حوكمة NOVA-ONE")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("list", "validate"):
        sub.add_parser(name)
    p_list = sub.choices["list"]
    p_list.add_argument("--registry", default="registry/registry.yaml")
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--type", dest="type_", default=None)
    sub.choices["validate"].add_argument("--registry", default="registry/registry.yaml")
    p_get = sub.add_parser("get")
    p_get.add_argument("piece_id")
    p_get.add_argument("--registry", default="registry/registry.yaml")
    p_set = sub.add_parser("set-status")
    p_set.add_argument("piece_id")
    p_set.add_argument("new_status")
    p_set.add_argument("--registry", default="registry/registry.yaml")
    p_set.add_argument("--actor", default="cli")
    p_set.add_argument("--reason", default="")
    p_set.add_argument("--audit", default="var/audit.jsonl")
    args = parser.parse_args(argv)

    audit = AuditLog(args.audit) if getattr(args, "audit", None) else None
    registry = Registry(args.registry, audit=audit)
    if args.cmd == "validate":
        print(f"[OK] السجل صالح — {len(registry)} قطعة في {args.registry}")
        for p in registry.list():
            hs = p.get("holdout_scores") or {}
            print(f"  {p['id']:<34} {p['type']:<14} {p['status']:<10} "
                  f"standalone={hs.get('standalone')} delta={hs.get('compositional_delta')}")
        return 0
    if args.cmd == "list":
        for p in registry.list(status=args.status, type_=args.type_):
            print(f"{p['id']:<34} {p['type']:<14} {p['status']:<10} v{p['version']}")
        return 0
    if args.cmd == "get":
        import json
        print(json.dumps(registry.get(args.piece_id), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "set-status":
        p = registry.set_status(args.piece_id, args.new_status, actor=args.actor, reason=args.reason)
        print(f"[OK] {p['id']}: {p['status']}")
        return 0
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
