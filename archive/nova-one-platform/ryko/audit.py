"""سجل تدقيق غير قابل للتعديل — سلسلة تجزئة SHA-256 (القاعدة الذهبية 5).

كل إجراء في المنظومة (إضافة قطعة، تغيير حالة، تشغيل مهمة، قرار مراجعة)
يُلحق هنا كسجل واحد يحمل تجزئة السجل السابق. أي تعديل يدوي بأثر رجعي
يكسر السلسلة ويكشفه `verify()` فوراً — لا فشل صامت.

الصيغة: JSONL — سطر واحد لكل إجراء، حقول: seq, ts, actor, action,
resource, payload, prev_hash, hash.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

GENESIS_HASH = "0" * 64


def canonical(payload: Any) -> str:
    """تمثيل حتمي مستقر — نفس البيانات = نفس السلسلة دائماً."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_entry(prev_hash: str, entry_body: Dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical(entry_body)).encode("utf-8")).hexdigest()


class AuditLog:
    """ملف JSONL append-only بسلسلة تجزئة."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    # ---------- كتابة ----------

    def append(
        self,
        action: str,
        actor: str,
        resource: str,
        payload: Optional[Dict[str, Any]] = None,
        ts: Optional[str] = None,
    ) -> Dict[str, Any]:
        entries = self.entries()
        prev_hash = entries[-1]["hash"] if entries else GENESIS_HASH
        body = {
            "seq": len(entries) + 1,
            "ts": ts or datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "action": action,
            "resource": resource,
            "payload": payload or {},
        }
        entry = {**body, "prev_hash": prev_hash}
        entry["hash"] = _hash_entry(prev_hash, body)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        return entry

    # ---------- قراءة وتحقق ----------

    def entries(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out: List[Dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def last_hash(self) -> str:
        entries = self.entries()
        return entries[-1]["hash"] if entries else GENESIS_HASH

    def verify(self) -> Tuple[bool, Optional[int]]:
        """يعيد (سليمة؟، رقم أول سجل فاسد أو None).

        التغطية: التجزئة = sha256(prev_hash + canonical(الجسد)) — فهي تغطي
        محتوى السجل وتربطه بالسابق؛ أي تعديل بأثر رجعي يكسر السلسلة.
        """
        prev = GENESIS_HASH
        for i, entry in enumerate(self.entries()):
            body = {k: entry[k] for k in ("seq", "ts", "actor", "action", "resource", "payload")}
            if entry.get("prev_hash") != prev:
                return False, i
            expected = _hash_entry(entry["prev_hash"], body)
            if entry.get("hash") != expected:
                return False, i
            if entry["seq"] != i + 1:
                return False, i
            prev = entry["hash"]
        return True, None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.entries())


def _main(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="سجل تدقيق Ryko — سلسلة تجزئة")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_verify = sub.add_parser("verify", help="التحقق من سلامة السلسلة")
    p_verify.add_argument("path")
    p_tail = sub.add_parser("tail", help="عرض آخر سجلات")
    p_tail.add_argument("path")
    p_tail.add_argument("-n", type=int, default=5)
    args = parser.parse_args(argv)

    log = AuditLog(args.path)
    if args.cmd == "verify":
        ok, bad = log.verify()
        if ok:
            print(f"[OK] السلسلة سليمة — {len(log.entries())} سجل")
            return 0
        print(f"[FAIL] كسر في السلسلة عند السجل رقم {bad + 1} (السطر {bad + 1})")
        return 1
    for entry in log.entries()[-args.n:]:
        print(f"{entry['seq']:>4} {entry['ts'][:19]} {entry['actor']:<12} {entry['action']:<24} {entry['resource']}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main(sys.argv[1:]))
