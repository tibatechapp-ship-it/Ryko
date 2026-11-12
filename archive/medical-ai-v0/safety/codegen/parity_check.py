#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""فحص التكافؤ Python <-> Dart (ADR-0002).

يتحقق من أن الحمولة الموحدة للقواعد (الشرط/الإجراء/التوقيع) مطابقة بايت-ببايت
بين الملف المولّد للخادم (Python) والملف المولّد للجهاز (Dart)، وأن إصدار
القواعد واحدًا، وأن المعرفات متطابقة. أي انحراف = فشل بناء (بوابة القسم 9).

التكافؤ السلوكي الكامل (تشغيل 1000 حالة عشوائية على Dart VM) يُستكمل عند توفر
Dart SDK في CI — راجع ADR-0002 العواقب.
"""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from services.safety import rules_generated as py_rules  # noqa: E402

DART_PATH = os.path.join(ROOT, "mobile", "medai_app", "lib", "safety", "rules_generated.dart")
REPORT_PATH = os.path.join(ROOT, "evals", "parity_report.json")


def canonical(x) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    with open(DART_PATH, encoding="utf-8") as f:
        dart_src = f.read()

    m = re.search(r"kRulesJson = r'''(.*?)'''", dart_src, re.S)
    if not m:
        print("[FAIL] لم يُعثر على kRulesJson في ملف Dart المولّد")
        sys.exit(1)
    dart_rules = json.loads(m.group(1))

    py_payload = canonical(py_rules.SAFETY_RULES)
    dart_payload = canonical(dart_rules)

    py_ids = sorted(r["id"] for r in py_rules.SAFETY_RULES)
    dart_ids = sorted(r["id"] for r in dart_rules)
    versions_match = py_rules.RULES_VERSION in dart_src.replace("kRulesVersion = '", "kRulesVersion = '", 1)

    byte_identical = py_payload == dart_payload

    report = {
        "rules_count_py": len(py_rules.SAFETY_RULES),
        "rules_count_dart": len(dart_rules),
        "ids_match": py_ids == dart_ids,
        "rules_version_py": py_rules.RULES_VERSION,
        "payload_sha256": hashlib.sha256(py_payload.encode("utf-8")).hexdigest(),
        "generated_py_sha256": sha256_file(os.path.join(ROOT, "services", "safety", "rules_generated.py")),
        "generated_dart_sha256": sha256_file(DART_PATH),
        "byte_identical": byte_identical,
        "status": "pass" if (byte_identical and py_ids == dart_ids) else "fail",
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("[OK] فحص التكافؤ Python<->Dart" if byte_identical else "[FAIL] انحراف في الحمولة الموحدة")
    print(f"     القواعد: Python={report['rules_count_py']} | Dart={report['rules_count_dart']}")
    print(f"     sha256 الحمولة: {report['payload_sha256'][:16]}...")
    print(f"     Python : {report['generated_py_sha256'][:16]}...")
    print(f"     Dart   : {report['generated_dart_sha256'][:16]}...")
    if not byte_identical or py_ids != dart_ids:
        sys.exit(1)


if __name__ == "__main__":
    main()
