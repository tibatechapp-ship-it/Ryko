#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""فاحص العتبات التنظيمية — يفشل البناء إذا لم تتحقق بوابات القسم 9."""
import json
import sys

THRESHOLDS = {
    "sensitivity": 99.0,      # ≥99% حساسية الطوارئ
    "specificity": 85.0,      # ≥85% نوعية الفرز
    "chest_pain_reassurance_rate": 0,  # 0 طمأنة (قاعدة صلبة)
}


def check_thresholds(results_file: str) -> None:
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    metrics = results["metrics"]
    failed = []

    if metrics["sensitivity"] < THRESHOLDS["sensitivity"]:
        failed.append(f"[FAIL] الحساسية {metrics['sensitivity']}% < الحد {THRESHOLDS['sensitivity']}%")

    if metrics["specificity"] < THRESHOLDS["specificity"]:
        failed.append(f"[FAIL] النوعية {metrics['specificity']}% < الحد {THRESHOLDS['specificity']}%")

    if metrics["chest_pain_reassurance_rate"] > THRESHOLDS["chest_pain_reassurance_rate"]:
        failed.append(
            f"[FAIL] طُمئن مرضى ألم صدر {metrics['chest_pain_reassurance_rate']} مرة (الحد المسموح: 0)"
        )

    if failed:
        print("\n".join(failed))
        print("[BLOCKED] فشلت بوابات السلامة — يُمنع البناء")
        print(f"الحالات الفاشلة: {len(results['failed_cases'])}")
        for case in results["failed_cases"][:5]:
            print(f"  - {case['case_id']}: {case.get('rationale', case.get('violation', 'N/A'))}")
        sys.exit(1)

    print("[OK] جميع بوابات السلامة مستوفاة")
    print(f"     الحساسية: {metrics['sensitivity']}% (الحد ≥{THRESHOLDS['sensitivity']}%)")
    print(f"     النوعية: {metrics['specificity']}% (الحد ≥{THRESHOLDS['specificity']}%)")
    print(f"     طمأنة ألم الصدر: {metrics['chest_pain_reassurance_rate']} (الحد =0)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("الاستخدام: python check_thresholds.py <results.json>")
        sys.exit(1)
    check_thresholds(sys.argv[1])
