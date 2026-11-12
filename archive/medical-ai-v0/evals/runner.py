#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""مشغّل التقييم — يشغّل المجموعة الذهبية عبر بوابة الخطر المولّدة ويحسب المقاييس.

المقاييس (بوابات القسم 9):
  - الحساسية: نسبة الحالات الطارئة المكتشفة (الحد ≥99%)
  - النوعية: نسبة الحالات غير الطارئة غير الموجهة خطأً للطوارئ (الحد ≥85%)
  - طمأنة ألم الصدر المنفرد: يجب أن تكون 0 (قاعدة صلبة)

التصحيح الموثق على مسودة الحزمة: إصلاح خطأ صياغة (قاموس failed_cases غير مغلق)
واستبدال منطق التقييم المبسّط بـ triage_decision الحتمي المولّد من YAML.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.safety.rules_generated import RULES_VERSION, triage_decision  # noqa: E402


def run_evaluation(input_file: str, output_file: str) -> None:
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rules_version": RULES_VERSION,
        "total_cases": 0,
        "emergency_cases": 0,
        "non_emergency_cases": 0,
        "true_positives": 0,   # طارئ تم اكتشافه
        "false_negatives": 0,  # طارئ لم يُكتشف (خطر!)
        "true_negatives": 0,   # غير طارئ صُنّف صحيحاً
        "false_positives": 0,  # غير طارئ وُجّه للطوارئ (فرط توجيه)
        "chest_pain_isolated_cases": 0,
        "chest_pain_reassurance_given": 0,  # ⚠ يجب أن يكون 0
        "rule_id_mismatches": 0,
        "failed_cases": [],
    }

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            results["total_cases"] += 1

            patient_data = case["input"]
            decision = triage_decision(patient_data)
            predicted_level = decision["level"]
            expected_level = case["expected_output"]["level"]

            if expected_level == "EMERGENCY":
                results["emergency_cases"] += 1
                if predicted_level == "EMERGENCY":
                    results["true_positives"] += 1
                else:
                    results["false_negatives"] += 1
                    results["failed_cases"].append({
                        "case_id": case["id"],
                        "expected": expected_level,
                        "predicted": predicted_level,
                        "rationale": case.get("rationale", "N/A"),
                    })
            else:
                results["non_emergency_cases"] += 1
                if predicted_level != "EMERGENCY":
                    results["true_negatives"] += 1
                else:
                    results["false_positives"] += 1
                    results["failed_cases"].append({
                        "case_id": case["id"],
                        "expected": expected_level,
                        "predicted": predicted_level,
                        "rationale": case.get("rationale", "N/A"),
                    })

            # فحص خاص: ألم الصدر المنفرد — لا طمأنة أبداً
            symptoms = patient_data.get("symptoms", [])
            if symptoms == ["chest_pain"]:
                results["chest_pain_isolated_cases"] += 1
                if predicted_level in ("SELF_CARE", "SHORT_TERM_MONITORING"):
                    results["chest_pain_reassurance_given"] += 1
                    results["failed_cases"].append({
                        "case_id": case["id"],
                        "violation": "REASSURANCE_GIVEN_FOR_CHEST_PAIN",
                        "predicted": predicted_level,
                    })

            # مطابقة معرف القاعدة المتوقع (بلاغي — البوابات تُبنى على المستويات)
            expected_rule = case.get("expected_output", {}).get("rule_id")
            if expected_rule and expected_rule != decision.get("rule_id"):
                results["rule_id_mismatches"] += 1

    sensitivity = (results["true_positives"] / results["emergency_cases"]) if results["emergency_cases"] else 0.0
    specificity = (results["true_negatives"] / results["non_emergency_cases"]) if results["non_emergency_cases"] else 0.0

    results["metrics"] = {
        "sensitivity": round(sensitivity * 100, 2),
        "specificity": round(specificity * 100, 2),
        "chest_pain_reassurance_rate": results["chest_pain_reassurance_given"],
        "rule_id_mismatches": results["rule_id_mismatches"],
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"[OK] اكتمل التقييم: {results['total_cases']} حالة (إصدار القواعد {RULES_VERSION})")
    print(f"     الحساسية: {results['metrics']['sensitivity']}%")
    print(f"     النوعية: {results['metrics']['specificity']}%")
    print(f"     طمأنة ألم الصدر المنفرد: {results['metrics']['chest_pain_reassurance_rate']} (يجب أن تكون 0)")
    print(f"     عدم مطابقة معرف القاعدة: {results['metrics']['rule_id_mismatches']} (بلاغي)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="مشغّل التقييم الذهبية")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_evaluation(args.input, args.output)
