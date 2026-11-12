#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""مولّد المجموعة الذهبية للفرز — 500 حالة (W1).

التوزيع يخدم بوابات القبول (القسم 9 من الوثيقة الهندسية):
  - 100 حالة طارئة محكّمة   ⇒ حساسية ≥99%
  - 100 حالة ألم صدر منفرد ⇒ 0 طمأنة (قاعدة صلبة) + نوعية ≥85% على غير الطارئة
  - 150 حالة رعاية ذاتية + 150 حالة مراقبة قصيرة المدى

التصحيح الموثق على مسودة الحزمة: بذرة عشوائية ثابتة (SEED=42) عبر
random.Random بدل random العام — إعادة الإنتاج بايت-ببايت شرط بوابة golden-check.
"""
import argparse
import json
import os
import random

SEED = 42

MILD_SYMPTOMS = ["mild_headache", "mild_fatigue", "sneezing"]


def generate(output_path: str = "safety/golden/triage_500.jsonl") -> None:
    rng = random.Random(SEED)
    cases = []

    # ===== 1. حالات طارئة (100) — لضمان حساسية ≥99% =====
    # 1أ. ألم صدر + عرض مرافق (30) → RF-001
    for i in range(30):
        cases.append({
            "id": f"T-EMG-{i:03d}",
            "input": {
                "symptoms": ["chest_pain", rng.choice(["dyspnea", "diaphoresis", "radiating_pain"])],
                "duration_min": rng.randint(15, 120),
                "age_band": "40-60",
                "locale": "ar-SA",
            },
            "expected_output": {"level": "EMERGENCY", "rule_id": "RF-001", "action": "call_ambulance"},
            "rationale": "ألم صدر مع عرض مرافق ومدة ≥ 15 دقيقة",
        })
    # 1ب. قيمة مختبرية حرجة مؤكدة (30) → CV-K-01
    for i in range(30, 60):
        cases.append({
            "id": f"T-EMG-{i:03d}",
            "input": {
                "symptoms": ["weakness"],
                "duration_min": 60,
                "confirmed_critical_labs": [{
                    "loinc": "2823-3",
                    "value": round(rng.uniform(6.5, 8.0), 2),
                    "unit": "mmol/L",
                }],
                "age_band": "40-60",
                "locale": "ar-SA",
            },
            "expected_output": {"level": "EMERGENCY", "rule_id": "CV-K-01", "action": "contact_doctor_now"},
            "rationale": "قيمة بوتاسيوم حرجة مؤكدة (≥ 6.5 mmol/L) تتجاوز أي عرض آخر",
        })
    # 1ج. ضيق تنفس حاد مع زرقة (40) → RF-005
    for i in range(60, 100):
        cases.append({
            "id": f"T-EMG-{i:03d}",
            "input": {
                "symptoms": ["severe_dyspnea", "cyanosis"],
                "duration_min": rng.randint(5, 30),
                "age_band": "60+",
                "locale": "ar-SA",
            },
            "expected_output": {"level": "EMERGENCY", "rule_id": "RF-005", "action": "call_ambulance"},
            "rationale": "ضيق تنفس حاد مع زرقة",
        })

    # ===== 2. ألم صدر منفرد (100) — حظر مطلق للطمأنة → RF-010 =====
    for i in range(100):
        cases.append({
            "id": f"T-URG-{i:03d}",
            "input": {
                "symptoms": ["chest_pain"],
                "duration_min": rng.randint(5, 14),
                "age_band": rng.choice(["20-30", "30-45", "45-60"]),
                "locale": "ar-SA",
            },
            "expected_output": {"level": "URGENT", "rule_id": "RF-010", "action": "see_doctor_today"},
            "rationale": "ألم صدر منفرد: قاعدة عدم الطمأنة المطلقة تنطبق بغض النظر عن العمر أو المدة",
        })

    # ===== 3. رعاية ذاتية (150) → SC-001 =====
    for i in range(150):
        cases.append({
            "id": f"T-SC-{i:03d}",
            "input": {
                "symptoms": [rng.choice(MILD_SYMPTOMS)],
                "duration_min": rng.randint(30, 120),
                "confirmed_critical_labs": [],
                "locale": "ar-SA",
            },
            "expected_output": {"level": "SELF_CARE", "rule_id": "SC-001", "action": "monitor"},
            "rationale": "أعراض خفيفة بدون قيم حرجة",
        })

    # ===== 4. مراقبة قصيرة المدى (150) → MON-002 =====
    for i in range(150):
        cases.append({
            "id": f"T-MON-{i:03d}",
            "input": {
                "symptoms": ["low_grade_fever", "cough"],
                "duration_min": rng.randint(120, 2880),
                "confirmed_critical_labs": [],
                "locale": "ar-SA",
            },
            "expected_output": {"level": "SHORT_TERM_MONITORING", "rule_id": "MON-002", "action": "monitor"},
            "rationale": "أعراض متوسطة تستدعي المراقبة",
        })

    # خلط حتمي لمحاكاة الترتيب الواقعي
    rng.shuffle(cases)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    by_level = {}
    for c in cases:
        by_level[c["expected_output"]["level"]] = by_level.get(c["expected_output"]["level"], 0) + 1
    print(f"[OK] تم توليد {len(cases)} حالة في {output_path}")
    for level in ("EMERGENCY", "URGENT", "SELF_CARE", "SHORT_TERM_MONITORING"):
        print(f"     - {level}: {by_level.get(level, 0)}")
    print(f"     - البذرة: {SEED} (إعادة إنتاج بايت-ببايت)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="مولّد المجموعة الذهبية triage_500")
    parser.add_argument("--output", default="safety/golden/triage_500.jsonl")
    args = parser.parse_args()
    generate(args.output)
