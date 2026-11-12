# -*- coding: utf-8 -*-
"""اختبارات بوابة الخطر المولّدة (W1 — السلامة أولاً).

القاعدة الصلبة: ألم صدر ⇒ URGENT كحد أدنى — لا طمأنة أبداً.
كل القيم هنا من قواعد YAML المولّدة حصراً — بلا أي رقم من LLM.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.safety.rules_generated import (  # noqa: E402
    DISCLAIMER,
    RULES_VERSION,
    evaluate_rules,
    triage_decision,
)


def test_rf001_chest_pain_with_companion_is_emergency():
    d = triage_decision({"symptoms": ["chest_pain", "dyspnea"], "duration_min": 20, "age_band": "45-60"})
    assert d["level"] == "EMERGENCY"
    assert d["rule_id"] == "RF-001"
    assert d["action"] == "call_ambulance"
    assert d["emergency_number"] == "997"


def test_rf001_duration_alone_is_emergency():
    d = triage_decision({"symptoms": ["chest_pain"], "duration_min": 20})
    assert d["level"] == "EMERGENCY"


def test_rf010_isolated_chest_pain_never_reassured():
    d = triage_decision({"symptoms": ["chest_pain"], "duration_min": 5, "age_band": "20-30"})
    assert d["level"] == "URGENT"
    assert d["rule_id"] == "RF-010"
    assert d["action"] == "see_doctor_today"


def test_isolated_chest_pain_immune_to_self_care_rules():
    # ألم صدر وحده لا يستوفي SC-001 (subset_of) مهما كانت الأعراض الخفيفة المصاحبة ضمن القاعدة
    triggered_ids = {r["id"] for r in evaluate_rules({"symptoms": ["chest_pain"], "duration_min": 1})}
    assert "SC-001" not in triggered_ids
    assert triggered_ids == {"RF-010"}


def test_cv_k_01_critical_potassium_from_confirmed_labs_only():
    d = triage_decision({
        "symptoms": ["weakness"],
        "duration_min": 60,
        "confirmed_critical_labs": [{"loinc": "2823-3", "value": 6.8, "unit": "mmol/L"}],
    })
    assert d["level"] == "EMERGENCY"
    assert d["rule_id"] == "CV-K-01"
    assert d["action"] == "contact_doctor_now"


def test_cv_k_01_below_threshold_does_not_trigger():
    triggered_ids = {r["id"] for r in evaluate_rules({
        "symptoms": ["weakness"],
        "confirmed_critical_labs": [{"loinc": "2823-3", "value": 6.4, "unit": "mmol/L"}],
    })}
    assert "CV-K-01" not in triggered_ids


def test_cv_k_01_wrong_loinc_does_not_trigger():
    triggered_ids = {r["id"] for r in evaluate_rules({
        "symptoms": [],
        "confirmed_critical_labs": [{"loinc": "718-7", "value": 9.0, "unit": "g/dL"}],
    })}
    assert "CV-K-01" not in triggered_ids


def test_rf005_severe_dyspnea_with_cyanosis():
    d = triage_decision({"symptoms": ["severe_dyspnea", "cyanosis"], "duration_min": 10, "age_band": "60+"})
    assert d["level"] == "EMERGENCY"
    assert d["rule_id"] == "RF-005"


def test_sc001_mild_symptom_self_care():
    d = triage_decision({"symptoms": ["mild_headache"], "duration_min": 30, "confirmed_critical_labs": []})
    assert d["level"] == "SELF_CARE"
    assert d["rule_id"] == "SC-001"


def test_mon002_fever_cough_short_term_monitoring():
    d = triage_decision({"symptoms": ["low_grade_fever", "cough"], "duration_min": 2880})
    assert d["level"] == "SHORT_TERM_MONITORING"
    assert d["rule_id"] == "MON-002"


def test_empty_input_defaults_to_safe_self_care():
    d = triage_decision({})
    assert d["level"] == "SELF_CARE"
    assert d["rule_id"] == "SC-000"
    assert d["action"] == "monitor"


def test_emergency_beats_urgent_when_both_trigger():
    # ألم صدر + عرض مرافق يستوفي RF-001 وRF-010 معاً — الطوارئ تتقدم
    d = triage_decision({"symptoms": ["chest_pain", "diaphoresis"], "duration_min": 40})
    assert d["level"] == "EMERGENCY"
    assert d["rule_id"] == "RF-001"


def test_disclaimer_matches_contract_const():
    d = triage_decision({"symptoms": ["chest_pain"]})
    assert d["disclaimer"] == DISCLAIMER
    assert d["disclaimer"] == "هذه الأداة لا تشخص ولا تعالج. في حالة الطوارئ، اتصل فوراً بخدمات الطوارئ المحلية."


def test_decision_keys_match_triage_contract():
    d = triage_decision({"symptoms": ["mild_fatigue"]})
    assert set(d.keys()) == {
        "level", "rule_id", "rules_version", "action",
        "emergency_number", "message_key", "disclaimer",
    }
    assert d["rules_version"] == RULES_VERSION


@pytest.mark.parametrize("bad_input", [
    {"symptoms": "chest_pain"},               # أعراض ليست قائمة
    {"symptoms": None},
    {"duration_min": "not-a-number"},
    {"confirmed_critical_labs": "bad"},
])
def test_malformed_input_fails_safe(bad_input):
    # مدخل مشوه لا يُطلق أي قاعدة خطر — فشل آمن نحو SELF_CARE بلا استثناء
    d = triage_decision(bad_input)
    assert d["level"] in ("SELF_CARE", "URGENT")
