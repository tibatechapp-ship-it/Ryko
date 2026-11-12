# GENERATED FILE - DO NOT EDIT MANUALLY.
# Source: safety/rules/*.yaml | Generator: safety/codegen/generator.py
# MEDICAL-AI CORE v0 - W1 (السلامة أولاً) - المستودع: Ryko-Arabic (ريكو عربي)
# ⚠ كل العتبات والقواعد هنا قيم افتراضية معلقة على التوقيع السريري (docs/00-intended-use.md §0.6).
# ADR-0002: الحمولة الموحدة RULES_JSON مطابقة بايت-ببايت مع kRulesJson في Dart (make parity).
# أي شرط غير قابل للتقييم لا يُطلق القاعدة (فشل آمن). لا أرقام مولّدة هنا — قرارات حتمية فقط.

import json

RULES_VERSION = "1.0.0"
DISCLAIMER = "هذه الأداة لا تشخص ولا تعالج. في حالة الطوارئ، اتصل فوراً بخدمات الطوارئ المحلية."

RULES_JSON = r"""[
  {
    "action": {
      "action": "contact_doctor_now",
      "default_emergency_number": "997",
      "flag": "critical_value",
      "level": "EMERGENCY",
      "message_key": "safety.emergency.critical_potassium"
    },
    "clinical_signoff": {
      "date": null,
      "license_number": "[رقم الترخيص السعودي] ⚠",
      "name": "د. [الاسم الكامل] ⚠"
    },
    "description": "بوتاسيوم ≥ 6.5 mmol/L من قيم مؤكدة بشرياً فقط — طوارئ",
    "id": "CV-K-01",
    "trigger": {
      "all_of": [
        {
          "field": "confirmed_critical_labs",
          "operator": "lab_any",
          "value": {
            "loinc": "2823-3",
            "op": "gte",
            "unit": "mmol/L",
            "value": 6.5
          }
        }
      ]
    },
    "version": "1.0.0"
  },
  {
    "action": {
      "action": "monitor",
      "default_emergency_number": "997",
      "level": "SHORT_TERM_MONITORING",
      "message_key": "safety.monitoring.fever_cough"
    },
    "clinical_signoff": {
      "date": null,
      "license_number": "[رقم الترخيص السعودي] ⚠",
      "name": "د. [الاسم الكامل] ⚠"
    },
    "description": "حمى خفيفة مع كحلة — مراقبة قصيرة المدى، بلا قيم حرجة مؤكدة",
    "id": "MON-002",
    "trigger": {
      "all_of": [
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "low_grade_fever"
        },
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "cough"
        }
      ]
    },
    "version": "1.0.0"
  },
  {
    "action": {
      "action": "call_ambulance",
      "default_emergency_number": "997",
      "level": "EMERGENCY",
      "message_key": "safety.emergency.chest_pain_complex"
    },
    "clinical_signoff": {
      "date": null,
      "license_number": "[رقم الترخيص السعودي] ⚠",
      "name": "د. [الاسم الكامل] ⚠"
    },
    "description": "ألم صدر مع أي عرض مرافق أو مدة تتجاوز 15 دقيقة",
    "id": "RF-001",
    "trigger": {
      "all_of": [
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "chest_pain"
        }
      ],
      "any_of": [
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "dyspnea"
        },
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "diaphoresis"
        },
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "radiating_pain"
        },
        {
          "field": "duration_min",
          "operator": "gte",
          "value": 15
        }
      ]
    },
    "version": "1.0.0"
  },
  {
    "action": {
      "action": "call_ambulance",
      "default_emergency_number": "997",
      "level": "EMERGENCY",
      "message_key": "safety.emergency.severe_dyspnea_cyanosis"
    },
    "clinical_signoff": {
      "date": null,
      "license_number": "[رقم الترخيص السعودي] ⚠",
      "name": "د. [الاسم الكامل] ⚠"
    },
    "description": "ضيق تنفس حاد مع زرقة — مؤشر طوارئ",
    "id": "RF-005",
    "trigger": {
      "all_of": [
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "severe_dyspnea"
        },
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "cyanosis"
        }
      ]
    },
    "version": "1.0.0"
  },
  {
    "action": {
      "action": "see_doctor_today",
      "default_emergency_number": "997",
      "level": "URGENT",
      "message_key": "safety.urgent.chest_pain_isolated"
    },
    "clinical_signoff": {
      "date": null,
      "license_number": "[رقم الترخيص السعودي] ⚠",
      "name": "د. [الاسم الكامل] ⚠"
    },
    "description": "ألم صدر منفرد: لا طمأنة أبداً — URGENT كحد أدنى بغض النظر عن العمر أو المدة",
    "id": "RF-010",
    "trigger": {
      "all_of": [
        {
          "field": "symptoms",
          "operator": "contains",
          "value": "chest_pain"
        }
      ]
    },
    "version": "1.0.0"
  },
  {
    "action": {
      "action": "monitor",
      "default_emergency_number": "997",
      "level": "SELF_CARE",
      "message_key": "safety.selfcare.mild_symptoms"
    },
    "clinical_signoff": {
      "date": null,
      "license_number": "[رقم الترخيص السعودي] ⚠",
      "name": "د. [الاسم الكامل] ⚠"
    },
    "description": "عرض خفيف معروف ضمن قائمة الأعراض غير المقلقة، بلا قيم حرجة مؤكدة",
    "id": "SC-001",
    "trigger": {
      "all_of": [
        {
          "field": "symptoms",
          "operator": "subset_of",
          "value": [
            "mild_headache",
            "mild_fatigue",
            "sneezing"
          ]
        }
      ]
    },
    "version": "1.0.0"
  }
]"""

SAFETY_RULES = json.loads(RULES_JSON)

_LEVEL_RANK = {"SELF_CARE": 1, "SHORT_TERM_MONITORING": 2, "URGENT": 3, "EMERGENCY": 4}


def _as_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _num_cmp(a, op, b):
    na, nb = _as_num(a), _as_num(b)
    if na is None or nb is None:
        return False
    if op == "gte":
        return na >= nb
    if op == "lte":
        return na <= nb
    if op == "gt":
        return na > nb
    if op == "lt":
        return na < nb
    if op == "eq":
        return na == nb
    raise ValueError("معامل رقمي غير معروف: %s" % op)


def _match_condition(patient, cond):
    field = cond["field"]
    op = cond["operator"]
    value = cond.get("value")
    if field == "symptoms":
        sym = patient.get("symptoms") or []
        if not isinstance(sym, list):
            return False
        if op == "contains":
            return value in sym
        if op == "subset_of":
            return len(sym) > 0 and set(sym).issubset(set(value))
        raise ValueError("معامل أعراض غير معروف: %s" % op)
    if field == "confirmed_critical_labs":
        labs = patient.get("confirmed_critical_labs") or []
        if op == "lab_any":
            for lab in labs:
                if str(lab.get("loinc")) == str(value["loinc"]) and _num_cmp(
                    lab.get("value"), value.get("op", "gte"), value["value"]
                ):
                    return True
            return False
        raise ValueError("معامل مختبرات غير معروف: %s" % op)
    v = patient.get(field)
    if op == "eq":
        return v == value
    if op == "gte":
        return v is not None and _num_cmp(v, "gte", value)
    if op == "lte":
        return v is not None and _num_cmp(v, "lte", value)
    raise ValueError("معامل قيمة غير معروف: %s" % op)


def _eval_group(patient, trigger):
    for cond in trigger.get("all_of", []):
        if not _match_condition(patient, cond):
            return False
    any_conds = trigger.get("any_of", [])
    if any_conds and not any(_match_condition(patient, c) for c in any_conds):
        return False
    return True


def evaluate_rules(patient_data):
    """يعيد القواعد المستوفاة (قائمة). حتمي تماماً — بلا LLM."""
    triggered = []
    for rule in SAFETY_RULES:
        try:
            if _eval_group(patient_data, rule["trigger"]):
                triggered.append(rule)
        except Exception:
            continue  # فشل آمن: شرط غير قابل للتقييم لا يُطلق القاعدة
    return triggered


def triage_decision(patient_data, emergency_number="997"):
    """القرار النهائي وفق عقد contracts/v1/triage.schema.json.

    ألم صدر منفرد ⇒ URGENT كحد أدنى (قاعدة صلبة — لا طمأنة أبداً).
    الطوارئ تتقدم على كل المستويات. بلا قاعدة مستوفاة ⇒ SELF_CARE افتراضي آمن.
    """
    triggered = evaluate_rules(patient_data)
    if not triggered:
        return {
            "level": "SELF_CARE",
            "rule_id": "SC-000",
            "rules_version": RULES_VERSION,
            "action": "monitor",
            "emergency_number": emergency_number,
            "message_key": "safety.selfcare.default",
            "disclaimer": DISCLAIMER,
        }
    top = max(triggered, key=lambda r: _LEVEL_RANK[r["action"]["level"]])
    act = top["action"]
    return {
        "level": act["level"],
        "rule_id": top["id"],
        "rules_version": RULES_VERSION,
        "action": act["action"],
        "emergency_number": str(act.get("default_emergency_number", emergency_number)),
        "message_key": act["message_key"],
        "disclaimer": DISCLAIMER,
    }
