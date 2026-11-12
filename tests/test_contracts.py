# -*- coding: utf-8 -*-
"""اختبارات العقود — كل قرار فرز من المولّد يجب أن يطابق contracts/v1/triage.schema.json،
وعينة EngineResult تطابق engine_result.schema.json (عقود أولاً — لا كود محرك قبل العقد)."""
import glob
import json
import os
import sys

import jsonschema
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services.safety.rules_generated import triage_decision  # noqa: E402

CONTRACTS = os.path.join(ROOT, "contracts", "v1")


def _load_schema(name):
    with open(os.path.join(CONTRACTS, name), encoding="utf-8") as f:
        return json.load(f)


TRIAGE_SCHEMA = _load_schema("triage.schema.json")
ENGINE_SCHEMA = _load_schema("engine_result.schema.json")
GOLDEN = os.path.join(ROOT, "safety", "golden", "triage_500.jsonl")


@pytest.mark.parametrize("schema_file", sorted(glob.glob(os.path.join(CONTRACTS, "*.schema.json"))))
def test_all_contracts_are_valid_json_schema_draft2020(schema_file):
    # صلاحية المخطط نفسه — العقد قبل الكود
    schema = json.load(open(schema_file, encoding="utf-8"))
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)


def test_golden_exists_with_500_cases():
    assert os.path.exists(GOLDEN), "شغّل: make golden"
    with open(GOLDEN, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 500


def _golden_cases():
    with open(GOLDEN, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@pytest.mark.parametrize("case", _golden_cases(), ids=lambda c: c["id"])
def test_every_golden_decision_validates_triage_contract(case):
    d = triage_decision(case["input"])
    jsonschema.validate(instance=d, schema=TRIAGE_SCHEMA)


def test_sample_engine_result_validates_engine_contract():
    sample = {
        "engine": "lab",
        "engine_version": "0.1.0",
        "rules_version": None,
        "status": "ok",
        "data": {
            "extractions": [{
                "original_name": "الهيموغلوبين",
                "loinc_code": "718-7",
                "value_num": 13.5,
                "unit_ucum": "g/dL",
                "confirmation_status": "extracted",
            }]
        },
        "evidence": [{
            "source_document_id": "3f2504e0-4f89-11d3-9a0c-0305e82c3301",
            "page": 2,
            "bbox": [120.0, 340.0, 480.0, 380.0],
            "quote": "Hemoglobin 13.5 g/dL",
        }],
        "confidence": {"method": "ocr_field_score", "value": 0.97},
        "abstain_reason": None,
        "limitations": ["قيم غير مؤكدة مستثناة"],
        "safety_flags": [],
    }
    jsonschema.validate(instance=sample, schema=ENGINE_SCHEMA)


def test_engine_result_ok_without_evidence_is_invalid():
    # قاعدة العقد: لا يجوز status:ok مع evidence فارغة في المحركات السريرية
    bad = {
        "engine": "lab",
        "engine_version": "0.1.0",
        "status": "ok",
        "data": {},
        "evidence": [],
        "confidence": {"method": "rule", "value": 1.0},
        "limitations": [],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=ENGINE_SCHEMA)


def test_pre_visit_report_never_carries_unconfirmed_values():
    schema = _load_schema("pre_visit_report.schema.json")
    report = {
        "patient_ref": "3f2504e0-4f89-11d3-9a0c-0305e82c3302",
        "generated_at": "2026-11-12T09:00:00Z",
        "locale": "ar-SA",
        "sections": [{"kind": "labs", "items": [{"loinc_code": "718-7", "value_num": 13.5}]}],
        "unconfirmed_included": False,
        "disclaimer": "هذا التقرير ليس تشخيصاً ويُقرأ للطبيب فقط.",
    }
    jsonschema.validate(instance=report, schema=schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={**report, "unconfirmed_included": True}, schema=schema)
