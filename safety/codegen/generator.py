#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""مولّد كود السلامة — مصدر واحد للحقيقة (YAML) → Python + Dart.

ADR-0002: قواعد الخطر والقيم الحرجة تُكتب مرة واحدة في safety/rules/*.yaml
ويولّد هذا المولّد نسختين متكافئتين بايت-ببايت على مستوى الحمولة الموحدة:
  - services/safety/rules_generated.py            (بوابة الخطر — الخادم)
  - mobile/medai_app/lib/safety/rules_generated.dart (بوابة الخطر — على الجهاز offline)

قواعد التوليد:
  - أي ملف YAML غير صالح أو معرف مكرر أو عتبة مخالفة للعقد ⇒ فشل فوري (exit 1).
  - المخرجات حتمية: نفس المصدر ⇒ نفس البايتات (يفحصه tests/test_codegen_idempotent.py).
"""
import glob
import json
import os
import re
import sys

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES_DIR = os.path.join(ROOT, "safety", "rules")
TPL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

# عقد المعرفات — موسّع موثقاً مقابل مسودة الحزمة ليغطي CV-K-01 وMON-002
ID_PATTERN = re.compile(r"^[A-Z]{2,3}-[A-Z]{0,2}-?[0-9]{2,3}$")
MSG_PATTERN = re.compile(r"^safety\.[a-z_]+(\.[a-z_]+)*$")
NUM_PATTERN = re.compile(r"^\d{3,4}$")
LEVELS = {"EMERGENCY", "URGENT", "SHORT_TERM_MONITORING", "SELF_CARE"}
ACTIONS = {"call_ambulance", "contact_doctor_now", "see_doctor_today", "monitor"}
OPERATORS = {"contains", "subset_of", "gte", "lte", "gt", "lt", "eq", "lab_any"}

DISCLAIMER = "هذه الأداة لا تشخص ولا تعالج. في حالة الطوارئ، اتصل فوراً بخدمات الطوارئ المحلية."


def _fail(msg: str) -> None:
    print("[FAIL]", msg)
    sys.exit(1)


def _validate_condition(cond: dict, rule_id: str) -> None:
    if not isinstance(cond, dict) or "field" not in cond or "operator" not in cond:
        _fail(f"{rule_id}: شرط غير صالح: {cond}")
    if cond["operator"] not in OPERATORS:
        _fail(f"{rule_id}: معامل غير معروف: {cond['operator']}")
    if cond["operator"] == "lab_any" and "loinc" not in (cond.get("value") or {}):
        _fail(f"{rule_id}: lab_any يتطلب loinc في value")


def load_rules():
    paths = sorted(glob.glob(os.path.join(RULES_DIR, "*.yaml")))
    if not paths:
        _fail("لا توجد قواعد في safety/rules/")
    rules = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for doc in yaml.safe_load_all(f):
                if not doc:
                    continue
                doc["_source"] = os.path.relpath(path, ROOT)
                rules.append(doc)

    ids = [r.get("id", "") for r in rules]
    if len(ids) != len(set(ids)):
        _fail(f"معرفات مكررة: {ids}")

    for r in rules:
        rid = r.get("id", "")
        if not ID_PATTERN.match(rid):
            _fail(f"{rid}: معرف غير مطابق لعقد triage.schema.json")
        if "trigger" not in r or "action" not in r:
            _fail(f"{rid}: trigger/action مفقودة")
        act = r["action"]
        if act.get("level") not in LEVELS:
            _fail(f"{rid}: level غير صالح: {act.get('level')}")
        if act.get("action") not in ACTIONS:
            _fail(f"{rid}: action غير صالح: {act.get('action')}")
        if not MSG_PATTERN.match(act.get("message_key", "")):
            _fail(f"{rid}: message_key لا يطابق العقد: {act.get('message_key')}")
        if not NUM_PATTERN.match(str(act.get("default_emergency_number", "997"))):
            _fail(f"{rid}: رقم الطوارئ لا يطابق العقد")
        trigger = r["trigger"]
        for cond in trigger.get("all_of", []):
            _validate_condition(cond, rid)
        for cond in trigger.get("any_of", []):
            _validate_condition(cond, rid)

    versions = {r.get("version") for r in rules}
    if len(versions) != 1:
        _fail(f"إصدارات القواعد غير متسقة: {versions}")
    rules.sort(key=lambda r: r["id"])
    return rules, versions.pop()


def main() -> None:
    rules, rules_version = load_rules()
    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rules]
    # الحمولة الموحدة — نفس السلاسل تُحقن في Python وDart (تكافؤ بايت-ببايت)
    rules_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)

    env = Environment(loader=FileSystemLoader(TPL_DIR), keep_trailing_newline=True)
    ctx = {
        "rules_json": rules_json,
        "rules_version": rules_version,
        "disclaimer": DISCLAIMER,
        "count": len(rules),
    }

    out_py = os.path.join(ROOT, "services", "safety", "rules_generated.py")
    out_dart = os.path.join(ROOT, "mobile", "medai_app", "lib", "safety", "rules_generated.dart")
    os.makedirs(os.path.dirname(out_py), exist_ok=True)
    os.makedirs(os.path.dirname(out_dart), exist_ok=True)

    with open(out_py, "w", encoding="utf-8") as f:
        f.write(env.get_template("rules.py.j2").render(**ctx))
    with open(out_dart, "w", encoding="utf-8") as f:
        f.write(env.get_template("rules.dart.j2").render(**ctx))

    print(f"[OK] وُلّدت قواعد السلامة ({len(rules)} قاعدة، إصدار {rules_version})")
    print(f"     -> {os.path.relpath(out_py, ROOT)}")
    print(f"     -> {os.path.relpath(out_dart, ROOT)}")


if __name__ == "__main__":
    main()
