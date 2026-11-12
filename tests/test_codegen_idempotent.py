# -*- coding: utf-8 -*-
"""اختبار حتمية المولّد — نفس المصدر ⇒ نفس البايتات (ADR-0002)."""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = [
    os.path.join(ROOT, "services", "safety", "rules_generated.py"),
    os.path.join(ROOT, "mobile", "medai_app", "lib", "safety", "rules_generated.dart"),
]


def _hashes():
    return {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in GENERATED}


def _run_generator():
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "safety", "codegen", "generator.py")],
        check=True, cwd=ROOT,
    )


def test_generator_is_deterministic_byte_for_byte():
    _run_generator()
    first = _hashes()
    _run_generator()
    second = _hashes()
    for path in GENERATED:
        assert first[path] == second[path], f"المولّد غير حتمي: {path}"


def test_generated_files_exist_and_embed_rules_payload():
    for path in GENERATED:
        assert os.path.exists(path) and os.path.getsize(path) > 0, f"مفقود: {path}"
    py_src = open(GENERATED[0], encoding="utf-8").read()
    dart_src = open(GENERATED[1], encoding="utf-8").read()
    assert "RULES_JSON" in py_src and "kRulesJson" in dart_src
    # القاعدة الصلبة حاضرة في المنصتين
    assert "RF-010" in py_src and "RF-010" in dart_src
    assert "CV-K-01" in py_src and "CV-K-01" in dart_src
