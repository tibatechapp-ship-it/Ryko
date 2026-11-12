# -*- coding: utf-8 -*-
"""اختبارات سلسلة التدقيق — كشف التلاعب إلزامي (معيار 03 §4)."""
import json

import pytest

from ryko.audit import AuditLog, GENESIS_HASH, canonical


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "audit.jsonl")


def test_empty_log_verifies(log_path):
    log = AuditLog(log_path)
    ok, bad = log.verify()
    assert ok and bad is None and log.entries() == []


def test_append_creates_chained_entries(log_path):
    log = AuditLog(log_path)
    e1 = log.append(action="a", actor="x", resource="r1", payload={"k": 1})
    e2 = log.append(action="b", actor="x", resource="r2", payload={"k": 2})
    assert e1["prev_hash"] == GENESIS_HASH
    assert e2["prev_hash"] == e1["hash"]
    assert log.verify() == (True, None)


def test_genesis_hash_shape():
    assert GENESIS_HASH == "0" * 64 and len(GENESIS_HASH) == 64


def test_canonical_is_deterministic():
    assert canonical({"b": 1, "a": "م"}) == canonical({"a": "م", "b": 1})


def test_tampering_payload_detected(log_path):
    log = AuditLog(log_path)
    log.append(action="a", actor="x", resource="r", payload={"v": 1})
    log.append(action="b", actor="x", resource="r", payload={"v": 2})
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entry = json.loads(lines[0])
    entry["payload"] = {"v": 999}  # تعديل بأثر رجعي
    lines[0] = json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    assert AuditLog(log_path).verify() == (False, 0)


def test_tampering_deletion_detected(log_path):
    log = AuditLog(log_path)
    for i in range(3):
        log.append(action=f"a{i}", actor="x", resource="r", payload={})
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    del lines[1]  # إزالة سجل وسطي
    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    assert AuditLog(log_path).verify() == (False, 1)


def test_entries_are_jsonl_with_unicode(log_path):
    log = AuditLog(log_path)
    log.append(action="a", actor="x", resource="r", payload={"س": "عربي"})
    with open(log_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 1 and lines[0]["payload"]["س"] == "عربي"


def test_verify_cli(log_path, capsys):
    from ryko.audit import _main
    log = AuditLog(log_path)
    log.append(action="x", actor="cli", resource="res")
    assert _main(["verify", log_path]) == 0
    assert "سليمة" in capsys.readouterr().out
