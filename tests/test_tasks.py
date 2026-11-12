# -*- coding: utf-8 -*-
"""اختبارات مواصفات المهام — عقد صارم بين المطالب والمنفذ."""
import pytest

from ryko.tasks import TaskError, TaskSpec, load_tasks

VALID = {
    "id": "t-999-demo",
    "title": "مهمة اختبار",
    "instruction": "اكتب شيئاً",
    "fixture_dir": "tasks/fixtures/t-001-fib",
    "target_path": "solution.py",
    "test_command": "python3 -m pytest -q",
    "implementer_piece": "implementer-mock-v1",
    "reviewer_piece": "reviewer-mock-v1",
}


def test_valid_task_loads():
    spec = TaskSpec.from_dict(dict(VALID))
    assert spec.max_attempts == 3 and spec.timeout_sec == 60


@pytest.mark.parametrize("missing", ["id", "title", "instruction", "fixture_dir",
                                     "target_path", "test_command",
                                     "implementer_piece", "reviewer_piece"])
def test_missing_field_rejected(missing):
    doc = dict(VALID)
    del doc[missing]
    with pytest.raises(TaskError, match="مفقودة"):
        TaskSpec.from_dict(doc)


def test_unknown_field_rejected():
    doc = dict(VALID, ninja=True)
    with pytest.raises(TaskError, match="غير معروفة"):
        TaskSpec.from_dict(doc)


def test_bad_task_id_rejected():
    with pytest.raises(TaskError, match="معرف"):
        TaskSpec.from_dict(dict(VALID, id="demo_task"))


def test_attempts_bounds():
    with pytest.raises(TaskError, match="max_attempts"):
        TaskSpec.from_dict(dict(VALID, max_attempts=9))
    with pytest.raises(TaskError, match="max_attempts"):
        TaskSpec.from_dict(dict(VALID, max_attempts=0))


def test_timeout_bounds():
    with pytest.raises(TaskError, match="timeout_sec"):
        TaskSpec.from_dict(dict(VALID, timeout_sec=1))


def test_load_task_from_file(tmp_path):
    import yaml
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(VALID, allow_unicode=True), encoding="utf-8")
    spec = TaskSpec.load(str(p))
    assert spec.id == "t-999-demo"


def test_load_tasks_from_dir_sorted(tmp_path):
    import yaml
    for tid in ("t-002-b", "t-001-a"):
        doc = dict(VALID, id=tid)
        (tmp_path / f"{tid}.yaml").write_text(
            yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    specs = load_tasks([str(tmp_path)])
    assert [s.id for s in specs] == ["t-001-a", "t-002-b"]
