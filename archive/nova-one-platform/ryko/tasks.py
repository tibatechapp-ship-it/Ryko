"""مواصفات المهام — عقد صارم بين المطالب والمنفذ (عقود أولاً)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml

TASK_ID = re.compile(r"^t-\d{3,4}-[a-z0-9-]+$")
FAILURE_MODES = ("none", "repair_after", "always_broken", "review_reject")


class TaskError(ValueError):
    pass


@dataclass
class TaskSpec:
    id: str
    title: str
    instruction: str
    fixture_dir: str
    target_path: str
    test_command: str
    implementer_piece: str
    reviewer_piece: str
    max_attempts: int = 3
    timeout_sec: int = 60
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "TaskSpec":
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        return TaskSpec.from_dict(doc, source=path)

    @staticmethod
    def from_dict(doc: Dict[str, Any], source: str = "<memory>") -> "TaskSpec":
        if not isinstance(doc, dict):
            raise TaskError(f"{source}: المهمة يجب أن تكون خريطة")
        required = (
            "id", "title", "instruction", "fixture_dir", "target_path",
            "test_command", "implementer_piece", "reviewer_piece",
        )
        missing = [k for k in required if k not in doc]
        if missing:
            raise TaskError(f"{source}: حقول إلزامية مفقودة: {missing}")
        unknown = sorted(set(doc) - set(required) - {"max_attempts", "timeout_sec"})
        if unknown:
            raise TaskError(f"{source}: حقول غير معروفة (عقد صارم): {unknown}")
        if not TASK_ID.match(doc["id"]):
            raise TaskError(f"{source}: معرف مهمة غير مطابق: {doc['id']!r}")
        attempts = int(doc.get("max_attempts", 3))
        if not 1 <= attempts <= 5:
            raise TaskError(f"{source}: max_attempts بين 1 و5")
        timeout = int(doc.get("timeout_sec", 60))
        if not 5 <= timeout <= 600:
            raise TaskError(f"{source}: timeout_sec بين 5 و600")
        return TaskSpec(
            id=doc["id"],
            title=doc["title"],
            instruction=doc["instruction"],
            fixture_dir=doc["fixture_dir"],
            target_path=doc["target_path"],
            test_command=doc["test_command"],
            implementer_piece=doc["implementer_piece"],
            reviewer_piece=doc["reviewer_piece"],
            max_attempts=attempts,
            timeout_sec=timeout,
        )


def load_tasks(paths: List[str]) -> List[TaskSpec]:
    """يحمّل ملفات مهام من مسارات أو مجلدات (بترتيب المعرف)."""
    specs: List[TaskSpec] = []
    for p in paths:
        if os.path.isdir(p):
            files = sorted(
                os.path.join(p, name) for name in os.listdir(p)
                if name.endswith((".yaml", ".yml")) and not name.startswith("_")
            )
        else:
            files = [p]
        for fp in files:
            specs.append(TaskSpec.load(fp))
    specs.sort(key=lambda s: s.id)
    return specs
