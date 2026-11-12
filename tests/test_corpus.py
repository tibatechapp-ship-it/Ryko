# -*- coding: utf-8 -*-
"""اختبارات سلامة الأصول — الفهرس يُحصن باختبار (معيار 06 §5)."""
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STYLE_DIR = os.path.join(REPO_ROOT, "corpus", "style")

EXPECTED_STANDARDS = [
    "01-naming.md",
    "02-project-structure.md",
    "03-testing-standards.md",
    "04-review-checklist.md",
    "05-commit-and-tags.md",
    "06-docs-standards.md",
    "07-security-basics.md",
    "08-arabic-content-rules.md",
]


def test_index_lists_every_standard():
    with open(os.path.join(STYLE_DIR, "README.md"), encoding="utf-8") as f:
        index = f.read()
    for name in EXPECTED_STANDARDS:
        assert name in index, f"المعيار {name} غير مفهرس"


def test_every_file_on_disk_is_indexed():
    with open(os.path.join(STYLE_DIR, "README.md"), encoding="utf-8") as f:
        index = f.read()
    on_disk = sorted(f for f in os.listdir(STYLE_DIR)
                     if f.endswith(".md") and f != "README.md")
    assert on_disk == sorted(EXPECTED_STANDARDS), "ملف بلا فهرس أو فهرس بلا ملف"


def test_every_standard_has_version_line():
    for name in EXPECTED_STANDARDS:
        with open(os.path.join(STYLE_DIR, name), encoding="utf-8") as f:
            head = f.read(400)
        assert re.search(r"v\d+\.\d+\.\d+", head), f"{name} بلا إصدار"


def test_no_todo_left_in_corpus():
    for name in EXPECTED_STANDARDS:
        with open(os.path.join(STYLE_DIR, name), encoding="utf-8") as f:
            assert "TODO" not in f.read(), f"{name} يبقَ فيه TODO"


def test_no_emoji_in_corpus():
    for name in EXPECTED_STANDARDS:
        with open(os.path.join(STYLE_DIR, name), encoding="utf-8") as f:
            text = f.read()
        assert not any(ch in text for ch in "✅❌🚫🎉"), f"{name} يحوي إيموجي"
