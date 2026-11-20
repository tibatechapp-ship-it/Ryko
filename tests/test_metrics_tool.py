# -*- coding: utf-8 -*-
"""اختبار أداة مؤشرات الصحة (v1.28.0) — عقد المخارج الثابتة بلا شبكة وtorch."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "project_metrics.py"


def test_project_metrics_snapshot_contract():
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, f"exit={out.returncode} stderr={out.stderr[-400:]}"
    snap = json.loads(out.stdout)
    assert snap["schema"] == "ryko-project-metrics/1"
    git = snap["git"]
    assert git["head"] and len(git["head"]) == 40
    assert git["branch"] == "main"
    # الجرد المتتبع ثابت البنية في أي استنساخ نظيف:
    repo = snap["repo"]
    assert repo["tracked_files"] >= 450
    assert repo["loc_src"] >= 10_000
    assert repo["loc_tests"] >= 5_000
    # الرمزنة الملتزمة: Byte-Level BPE 32,768 — عقد مثبت في مكان آخر أيضاً:
    assert snap["tokenizer"]["exists"] is True
    assert snap["tokenizer"]["vocab_size"] == 32_768
    # المعرفة والوثائق:
    assert snap["knowledge"]["receipts_tracked"] >= 20
    assert snap["docs"]["total"] >= 100
    assert snap["docs"]["results_reports"] >= 90
    # نقطة الخدمة المُرقّاة وإيصالها (عقد v1.24.0):
    sv = snap["serving"]
    assert sv["checkpoint"] == "artifacts/dpo_round2_small/final.pt"
    assert sv["checkpoint_exists"] is True
    assert sv["receipt_exists"] is True
