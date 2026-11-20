#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""مؤشرات صحة المشروع — لقطة واحدة بأمر واحد (v1.28.0).

أداة الجرد الكمي القابلة للتكرار: بلا شبكة، بلا torch، بلا اعتماديات خارج
المكتبة القياسية — تُنتج لقطة JSON (أو Markdown مع ‎--md) بمؤشرات المستودع:

- git: HEAD والفرع وحالة الشجرة
- الجرد: الملفات المتتبعة وأسطر الكود (src/tests/scripts)
- الرمزنة: حجم المفردات الفعلي من artifacts/tokenizer/ryko-tokenizer.json
- المعرفة: عدد الإيصالات المتتبعة + وجود قاعدة المستودع
- الخدمة: نقطة الخدمة المُرقّاة وإيصالها

هذه أداة جردٍ لا بوابة: بوابات الجودة (اختبارات/أسرار/إيصالات) تجري في CI
وحزمة pytest؛ هذه الأداة تُغذي وثيقة الرؤية والفجوة (docs/VISION_GAP.md)
بأرقام متجددة عند كل إصدار.

الاستخدام:
    python scripts/project_metrics.py            # JSON إلى stdout
    python scripts/project_metrics.py --md out.md --out out.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVING_CHECKPOINT = "artifacts/dpo_round2_small/final.pt"
SERVING_RECEIPT = "artifacts/dpo_round2_small/final_receipt.json"
TOKENIZER_JSON = "artifacts/tokenizer/ryko-tokenizer.json"
KNOWLEDGE_DB = "data/knowledge/ryko.sqlite"


def _git(*args: str) -> str | None:
    """أمر git واحد آمن — يعيد None إن لم يكن المستودع متاحاً (نقش/أرشيف)."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def _tracked_files() -> list[str]:
    listing = _git("ls-files")
    return [] if listing is None else [l for l in listing.splitlines() if l]


def _loc(paths: list[str], prefix: str) -> int:
    total = 0
    for rel in paths:
        if not rel.startswith(prefix) or not rel.endswith(".py"):
            continue
        try:
            text = (ROOT / rel).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        total += text.count("\n") + (0 if text.endswith("\n") or not text else 1)
    return total


def _tokenizer_vocab() -> dict:
    path = ROOT / TOKENIZER_JSON
    if not path.is_file():
        return {"exists": False, "vocab_size": None}
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        vocab = data.get("model", {}).get("vocab", {})
        return {"exists": True, "vocab_size": len(vocab)}
    except (OSError, ValueError, AttributeError):
        return {"exists": True, "vocab_size": None}


def _serving_point() -> dict:
    ckpt = ROOT / SERVING_CHECKPOINT
    receipt = ROOT / SERVING_RECEIPT
    return {
        "checkpoint": SERVING_CHECKPOINT,
        "checkpoint_exists": ckpt.is_file(),
        "checkpoint_bytes": ckpt.stat().st_size if ckpt.is_file() else None,
        "receipt": SERVING_RECEIPT,
        "receipt_exists": receipt.is_file(),
    }


def snapshot() -> dict:
    tracked = _tracked_files()
    receipts = [f for f in tracked if f.endswith("receipt.json")]
    docs = [f for f in tracked if f.startswith("docs/")]
    results = [f for f in docs if f.startswith("docs/results/")]
    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    dirty = None if status is None else (len(status) > 0)
    return {
        "schema": "ryko-project-metrics/1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"head": head, "branch": branch, "dirty": dirty},
        "repo": {
            "tracked_files": len(tracked),
            "loc_src": _loc(tracked, "src/"),
            "loc_tests": _loc(tracked, "tests/"),
            "loc_scripts": _loc(tracked, "scripts/"),
        },
        "tokenizer": _tokenizer_vocab(),
        "knowledge": {
            "receipts_tracked": len(receipts),
            "db_exists": (ROOT / KNOWLEDGE_DB).is_file(),
        },
        "docs": {"total": len(docs), "results_reports": len(results)},
        "serving": _serving_point(),
    }


def to_markdown(s: dict) -> str:
    loc = s["repo"]
    tok = s["tokenizer"]
    kn = s["knowledge"]
    dg = s["docs"]
    sv = s["serving"]
    git = s["git"]
    lines = [
        "# لقطة مؤشرات المشروع (ryko-project-metrics/1)",
        "",
        f"- **الوقت (UTC):** {s['generated_at_utc']}",
        f"- **git:** head `{git['head']}` · فرع `{git['branch']}` · "
        f"شجرة نظيفة: {git['dirty'] is False}",
        f"- **الملفات المتتبعة:** {loc['tracked_files']}",
        f"- **أسطر الكود:** src={loc['loc_src']} · tests={loc['loc_tests']} · "
        f"scripts={loc['loc_scripts']}",
        f"- **الرمزنة:** exists={tok['exists']} · حجم المفردات={tok['vocab_size']}",
        f"- **المعرفة:** إيصالات متتبعة={kn['receipts_tracked']} · "
        f"قاعدة المستودع موجودة={kn['db_exists']}",
        f"- **الوثائق:** {dg['total']} ملفاً منها {dg['results_reports']} تقرير نتائج",
        f"- **نقطة الخدمة:** {sv['checkpoint']} exists={sv['checkpoint_exists']} · "
        f"إيصال exists={sv['receipt_exists']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="لقطة مؤشرات صحة مشروع Ryko")
    parser.add_argument("--out", help="كتابة JSON إلى ملف")
    parser.add_argument("--md", help="كتابة ملخص Markdown إلى ملف")
    ns = parser.parse_args(argv)

    s = snapshot()
    payload = json.dumps(s, ensure_ascii=False, indent=2)
    print(payload)
    if ns.out:
        Path(ns.out).write_text(payload + "\n", encoding="utf-8")
    if ns.md:
        Path(ns.md).write_text(to_markdown(s), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
