#!/usr/bin/env python3
"""قاعدة التشغيل المزدوج — تنفيذ حرفي لبند 5.5 في docs/GPU_SESSION_PLAN.md.

- `benchmarks run` (القياسات 1 و2 و4): ملفا JSON متطابقان بايت-ببايت
  (تقارير المشروع بلا أختام زمنية) — المطلوب sha256(a)==sha256(b).
- `ryko1000_eval`: التقرير يتضمن `latency_ms` و`ms` لكل صف — هما القياسان
  الوحيدان المسموح اختلافهما؛ المطلوب تطابق كل المقاييس بين الجولتين.

الاستعمال:
    python scripts/gpu_session/double_run_check.py \
        --bench-a docs/results/gpu_bench_final.json --bench-b /tmp/gpu_bench_final_run2.json \
        --ling-a docs/results/gpu_ling_final.json --ling-b /tmp/gpu_ling_final_run2.json \
        --palmx-a docs/results/palmx_ic_v5.json --palmx-b /tmp/palmx_ic_v5_run2.json \
        --r1k-a docs/results/gpu_ryko1000_final.json --r1k-b /tmp/gpu_ryko1000_final_run2.json

أي عدم تطابق => توقف (§5.5): قفزة أجهزة/عدم حتمية — يُحل قبل أي نشر.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strip_r1k(d: dict) -> dict:
    """نسخة القياس من §5.5: تُسقِط latency_ms وms من كل صف."""
    d = json.loads(json.dumps(d))
    d["base"]["latency_ms"] = {}
    for r in d["base"].get("rows", []):
        r.pop("ms", None)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-a", required=True)
    ap.add_argument("--bench-b", required=True)
    ap.add_argument("--ling-a", required=True)
    ap.add_argument("--ling-b", required=True)
    ap.add_argument("--palmx-a", required=True)
    ap.add_argument("--palmx-b", required=True)
    ap.add_argument("--r1k-a", required=True)
    ap.add_argument("--r1k-b", required=True)
    a = ap.parse_args()

    ok = True
    for name, pa, pb in (("bench", a.bench_a, a.bench_b),
                         ("ling", a.ling_a, a.ling_b),
                         ("palmx", a.palmx_a, a.palmx_b)):
        A, B = pathlib.Path(pa), pathlib.Path(pb)
        if not A.exists() or not B.exists():
            print(f"[double-run] {name}: MISSING ({A} / {B})")
            ok = False
            continue
        ha, hb = sha256(A), sha256(B)
        same = ha == hb
        ok = ok and same
        print(f"[double-run] {name}: {'BYTE-IDENTICAL' if same else 'MISMATCH'} sha256={ha[:16]} vs {hb[:16]}")

    A, B = pathlib.Path(a.r1k_a), pathlib.Path(a.r1k_b)
    if not A.exists() or not B.exists():
        print(f"[double-run] ryko1000: MISSING ({A} / {B})")
        ok = False
    else:
        da = strip_r1k(json.loads(A.read_text(encoding="utf-8")))
        db = strip_r1k(json.loads(B.read_text(encoding="utf-8")))
        same = da == db
        ok = ok and same
        print(f"[double-run] ryko1000: {'METRICS IDENTICAL (latency stripped)' if same else 'METRICS MISMATCH'}")

    if ok:
        print("[double-run] كل الجولات الأربع متطابقة — بند 5.5 مستوفى.")
        return 0
    print("[double-run] عدم تطابق => توقف (§5.5): قفزة أجهزة/عدم حتمية — يُحل قبل أي نشر.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
