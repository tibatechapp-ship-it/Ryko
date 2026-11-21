#!/usr/bin/env python3
"""بوابة القبول — تنفيذ حرفي لجدول docs/GPU_SESSION_PLAN.md §4.1 + §4.2.

كل الأرقام تُقاس على نقطة واحدة: artifacts/gpu_bundle_dpo/final.pt
(ملفات JSON للرباعي الكامل + قيمة PPL المسطرة على النقطة النهائية).

الاستعمال:
    PYTHONPATH=src python scripts/gpu_session/check_gates.py \
        --bench docs/results/gpu_bench_final.json \
        --ling docs/results/gpu_ling_final.json \
        --ryko1000 docs/results/gpu_ryko1000_final.json \
        --palmx docs/results/palmx_ic_v5.json \
        --ppl-file /tmp/gpu_final_ppl.txt

رموز الخروج: 0 = كل البوابات خضراء (ترقية وفق §6) | 1 = التوأم أخضر وبوابة
عدم تراجع انكسرت (document-and-hold وفق §7.1) | 2 = التوأم نفسه أحمر أو ملف ناقص.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# ---- العتبات الحرفية من §4.1 (التوأم) و§4.2 (عدم التراجع) — لا رقم مُختلَق ----
TWIN = {
    "rb_ling": 0.4423,          # ≥ 23/52 — نقطة الخدمة ling_probes_round2.json
    "hujja": 0.70,              # ≥ 21/30 — التجربة 2 و5 و6 = 76.67
}
NO_REGRESSION = {
    "internal_mcq": 0.37,       # ≥ 59/159 — نقطة الخدمة 39.62
    "internal_correct_abstain": 1.00,   # = 40/40 خارج-المتن
    "internal_wrong_abstain": 0.65,     # ≤ 10/16 — 0.6875 هو المستوى المرفوض تاريخياً
    "internal_cite": 1.00,      # = 100/100
    "palmx_ic": 0.2905,         # ≥ 285/981 — رقم نقطة الخدمة palmx_ic_v4.json
    "palmx_ic_target": 0.3109,  # الهدف: ≥ 305/981 (استرداد كلفة المحاذاة −2.04)
    "ryko1000_mcq": 0.3371,     # ≥ 337/1000 — نقطة الخدمة ryko1000_eval_dpo_r2.json
    "ryko1000_mcq_target": 0.4460,      # الهدف: شظية argrel
    "ryko1000_correct_abstain": 1.00,   # = 36/36
    "ryko1000_wrong_abstain": 0.65,     # ≤ 41/64
    "istidlal": 0.50, "mustalahat": 0.40, "tarjih": 0.35,
    "final_ppl": 101.34,        # سقف منع النسيان الكارثي (LINGUISTICS §5)
}

rows: list[tuple[str, object, object, bool, str]] = []  # (gate, measured, threshold, ok, note)


def load(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[gates] MISSING FILE: {p}")
        sys.exit(2)
    return json.loads(p.read_text(encoding="utf-8"))


def add(name: str, measured, threshold, ok: bool, note: str) -> None:
    rows.append((name, measured, threshold, ok, note))


def rel(name: str, measured, threshold, op: str, note: str) -> None:
    """بوابة معدل محدد اختيارياً: None (0/0 — لا حالات من هذا الجانب) => FAIL نظيف لا انهيار."""
    if measured is None:
        add(name, "UNDEFINED", threshold, False,
            note + " — لا حالات من هذا الجانب (المعدل غير معرّف)")
        return
    ok = (measured >= threshold) if op == ">=" else (measured <= threshold)
    add(name, measured, threshold, ok, note)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="docs/results/gpu_bench_final.json")
    ap.add_argument("--ling", default="docs/results/gpu_ling_final.json")
    ap.add_argument("--ryko1000", default="docs/results/gpu_ryko1000_final.json")
    ap.add_argument("--palmx", default="docs/results/palmx_ic_v5.json")
    ap.add_argument("--ppl-file", default="/tmp/gpu_final_ppl.txt")
    a = ap.parse_args()

    bench = load(a.bench)["results"]
    ling = load(a.ling)["results"]["mcq"]
    r1k = load(a.ryko1000)
    palmx = load(a.palmx)["results"]["mcq"]

    base = r1k["base"]
    fams = base["mcq"].get("per_family", {})

    # ---- §4.1 التوأم ----
    ling_acc = ling["accuracy"]
    add("twin rb-ling (52 @0.34) >=44.23%", ling_acc, TWIN["rb_ling"],
        ling_acc >= TWIN["rb_ling"], f"{round(ling_acc*52)} / 52")

    huj = bench["mcq"]["per_domain"]["علاقة-حجاج"]
    add("twin علاقة-حجاج (من 315 @0.65) >=70%", huj["rate"], TWIN["hujja"],
        huj["rate"] >= TWIN["hujja"], f"{round(huj['rate']*huj['n'])} / {huj['n']}")

    twin_ok = rows[-2][3] and rows[-1][3]

    # ---- §4.2 عدم التراجع ----
    m = bench["mcq"]
    add("internal MCQ (159 @0.65) >=37%", m["accuracy"], NO_REGRESSION["internal_mcq"],
        m["accuracy"] >= NO_REGRESSION["internal_mcq"], f"{round(m['accuracy']*m['n'])} / {m['n']}")

    ab = bench["abstain"]
    rel("internal امتنان صائب =1.00", ab.get("correct_abstain_rate"),
        NO_REGRESSION["internal_correct_abstain"], ">=", "خارج-المتن")
    rel("internal رفض زائد <=0.65", ab.get("wrong_abstain_rate"),
        NO_REGRESSION["internal_wrong_abstain"], "<=", "16 حالة")

    add("internal cite =1.0", bench["cite"]["match_rate"],
        NO_REGRESSION["internal_cite"],
        bench["cite"]["match_rate"] >= NO_REGRESSION["internal_cite"],
        f"{round(bench['cite']['match_rate']*bench['cite']['n'])} / {bench['cite']['n']}")

    add("PalmX-IC (981 @0.65) >=29.05%", palmx["accuracy"], NO_REGRESSION["palmx_ic"],
        palmx["accuracy"] >= NO_REGRESSION["palmx_ic"],
        f"{round(palmx['accuracy']*palmx['n'])} / {palmx['n']} (target >={NO_REGRESSION['palmx_ic_target']})")
    add("RYKO-1000 base MCQ >=33.71%", base["mcq"]["accuracy"],
        NO_REGRESSION["ryko1000_mcq"],
        base["mcq"]["accuracy"] >= NO_REGRESSION["ryko1000_mcq"],
        f"{round(base['mcq']['accuracy']*base['mcq']['n'])} / {base['mcq']['n']} (target >={NO_REGRESSION['ryko1000_mcq_target']})")

    rab = base["abstain"]
    rel("RYKO-1000 امتنان صائب =1.00", rab.get("correct_abstain_rate"),
        NO_REGRESSION["ryko1000_correct_abstain"], ">=", "36 حالة")
    rel("RYKO-1000 رفض زائد <=0.65", rab.get("wrong_abstain_rate"),
        NO_REGRESSION["ryko1000_wrong_abstain"], "<=", "64 حالة")

    for fam, th in (("istidlal", 0.50), ("mustalahat", 0.40), ("tarjih", 0.35)):
        f = fams.get(fam)
        if f is None:
            add(f"per_family {fam} >={th}", "MISSING", th, False, "العائلة غائبة من التقرير")
        else:
            add(f"per_family {fam} >={th}", f["rate"], th, f["rate"] >= th,
                f"{round(f['rate']*f['n'])} / {f['n']}")

    # PPL النهائية (ملف نصي يحوي قيمة واحدة يكتبه run_session.sh من سجل evaluate)
    ppl_path = pathlib.Path(a.ppl_file)
    if not ppl_path.exists():
        add("PPL مسطرة اللغة على النقطة النهائية <=101.34", "MISSING",
            NO_REGRESSION["final_ppl"], False, str(ppl_path))
    else:
        ppl = float(ppl_path.read_text().strip())
        add("PPL مسطرة اللغة على النقطة النهائية <=101.34", ppl,
            NO_REGRESSION["final_ppl"], ppl <= NO_REGRESSION["final_ppl"],
            "val_loss <= ln(101.34)=4.619")

    # ---- الطباعة ----
    print()
    print("=== جدول البوابات — docs/GPU_SESSION_PLAN.md §4.1 + §4.2 ===")
    n_fail = 0
    for name, measured, th, ok, note in rows:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"[{mark}] {name:52s} measured={measured!s:12s} thr={th!s:8s} ({note})")

    twin_row = [r for r in rows if r[0].startswith("twin")]
    twin_ok = all(r[3] for r in twin_row)
    print()
    if n_fail == 0:
        print("DECISION: كل البوابات خضراء على نقطة واحدة واعية => ترقية وفق §6 (الحارس + conftest + api + ROUTING + CHANGELOG + وسم بعد المسح السري).")
        return 0
    if twin_ok:
        print("DECISION: التوأم أخضر وبوابة/بوابات عدم تراجع انكسرت => لا ترقية — document-and-hold وفق §7.1 (دفتر gpu_session/ + أذرع إعادة التشكيل §7.2).")
        return 1
    print("DECISION: التوأم نفسه غير مكتمل => لا ترقية — مراجعة الجرعة (§7.2) وإعادة من 3أ.")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # انهيار الفاحص نفسه لا يتنكر أبداً كقرار عقد (0/1/2)
        print(f"[gates] CHECKER CRASHED: {type(e).__name__}: {e}")
        print("[gates] رمز 4 = خلل في الفاحص — راجع المدخلات، لا يُقرأ كقرار §7.1.")
        sys.exit(4)
