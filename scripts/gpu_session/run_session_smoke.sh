#!/usr/bin/env bash
# =============================================================================
# run_session_smoke.sh — البروفة العامة (dress rehearsal) لrun_session.sh
#
# ⚠️ تنصل صريح: هذا ليس الجلسة التعاقدية ولا ينتج مرشح ترقية. جرعات مُصغَّرة
#    ومسارات كلها في /tmp/smoke_gpu/ — لا يلمس artifacts/gpu_* ولا
#    docs/results/gpu_* ولا نقطة الخدمة ولا ثوابت الحارس. الغرض الوحيد:
#    إثبات ميكانيكا التدفق كاملاً (0-preflight -> 5أ -> 5ب -> 6أ -> 6ب ->
#    الرباعي ×2 -> التشغيل المزدوج -> PPL -> الأرشيف -> جدول البوابات/القرار)
#    على أي آلة، بما فيها CPU، قبل دفع ثمن جلسة GPU حقيقية.
#
# وضعا التشغيل:
#   1) التدفق الواحد (آلة الجلسة أو أي آلة تسمح بعملية أمامية طويلة):
#        bash scripts/gpu_session/run_session_smoke.sh
#   2) وضع القطع (بيئات تقتل الخلفية وتحصر الاستدعاء الواحد بعشر دقائق —
#      مثل بيئة التطوير هذه): نفس الكود حرفياً يُستدعى قطعة قطعة بالترتيب،
#      والسجل يتراكم على الملف نفسه:
#        SMOKE_STAGE=init   bash scripts/gpu_session/run_session_smoke.sh
#        SMOKE_STAGE=5a     bash scripts/gpu_session/run_session_smoke.sh
#        SMOKE_STAGE=5b     bash scripts/gpu_session/run_session_smoke.sh
#        SMOKE_STAGE=6a     bash scripts/gpu_session/run_session_smoke.sh
#        SMOKE_STAGE=6b     bash scripts/gpu_session/run_session_smoke.sh
#        SMOKE_STAGE=quad1-bench|quad1-ling|quad1-r1k|quad1-palmx  (كذلك quad2-*)
#        SMOKE_STAGE=finalize bash scripts/gpu_session/run_session_smoke.sh
#      (القطعة "quad1" وحدها تشغّل القياسات الأربعة للجولة 1 دفعة واحدة)
#
# الجرعات المُصغَّرة (تُطبع ولا تُخفى):
#   5أ: 3 خطوات batch 2        (العقد: 20,000 خطوة)    بوابة 5ب: < 400 (العقد: < 60)
#   6أ: 4 خطوات                (العقد: 82 خطوة)
#   6ب: --max-steps 2          (العقد: حقبة كاملة على 1,393 زوجاً مختلطاً)
#   الرباعي: bench subset 16 حالة + palmx 12 حالة + ryko1000 --limit 2
#            (ling_probes كامل 52 — ملفه متثبت بايتياً)
#   pytest متعمد الحذف هنا: الحارس الكامل إجباري في الجلسة الحقيقية (§2.6)
#   وقد أُثبت أخضر ×3 في §2 المنفذ؛ البروفة تختبر الميكانيكا لا الحارس.
#
# القرار النهائي للبروفة: جدول البوابات يُطبع بأرقام البروفة الفعلية والرمز
#   (0/1/2) هو رمز خروج قطعة finalize — على جرعة 3 خطوات المتوقع رمز 2
#   (توأم أحمر) وهو النتيجة الصحيحة ميكانيكياً: فرع القرار الثالث يُختبر حيّاً.
# =============================================================================
set -u
cd "$(dirname "$0")/../.." || exit 1          # جذر المستودع دائماً
export PYTHONPATH=src

SROOT=/tmp/smoke_gpu
LOG="$SROOT/session_smoke.log"
STAGE="init"
ALLOW_3=0   # 1 = مرحلة تدريب: الخروج 3 (time-budget) = توقف نظيف — نفس انضباط العقد §3أ
CP="$SROOT/bundle_dpo/final.pt"   # نقطة مرشح البروفة (تُبنى في 6ب)

SMOKE_STAGE="${SMOKE_STAGE:-all}"
want() { [ "$SMOKE_STAGE" = "all" ] && return 0
         local x; for x in "$@"; do [ "$SMOKE_STAGE" = "$x" ] && return 0; done
         return 1; }

fail() { echo "[$(date +%H:%M:%S)] STAGE=$STAGE ABORT: $1" | tee -a "$LOG"; exit 1; }
run()  {
  STAGE="$1"; shift
  echo "[$(date +%H:%M:%S)] == STAGE=$STAGE ==" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 3 ] && [ "$ALLOW_3" = "1" ]; then
    echo "[$(date +%H:%M:%S)] STAGE=$STAGE PAUSED (exit 3 — time-budget): أعد تشغيل نفس الأمر حرفياً للاستئناف (§3أ)" | tee -a "$LOG"
    exit 3
  fi
  [ "$rc" -eq 0 ] || fail "الأمر فشل برمز $rc"
  return 0
}

if want init; then
  rm -rf "$SROOT"; mkdir -p "$SROOT"
  echo "SMOKE REHEARSAL — جرعات مصغرة، كل المخرجات في $SROOT (ليست الجلسة التعاقدية)" > "$LOG"
  echo "العقد المرجعي: docs/GPU_SESSION_PLAN.md (HEAD $(git rev-parse --short HEAD))" >> "$LOG"
  python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" >> "$LOG" 2>&1

  # ------- المرحلة 0: طليعة مختصرة (بايتات + إيصال + قرص) — pytest متعمد حذفه هنا -------
  STAGE="0-preflight"
  echo "[$(date +%H:%M:%S)] == STAGE=$STAGE ==" >> "$LOG"
  sha256sum artifacts/dpo_round2_small/final.pt | tee -a "$LOG" | grep -q "f0361c4a54e01e1a198d2bd00fa2b5be2393d988204dabe7b59916240a52157d" || fail "sha256 نقطة الخدمة غير مطابق"
  sha256sum artifacts/tokenizer/ryko-tokenizer.json | tee -a "$LOG" | grep -q "fe6c68658568b2868d1e0777df70376c9c3184d21c9d114c1c30c6b12d80fce1" || fail "sha256 الرمّاز غير مطابق"
  PYTHONPATH=src python -m ryko_lm.provenance verify data/processed_ling/receipt.json | tail -1 | tee -a "$LOG" | grep -q PASSED || fail "إيصال data/processed_ling غير PASSED"
  FREE_GB=$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')
  [ "${FREE_GB:-0}" -ge 1 ] || fail "القرص الحر ${FREE_GB}GB < 1GB للبروفة"
  echo "[smoke] pytest متعمد حذفه في البروفة — إجباري في الجلسة الحقيقية (§2.6)" | tee -a "$LOG"
  PYTHONPATH=src python -c "import ryko_lm, ryko_lm.training.train, ryko_lm.training.dpo_train, ryko_lm.eval.benchmarks, ryko_lm.eval.ryko1000_eval, ryko_lm.training.pack_bundle; print('imports OK')" | tee -a "$LOG" | grep -q "imports OK" || fail "imports"

  # متجر المعرفة: نسخة معزولة (المتجر الملتزم لا يُمس حتى للقراءة المشتركة)
  cp data/knowledge/ryko.sqlite "$SROOT/store.sqlite"

  # بذور مصغرة حتمية (تُبنى مرة واحدة — الجولتان 1 و2 تستعملان الملف نفسه)
  python - "$SROOT" <<'EOF' | tee -a "$LOG"
import json, pathlib, sys
sroot = pathlib.Path(sys.argv[1])
src = [json.loads(l) for l in open('docs/benchmarks/ryko_bench_seed.jsonl', encoding='utf-8')]
mcq = [c for c in src if c.get('kind') == 'mcq']
sub = ([c for c in mcq if c.get('domain') == 'علاقة-حجاج'][:4]
       + [c for c in mcq if c.get('domain') != 'علاقة-حجاج'][:4]
       + [c for c in src if c.get('kind') == 'abstain'][:4]
       + [c for c in src if c.get('kind') == 'cite'][:4])
p = sroot / 'bench_seed_smoke.jsonl'
p.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in sub), encoding='utf-8')
huj = sum(1 for c in sub if c.get('domain') == 'علاقة-حجاج')
print(f"[smoke] bench subset: {len(sub)} cases (hujja={huj}, abstain=4, cite=4) -> {p}")
EOF
  head -12 data/bench/palmx_ic.jsonl > "$SROOT/palmx_smoke.jsonl"
  echo "[smoke] palmx subset: 12 cases -> $SROOT/palmx_smoke.jsonl" | tee -a "$LOG"
  echo "[smoke] chunk init: DONE" | tee -a "$LOG"
  [ "$SMOKE_STAGE" = "all" ] || exit 0
else
  echo "[smoke chunk] SMOKE_STAGE=$SMOKE_STAGE — $(date +%H:%M:%S)" | tee -a "$LOG"
  [ -d "$SROOT" ] || fail "قطعة init لم تُشغَّل — ابدأ بها"
fi

# -----------------------------------------------------------------------------
# 5أ (مصغرة) — التدريب الواسع من نقطة الخدمة: 3 خطوات batch 2
# -----------------------------------------------------------------------------
if want 5a; then
  STAGE="5a-wide-train-smoke"
  ALLOW_3=1
  run 5a python -m ryko_lm.training.train \
      --config configs/models/ryko-small.yaml \
      --data-dir data/processed_ling --out-dir "$SROOT/training_ling" \
      --init-from artifacts/dpo_round2_small/final.pt \
      --lr 1e-4 --min-lr 1e-5 --total-steps 3 --batch-size 2 --eval-every 1000
  ALLOW_3=0
  [ "$SMOKE_STAGE" = "all" ] || exit 0
fi

# -----------------------------------------------------------------------------
# 5ب (مصغرة) — بوابة PPL: < 400 للبروفة (العقد: < 60) + مسبارات ling كاملة + إيصال
# -----------------------------------------------------------------------------
if want 5b; then
  STAGE="5b-ppl-gate-smoke"
  run 5b-eval python -m ryko_lm.training.evaluate \
      --checkpoint "$SROOT/training_ling/final.pt" \
      --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --data-dir data/processed_ling --split val \
      --batch-size 8 --max-batches 32 --prompts
  PPL_WIDE=$(grep -oE "\[ppl\][^0-9]*val: *[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+ *$" | tr -d ' ')
  [ -n "${PPL_WIDE:-}" ] || fail "لم يُعثر على قيمة PPL في السجل (فحص الregex الحرفي)"
  echo "PPL(بروفة أساس واسع) = $PPL_WIDE — بوابة البروفة: < 400 (العقد: < 60)" | tee -a "$LOG"
  awk "BEGIN{exit !($PPL_WIDE < 400)}" || fail "بوابة بروفة 5ب فشلت: PPL=$PPL_WIDE >= 400"
  run 5b-probes python -m ryko_lm.eval.benchmarks run \
      --benchmark docs/benchmarks/ling_probes.jsonl \
      --checkpoint "$SROOT/training_ling/final.pt" \
      --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store "$SROOT/store.sqlite" --abstain-threshold 0.34 \
      --out "$SROOT/ling_base"
  STAGE="5b-receipt-smoke"
  run 5b-stamp python -m ryko_lm.provenance make "$SROOT/training_ling/final.pt" \
      --kind checkpoint --meta '{"phase":"SMOKE rehearsal 5a (3 steps, batch 2)","rehearsal":true,"init_from":"artifacts/dpo_round2_small/final.pt"}' \
      --out "$SROOT/training_ling/final_receipt.json"
  PYTHONPATH=src python -m ryko_lm.provenance verify "$SROOT/training_ling/final_receipt.json" | tail -1 | tee -a "$LOG" | grep -q PASSED || fail "إيصال بروفة الأساس الواسع"
  [ "$SMOKE_STAGE" = "all" ] || exit 0
fi

# -----------------------------------------------------------------------------
# 6أ (مصغرة) — SFT الحزمة المجمّعة: 4 خطوات بدل 82
# -----------------------------------------------------------------------------
if want 6a; then
  STAGE="6a-bundle-sft-smoke"
  ALLOW_3=1
  run 6a python -m ryko_lm.training.train \
      --config configs/models/ryko-small-sft.yaml \
      --data-dir data/training/bundle_pack \
      --out-dir "$SROOT/bundle_sft" \
      --init-from "$SROOT/training_ling/final.pt" \
      --total-steps 4 --lr 2e-5 --min-lr 2e-6
  ALLOW_3=0
  run 6a-stamp python -m ryko_lm.provenance make "$SROOT/bundle_sft/final.pt" \
      --kind checkpoint --meta '{"phase":"SMOKE rehearsal 6a (4 steps)","rehearsal":true}' \
      --out "$SROOT/bundle_sft/final_receipt.json"
  [ "$SMOKE_STAGE" = "all" ] || exit 0
fi

# -----------------------------------------------------------------------------
# 6ب (مصغرة) — DPO دفاعي: أول 2 خطوة تحسين من الملف المختلط الملتزم
# -----------------------------------------------------------------------------
if want 6b; then
  STAGE="6b-defensive-dpo-smoke"
  run 6b python -m ryko_lm.training.dpo_train \
      --config configs/models/ryko-small-sft.yaml \
      --init-from "$SROOT/bundle_sft/final.pt" \
      --pairs data/training/bundle_dpo_pairs_mixed.jsonl \
      --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --out-dir "$SROOT/bundle_dpo" --epochs 1 --max-steps 2 --lr 1e-5 --beta 0.1 \
      --seed 1337
  run 6b-stamp python -m ryko_lm.provenance make "$SROOT/bundle_dpo/final.pt" \
      --kind checkpoint --meta '{"phase":"SMOKE rehearsal 6b (max-steps 2)","rehearsal":true,"beta":0.1,"lr":1e-5,"seed":1337}' \
      --out "$SROOT/bundle_dpo/final_receipt.json"
  [ "$SMOKE_STAGE" = "all" ] || exit 0
fi

# -----------------------------------------------------------------------------
# الرباعي (مصغر) على المرشح ×2 — بنفس --out المختلف للجولة الثانية (§5.5)
# -----------------------------------------------------------------------------
if want quad1 quad1-bench; then
  run 7-bench-r1 python -m ryko_lm.eval.benchmarks run \
      --benchmark "$SROOT/bench_seed_smoke.jsonl" \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store "$SROOT/store.sqlite" --abstain-threshold 0.65 --out "$SROOT/bench_final"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad1" ] || exit 0
fi
if want quad1 quad1-ling; then
  run 7-ling-r1 python -m ryko_lm.eval.benchmarks run \
      --benchmark docs/benchmarks/ling_probes.jsonl \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store "$SROOT/store.sqlite" --abstain-threshold 0.34 --out "$SROOT/ling_final"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad1" ] || exit 0
fi
if want quad1 quad1-r1k; then
  run 7-r1k-r1 python -m ryko_lm.eval.ryko1000_eval \
      --seed docs/benchmarks/ryko1000_seed.jsonl \
      --store "$SROOT/store.sqlite" \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --mode base --limit 2 --out "$SROOT/ryko1000_final"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad1" ] || exit 0
fi
if want quad1 quad1-palmx; then
  run 7-palmx-r1 python -m ryko_lm.eval.benchmarks run \
      --benchmark "$SROOT/palmx_smoke.jsonl" \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --abstain-threshold 0.65 --out "$SROOT/palmx_final"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad1" ] || exit 0
fi
if want quad2 quad2-bench; then
  run 7-bench-r2 python -m ryko_lm.eval.benchmarks run \
      --benchmark "$SROOT/bench_seed_smoke.jsonl" \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store "$SROOT/store.sqlite" --abstain-threshold 0.65 --out "$SROOT/bench_final_run2"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad2" ] || exit 0
fi
if want quad2 quad2-ling; then
  run 7-ling-r2 python -m ryko_lm.eval.benchmarks run \
      --benchmark docs/benchmarks/ling_probes.jsonl \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store "$SROOT/store.sqlite" --abstain-threshold 0.34 --out "$SROOT/ling_final_run2"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad2" ] || exit 0
fi
if want quad2 quad2-r1k; then
  run 7-r1k-r2 python -m ryko_lm.eval.ryko1000_eval \
      --seed docs/benchmarks/ryko1000_seed.jsonl \
      --store "$SROOT/store.sqlite" \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --mode base --limit 2 --out "$SROOT/ryko1000_final_run2"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad2" ] || exit 0
fi
if want quad2 quad2-palmx; then
  run 7-palmx-r2 python -m ryko_lm.eval.benchmarks run \
      --benchmark "$SROOT/palmx_smoke.jsonl" \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --abstain-threshold 0.65 --out "$SROOT/palmx_final_run2"
  [ "$SMOKE_STAGE" = "all" ] || [ "$SMOKE_STAGE" = "quad2" ] || exit 0
fi

# -----------------------------------------------------------------------------
# الختام: التشغيل المزدوج (§5.5) + PPL النهائية + الأرشفة + جدول البوابات
# -----------------------------------------------------------------------------
if want finalize; then
  STAGE="7-double-run-smoke"
  run 7-double python scripts/gpu_session/double_run_check.py \
      --bench-a "$SROOT/bench_final.json" --bench-b "$SROOT/bench_final_run2.json" \
      --ling-a "$SROOT/ling_final.json" --ling-b "$SROOT/ling_final_run2.json" \
      --palmx-a "$SROOT/palmx_final.json" --palmx-b "$SROOT/palmx_final_run2.json" \
      --r1k-a "$SROOT/ryko1000_final.json" --r1k-b "$SROOT/ryko1000_final_run2.json"

  STAGE="7-final-ppl-smoke"
  run 7-ppl python -m ryko_lm.training.evaluate \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --data-dir data/processed_ling --split val \
      --batch-size 8 --max-batches 32 --prompts
  PPL_FINAL=$(grep -oE "\[ppl\][^0-9]*val: *[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+ *$" | tr -d ' ')
  [ -n "${PPL_FINAL:-}" ] || fail "لم يُعثر على PPL النهائية للبروفة"
  printf '%s\n' "$PPL_FINAL" > "$SROOT/final_ppl.txt"
  echo "PPL(بروفة نقطة نهائية) = $PPL_FINAL — للعلم فقط، السقف الحقيقي <= 101.34" | tee -a "$LOG"

  STAGE="archive-smoke"
  mkdir -p "$SROOT/archive"
  python - "$SROOT" <<'EOF' | tee -a "$LOG"
import hashlib, json, pathlib, sys
sroot = pathlib.Path(sys.argv[1])
dst = sroot / 'archive'
files = ['bench_final.json', 'ling_final.json', 'ryko1000_final.json',
         'palmx_final.json', 'ling_base.json']
idx = {}
for f in files:
    p = sroot / f
    if p.exists():
        (dst / f).write_bytes(p.read_bytes())
        idx[f] = hashlib.sha256((dst / f).read_bytes()).hexdigest()
    else:
        print(f"[archive] MISSING (خطأ ميكانيكي): {f}")
idx['session_log'] = 'session_smoke.log'
(dst / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[archive] {len(idx)} entries -> {dst/'index.json'}")
EOF
  sha256sum $CP | tee -a "$LOG"
  sha256sum $CP | awk '{print $1}' > "$SROOT/candidate_sha256.txt"

  STAGE="gates-smoke"
  PYTHONPATH=src python scripts/gpu_session/check_gates.py \
      --bench "$SROOT/bench_final.json" --ling "$SROOT/ling_final.json" \
      --ryko1000 "$SROOT/ryko1000_final.json" --palmx "$SROOT/palmx_final.json" \
      --ppl-file "$SROOT/final_ppl.txt" | tee -a "$LOG"
  DECISION=${PIPESTATUS[0]}
  echo "[smoke] رمز القرار للبروفة: $DECISION (متوقع 2 على جرعة 3 خطوات — التوأم أحمر ميكانيكياً وفرع §4.3 الثالث هو المُختبر)" | tee -a "$LOG"
  exit "$DECISION"
fi

# وصولاً إلى هنا في وضع all: كل المراحل مرت — البوابات أعلاه هي الخاتمة
if [ "$SMOKE_STAGE" = "all" ]; then
  fail "وضع all انتهى دون قطعة finalize — تدفق مكسور"
fi
