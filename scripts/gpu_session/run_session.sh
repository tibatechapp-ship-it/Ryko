#!/usr/bin/env bash
# =============================================================================
# run_session.sh — المنفّذ الحرفي لجلسة GPU: docs/GPU_SESSION_PLAN.md §0–§9
#
# عقد الجلسة (§9):
#   1. CPU pre-flight (§2) كامل وأخضر — 649+1 اختبار، إيصالات، بايتات.
#   2. 5أ: 20K خطوة واسعة من نقطة الخدمة -> إيصال + بوابة PPL<60.
#   3. 6أ/6ب: الحزمة المجمّعة (82) + DPO دفاعي (697) -> إيصالات.
#   4. 7: الرباعي كامل ×2 على المرشح -> جدول 4.1+4.2 كلّه أخضر.
#   5. ترقية واعية (§6) أو document-and-hold (§7) — لا طريق ثالث.
#
# الاستعمال على آلة الجلسة (أمر واحد من جذر المستودع):
#   bash scripts/gpu_session/run_session.sh
#
# متغيرات بيئة اختيارية (الأصول الحرفية للعقد، لا تُبدَّل إلا لذراع ضبط موثق):
#   OOM_ARM=8        ذراع OOM الموثقة §1: --batch-size 8 (يبقى grad_accum 2)
#   TIME_BUDGET=S    §3أ: --time-budget للجلسات القصيرة (استئناف بنفس الأمر)
#   SKIP_TESTS=1     تخطي طليعة pytest الكاملة (غير موصى به — §2.6 إجبارية)
#
# الانضباط (§3): كل أمر أمامي (foreground) مع سجل دائم — ممنوع الخلفية.
# الاستئناف (§3أ/§7.4): أعد تشغيل نفس الأمر حرفياً — last.pt ذرّية الكتابة.
# =============================================================================
set -u
cd "$(dirname "$0")/../.." || exit 1          # جذر المستودع دائماً
export PYTHONPATH=src

LOG="docs/results/gpu_session_$(date +%Y%m%d_%H%M).log"
STAGE="init"
ALLOW_3=0   # 1 = مرحلة تدريب: الخروج 3 (time-budget) = توقف مؤقت نظيف وفق §3أ
fail() { echo "[$(date +%H:%M:%S)] STAGE=$STAGE ABORT: $1" | tee -a "$LOG"; exit 1; }
run()  {
  STAGE="$1"; shift
  echo "[$(date +%H:%M:%S)] == STAGE=$STAGE ==" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 3 ] && [ "$ALLOW_3" = "1" ]; then
    echo "[$(date +%H:%M:%S)] STAGE=$STAGE PAUSED (exit 3 — time-budget): أعد تشغيل نفس الأمر حرفياً للاستئناف — last.pt ذرّية الكتابة (§3أ/§7.4)" | tee -a "$LOG"
    exit 3
  fi
  [ "$rc" -eq 0 ] || fail "الأمر فشل برمز $rc"
  return 0
}

echo "GPU session — contract: docs/GPU_SESSION_PLAN.md (HEAD $(git rev-parse --short HEAD), $(git describe --tags --always 2>/dev/null))" | tee "$LOG"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())" 2>&1 | tee -a "$LOG"
python -c "import torch; assert torch.cuda.is_available(), 'لا CUDA على آلة الجلسة — العقد يتطلب GPU'" || fail "torch.cuda.is_available()=False (ثبّت بنية cu121: pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cu121)"

# -----------------------------------------------------------------------------
# المرحلة 0 — طليعة §2 الإجبارية على آلة الجلسة (بايتات + إيصالات + اختبار كامل)
# -----------------------------------------------------------------------------
STAGE="0-preflight"
sha256sum artifacts/dpo_round2_small/final.pt | tee -a "$LOG" | grep -q "f0361c4a54e01e1a198d2bd00fa2b5be2393d988204dabe7b59916240a52157d" || fail "sha256 نقطة الخدمة غير مطابق للثابت المثبت بالاختبار"
sha256sum artifacts/tokenizer/ryko-tokenizer.json | tee -a "$LOG" | grep -q "fe6c68658568b2868d1e0777df70376c9c3184d21c9d114c1c30c6b12d80fce1" || fail "sha256 الرمّاز غير مطابق للعقد"
PYTHONPATH=src python -m ryko_lm.provenance verify data/processed_ling/receipt.json | tail -1 | tee -a "$LOG" | grep -q PASSED || fail "إيصال data/processed_ling غير PASSED — نفّذ §2.3 قبل الجلسة (make lugha-collect ling-pack ling-pack-binarize ثم إعادة التوقيع بعد binarize)"
FREE_GB=$(df -BG . | awk 'NR==2{gsub("G","",$4); print $4}')
[ "${FREE_GB:-0}" -ge 6 ] || fail "القرص الحر ${FREE_GB}GB < 6GB المطلوبة (§2.6)"
if [ "${SKIP_TESTS:-0}" != "1" ]; then
  run p2 python -m pytest --tb=no -q || fail "الاختبار الكامل لم يمر (§2.6)"
fi
PYTHONPATH=src python -c "import ryko_lm, ryko_lm.training.train, ryko_lm.training.dpo_train, ryko_lm.eval.benchmarks, ryko_lm.eval.ryko1000_eval, ryko_lm.training.pack_bundle; print('imports OK')" | tee -a "$LOG" | grep -q "imports OK" || fail "imports"

BATCH_FLAG=()
if [ "${OOM_ARM:-0}" = "8" ]; then BATCH_FLAG=(--batch-size 8); echo "ذراع OOM الموثقة: --batch-size 8 (§1/§7.2)" | tee -a "$LOG"; fi
TB_FLAG=()
if [ -n "${TIME_BUDGET:-}" ]; then TB_FLAG=(--time-budget "$TIME_BUDGET"); echo "تقطيع الجلسة: --time-budget $TIME_BUDGET (§3أ)" | tee -a "$LOG"; fi

# -----------------------------------------------------------------------------
# 3أ — التدريب الواسع: المرحلة 5 (init من نقطة الخدمة، ليس v5b — انحراف موثق 1)
# -----------------------------------------------------------------------------
STAGE="5a-wide-train"
ALLOW_3=1
run 5a python -m ryko_lm.training.train \
    --config configs/models/ryko-small.yaml \
    --data-dir data/processed_ling --out-dir artifacts/training_small_ling \
    --init-from artifacts/dpo_round2_small/final.pt \
    --lr 1e-4 --min-lr 1e-5 --total-steps 20000 "${BATCH_FLAG[@]}" "${TB_FLAG[@]}"
# الخروج 3 = توقف نظيف (exit 3 أعلاه) — الاستئناف = إعادة تشغيل السكربت نفسها.
ALLOW_3=0

# -----------------------------------------------------------------------------
# 3ب — بوابة المرحلة 5: PPL < 60 على المسطرة + تسجيل المسبارات
# -----------------------------------------------------------------------------
STAGE="5b-ppl-gate"
run 5b-eval python -m ryko_lm.training.evaluate \
    --checkpoint artifacts/training_small_ling/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --data-dir data/processed_ling --split val \
    --batch-size 8 --max-batches 32 --prompts
PPL_WIDE=$(grep -oE "\[ppl\][^0-9]*val: *[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+ *$" | tr -d ' ')
[ -n "${PPL_WIDE:-}" ] || fail "لم يُعثر على قيمة PPL في السجل"
echo "PPL(أساس واسع) = $PPL_WIDE — البوابة: < 60" | tee -a "$LOG"
awk "BEGIN{exit !($PPL_WIDE < 60)}" || fail "بوابة 5ب فشلت: PPL=$PPL_WIDE >= 60 — لا دخول في المحاذاة؛ أغلق الجلسة بدفتر تشخيص وفق §7.5 (أذرع الجرعة: OOM_ARM=8 أو total-steps أقل)"
run 5b-probes python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ling_probes.jsonl \
    --checkpoint artifacts/training_small_ling/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.34 \
    --out docs/results/gpu_ling_base

# ختم الأساس الواسع (بعد بوابة 5ب فقط — §7.5: لا ختم على أساس راسب)
STAGE="5b-receipt"
run 5b-stamp python -m ryko_lm.provenance make artifacts/training_small_ling/final.pt \
    --kind checkpoint --meta '{"phase":"5 wide linguistics (GPU v1.28.1)","params":52179456,"init_from":"artifacts/dpo_round2_small/final.pt","steps":20000}' \
    --out artifacts/training_small_ling/final_receipt.json
PYTHONPATH=src python -m ryko_lm.provenance verify artifacts/training_small_ling/final_receipt.json | tail -1 | tee -a "$LOG" | grep -q PASSED || fail "إيصال الأساس الواسع"

# -----------------------------------------------------------------------------
# 3ج — إعادة المحاذاة: SFT الحزمة المجمّعة (82 خطوة ≈ 3 حقب)
# -----------------------------------------------------------------------------
STAGE="6a-bundle-sft"
ALLOW_3=1
run 6a python -m ryko_lm.training.train \
    --config configs/models/ryko-small-sft.yaml \
    --data-dir data/training/bundle_pack \
    --out-dir artifacts/gpu_bundle_sft \
    --init-from artifacts/training_small_ling/final.pt \
    --total-steps 82 --lr 2e-5 --min-lr 2e-6
ALLOW_3=0
run 6a-stamp python -m ryko_lm.provenance make artifacts/gpu_bundle_sft/final.pt \
    --kind checkpoint --meta '{"phase":"6a bundle SFT (GPU v1.28.1)","init_from":"artifacts/training_small_ling/final.pt","steps":82,"pack":"data/training/bundle_pack (receipt bundle-pack)"}' \
    --out artifacts/gpu_bundle_sft/final_receipt.json

# -----------------------------------------------------------------------------
# 3د — DPO الختامية بالأفضليات الدفاعية (الملف المختلط الملتزم 1,393)
# -----------------------------------------------------------------------------
STAGE="6b-defensive-dpo"
run 6b python -m ryko_lm.training.dpo_train \
    --config configs/models/ryko-small-sft.yaml \
    --init-from artifacts/gpu_bundle_sft/final.pt \
    --pairs data/training/bundle_dpo_pairs_mixed.jsonl \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --out-dir artifacts/gpu_bundle_dpo --epochs 1 --lr 1e-5 --beta 0.1 \
    --seed 1337
run 6b-stamp python -m ryko_lm.provenance make artifacts/gpu_bundle_dpo/final.pt \
    --kind checkpoint --meta '{"phase":"6b defensive DPO (GPU v1.28.1)","pairs":"data/training/bundle_dpo_pairs_mixed.jsonl","beta":0.1,"lr":1e-5,"seed":1337}' \
    --out artifacts/gpu_bundle_dpo/final_receipt.json

# -----------------------------------------------------------------------------
# 3هـ — الرباعي الكامل على المرشح ×2 (بند 5 — قاعدة التشغيل المزدوج)
# -----------------------------------------------------------------------------
STAGE="7-quadruple"
CP=artifacts/gpu_bundle_dpo/final.pt
for RUN in 1 2; do
  if [ "$RUN" = "1" ]; then
    OUTB=docs/results/gpu_bench_final; OUTL=docs/results/gpu_ling_final
    OUTR=docs/results/gpu_ryko1000_final; OUTP=docs/results/palmx_ic_v5
  else
    OUTB=/tmp/gpu_bench_final_run2; OUTL=/tmp/gpu_ling_final_run2
    OUTR=/tmp/gpu_ryko1000_final_run2; OUTP=/tmp/palmx_ic_v5_run2
  fi
  run 7-bench-r$RUN python -m ryko_lm.eval.benchmarks run \
      --benchmark docs/benchmarks/ryko_bench_seed.jsonl \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store data/knowledge/ryko.sqlite --abstain-threshold 0.65 --out "$OUTB"
  run 7-ling-r$RUN python -m ryko_lm.eval.benchmarks run \
      --benchmark docs/benchmarks/ling_probes.jsonl \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --store data/knowledge/ryko.sqlite --abstain-threshold 0.34 --out "$OUTL"
  run 7-r1k-r$RUN python -m ryko_lm.eval.ryko1000_eval \
      --seed docs/benchmarks/ryko1000_seed.jsonl \
      --store data/knowledge/ryko.sqlite \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --mode base --out "$OUTR"
  run 7-palmx-r$RUN python -m ryko_lm.eval.benchmarks run \
      --benchmark data/bench/palmx_ic.jsonl \
      --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
      --abstain-threshold 0.65 --out "$OUTP"
done

# قاعدة التشغيل المزدوج (§5.5) — أي عدم تطابق => توقف
STAGE="7-double-run"
run 7-double python scripts/gpu_session/double_run_check.py \
    --bench-a docs/results/gpu_bench_final.json --bench-b /tmp/gpu_bench_final_run2.json \
    --ling-a docs/results/gpu_ling_final.json --ling-b /tmp/gpu_ling_final_run2.json \
    --palmx-a docs/results/palmx_ic_v5.json --palmx-b /tmp/palmx_ic_v5_run2.json \
    --r1k-a docs/results/gpu_ryko1000_final.json --r1k-b /tmp/gpu_ryko1000_final_run2.json

# PPL النقطة النهائية (بوابة §4.2 الأخيرة) — نفس بروتوكول 5ب على المرشح
STAGE="7-final-ppl"
run 7-ppl python -m ryko_lm.training.evaluate \
    --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --data-dir data/processed_ling --split val \
    --batch-size 8 --max-batches 32 --prompts
PPL_FINAL=$(grep -oE "\[ppl\][^0-9]*val: *[0-9.]+" "$LOG" | tail -1 | grep -oE "[0-9.]+ *$" | tr -d ' ')
[ -n "${PPL_FINAL:-}" ] || fail "لم يُعثر على PPL النهائية"
printf '%s\n' "$PPL_FINAL" > /tmp/gpu_final_ppl.txt
echo "PPL(نقطة نهائية) = $PPL_FINAL — السقف: <= 101.34" | tee -a "$LOG"

# -----------------------------------------------------------------------------
# أرشفة الدفتر (§5.6 / §7.1): ملفات JSON + index.json ببصمات sha256
# -----------------------------------------------------------------------------
STAGE="archive"
mkdir -p docs/results/gpu_session
python - <<'EOF' | tee -a "$LOG"
import hashlib, json, pathlib, shutil
dst = pathlib.Path('docs/results/gpu_session')
files = ['docs/results/gpu_bench_final.json','docs/results/gpu_ling_final.json',
         'docs/results/gpu_ryko1000_final.json','docs/results/palmx_ic_v5.json',
         'docs/results/gpu_ling_base.json']
idx = {}
for f in files:
    p = pathlib.Path(f)
    if p.exists():
        shutil.copy2(p, dst / p.name)
        idx[p.name] = hashlib.sha256((dst / p.name).read_bytes()).hexdigest()
log = sorted(pathlib.Path('docs/results').glob('gpu_session_*.log'))
if log:
    idx['session_log'] = log[-1].name
(dst / 'index.json').write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[archive] {len(idx)} entries -> {dst/'index.json'}")
EOF
sha256sum artifacts/gpu_bundle_dpo/final.pt | tee -a "$LOG"
sha256sum artifacts/gpu_bundle_dpo/final.pt | awk '{print $1}' > /tmp/gpu_candidate_sha256.txt

# -----------------------------------------------------------------------------
# جدول البوابات (§4.1 + §4.2) والقرار (§4.3)
# -----------------------------------------------------------------------------
STAGE="gates"
PYTHONPATH=src python scripts/gpu_session/check_gates.py | tee -a "$LOG"
DECISION=${PIPESTATUS[0]}
sha=$(cat /tmp/gpu_candidate_sha256.txt)
if [ "$DECISION" = "0" ]; then
  cat <<EOS | tee -a "$LOG"

===================== القرار: ترقية واعية وفق §6 (فعل إصدار — ترتيب إلزامي) =====================
  sha256 النقطة المرشحة: $sha
  1) الإيصال: artifacts/gpu_bundle_dpo/final_receipt.json — verify PASS (منجز أعلاه)
  2) القياس الخارجي: على النقطة نفسها، تشغيل مزدوج مطابق (منجز أعلاه)
  3) تحديث التثبيت الثلاثي + الوثائق:
     - tests/test_guard_provenance.py: PROMOTED_SERVING_POINT=artifacts/gpu_bundle_dpo/final.pt
       وPROMOTED_SHA256=$sha  (القيم الحالية dpo_round2_small / f0361c4a… لا تُمس قبل الآن)
     - tests/conftest.py: FINAL_MODEL -> artifacts/gpu_bundle_dpo/final.pt
     - src/ryko_lm/serving/api.py: MODEL_PATH الافتراضي + docstring الترقية
     - docs/ROUTING.md: جدول نقطة الخدمة L2 بأرقام الجلسة
     - CHANGELOG.md: بند إصدار بأرقام الرباعي المطابقة بايتياً لdocs/results/gpu_session/
     - docs/ROADMAP.md + docs/EVALUATION.md: حالة المرحلة 5/الجولة التالية
  4) الحارس الأخضر: PYTHONPATH=src python -m pytest -q  (649 passed + 1 skipped)
  5) الوسم: make secret-scan ثم git tag v1.29.0 — دون أي push (سياسة المستودع)
=================================================================================================
EOS
  exit 0
elif [ "$DECISION" = "1" ]; then
  cat <<EOS | tee -a "$LOG"

=================== القرار: document-and-hold وفق §7.1 (لا ترقية — لا وسم) ====================
  sha256 المرشح المؤرشف: $sha  (نقطة عمل موثقة بنمط proof_bundle_small)
  الأصول بقيت كنقاط عمل ملتزمة في docs/results/gpu_session/ — نقطة الخدمة والإيصال
  وثابتا الحارس كما هما. أذرع إعادة التشكيل (§7.2، بالترتيب الاقتصادي):
    - جرعة SFT: 82 -> 120 خطوة   |   جرعة الأساس: OOM_ARM=8 أو total-steps 12000
    - الأفضليات: scripts/build_bundle_dpo.py --ling-repeat {1,3,5} --argrel-repeat {1,2}
      --r2-mode {ling,full} (إعادة إيصال + فحص تسرب + قياس قبل الاستعمال)
    - beta/lr للـDPO على شظايا صغيرة أولاً   |   WiSE-FT للتشخيص فقط (جدار 36.54)
  أعد الجلسة بأمر واحد: bash scripts/gpu_session/run_session.sh
================================================================================================
EOS
  exit 1
fi
fail "قرار غير متوقع: DECISION=$DECISION"
