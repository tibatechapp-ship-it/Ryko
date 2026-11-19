# خطة جلسة GPU — الوصفة §6 (التدريب الواسع + إعادة المحاذاة) — v1.27.0

> وثيقة التنفيذ لجلسة GPU الواحدة المطلوبة لتحقيق الهدفين المزدوجين (rb-ling ≥44.23 و علاقة-حجاج ≥70 معًا) — تكمل `docs/LINGUISTICS.md §6` و`docs/results/bundle_treatment.md`.

## 0) لماذا GPU؟ (خلاصة الجدار الموثق)

سبع تجارب مضبوطة على CPU (دفتر المعالجات) أثبتت أن SFT الواسع مطلوب لكسر مقايضة rb-ling/حجاج: SFT الضيق يفتح rb-ling (−13.46 حتى −17.31)، وDPO بعد SFT يمحو المعرفة الجديدة بلا أفضليات دفاعية. الوصفة §6 تبني القاعدة الواسعة **أولًا** ثم تعيد المحاذاة **فوقها** — الترتيب الذي لم يُجرَّب على CPU لأن التدريب الواسع 34.24M رمز يحتاج GPU.

## 1) الأجهزة والزمن المتوقع

| المرحلة | العتاد | الزمن التقديري (A100 40GB / RTX 4090) |
|---|---|---|
| 5أ — التدريب الواسع 20,000 خطوة | GPU واحد ≥24GB | ~3.5–6 ساعات (ryko-small 52.18M، batch 8×512) |
| 5ب — بوابة PPL<60 + مسبارات الإسناد | CPU | ~20 دقيقة قياس |
| 6أ — SFT الحزمة المجمّعة | GPU | ~15 دقيقة |
| 6ب — DPO الختامية بأفضليات دفاعية | GPU | ~10 دقائق |
| 7 — الرباعي الكامل (داخلي 315 + rb-ling + RYKO-1000 + PalmX 981) | CPU/GPU | ~20 دقيقة |

## 2) الأوامر الجاهزة (بالترتيب، بلا تعديل)

```bash
# ─── 5أ) التدريب الواسع على الحزمة اللغوية الموقّعة (34,240,204 رمز تدريب) ───
python -m ryko_lm.training.train --config configs/models/ryko-small.yaml \
    --data-dir data/processed_ling --out-dir artifacts/training_small_ling \
    --init-from artifacts/training_v5b/final.pt --lr 1e-4 --total-steps 20000
# بديل: make ling-train-small

# ─── 5ب) بوابة القبول: PPL < 60 على المسطرة + مسبارات الإسناد ───
python -m ryko_lm.eval.perplexity --checkpoint artifacts/training_small_ling/final.pt
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ling_probes.jsonl \
    --checkpoint artifacts/training_small_ling/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.34 \
    --out docs/results/gpu_ling_base

# ─── 6أ) SFT الحزمة المجمّعة v1.26.0 (740 زوجًا — جاهزة بإيصال) ───
PYTHONPATH=src python -m ryko_lm.training.train \
    --config configs/models/ryko-small-sft.yaml \
    --data-dir data/training/bundle_pack \
    --out-dir artifacts/gpu_bundle_sft \
    --init-from artifacts/training_small_ling/final.pt \
    --total-steps 82 --lr 2e-5 --min-lr 2e-6

# ─── 6ب) DPO الختامية بالأفضليات الدفاعية (حجاج 105 + لغة 116 + r2 مرشّحة) ───
PYTHONPATH=src python -m ryko_lm.training.dpo_train \
    --config configs/models/ryko-small-sft.yaml \
    --init-from artifacts/gpu_bundle_sft/final.pt \
    --pairs data/training/bundle_dpo_pairs_mixed.jsonl \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --out-dir artifacts/gpu_bundle_dpo --epochs 1 --lr 1e-5 --beta 0.1 \
    --seed 1337

# ─── 7) الرباعي الكامل على المرشح gpu_bundle_dpo ───
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ryko_bench_seed.jsonl \
    --checkpoint artifacts/gpu_bundle_dpo/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.65 \
    --out docs/results/gpu_bench_final
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ling_probes.jsonl \
    --checkpoint artifacts/gpu_bundle_dpo/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.34 \
    --out docs/results/gpu_ling_final
PYTHONPATH=src python -m ryko_lm.eval.ryko1000_eval \
    --seed docs/benchmarks/ryko1000_seed.jsonl \
    --store data/knowledge/ryko.sqlite \
    --checkpoint artifacts/gpu_bundle_dpo/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --mode base --out docs/results/gpu_ryko1000_final
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark data/bench/palmx_ic.jsonl \
    --checkpoint artifacts/gpu_bundle_dpo/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --abstain-threshold 0.65 --out docs/results/palmx_ic_v6
```

## 3) بوابات القبول (كلها معًا أو لا إصدار)

| البوابة | العتبة | المرجع |
|---|---|---|
| PPL المسطرة بعد 5أ | **< 60** | §6 / ROADMAP المرحلة 5 |
| rb-ling (52 @0.34) | **≥ 44.23** | بوابة اللغة المنشورة |
| علاقة-حجاج (من 315) | **≥ 70** | هدف الحزمة المجمّعة |
| امتناب صحيح | **1.00** | بوابة v1.14.0 الفعلية |
| cite match | **1.0** | بوابة التوثيق |
| MCQ داخلي | **> 55.35** | أفضل نقطة مقاسة |
| PalmX-IC (981) | **≥ 29.05** | رقم نقطة الخدمة المنشور (لا تراجع) |
| istidlal/mustalahat/tarjih | **≥ 44/51/44** (مكتسبة) | قياسات v1.26.0 |
| البناء المزدوج | **مطابقة بايتية** | عقد المشروع الدائم |

## 4) انضباط الجلسة

1. كل أصل جديد بإيصال (`make_receipt`) قبل الانتقال للمرحلة التالية.
2. أي فشل بوابة ⇒ **لا ترقية** — توثيق قياس بصدق وإعادة تشكيل (أذرع الضبط الجاهزة: `--ling-repeat`/`--argrel-repeat` في `scripts/build_bundle_dpo.py`، وجرعة SFT بين 82–120 خطوة).
3. الترقية تتم فقط بعد: تحديث `PROMOTED_SERVING_POINT`/`PROMOTED_SHA256` في `tests/test_guard_provenance.py` + `conftest.FINAL_MODEL` + دفتر `api.py` + `docs/ROUTING.md` + الرباعي الكامل المنشور.
4. الوسم: `v1.27.0` بعد مرور الاختبار الكامل والمسح السري وصحة الإيصالات.
