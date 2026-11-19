# خطة جلسة GPU — الوصفة §6 (التدريب الواسع + إعادة المحاذاة) — v1.27.0

> عقد تنفيذ نهائي ومُتحقق منه سطراً بسطر ضد المستودع عند وسم v1.26.0
> (commit `1996089`). يكمّل `docs/LINGUISTICS.md §6` و`docs/results/bundle_treatment.md`.
> كل أمر هنا جرى التحقق من وجود ملفاته وأعلامه (flags) عند سطور الشيفرة الفعلية،
> وكل عتبة في جدول البوابات مبررة من سجل قياس JSON منشور — لا رقم مُختلَق.
> النثر بالعربية والأوامر بالإنجليزية جاهزة للنسخ الحرفي.

---

## 0) لماذا GPU؟ (خلاصة الجدار الموثق)

سبع تجارب مضبوطة على CPU (`docs/results/bundle_treatment.md` + ملفات
`docs/results/bundle_experiments/` الـ17 المفهرسة بـsha256) أثبتت أن SFT الواسع
مطلوب لكسر مقايضة rb-ling/حجاج: جرعة SFT تؤكل rb-ling (44.23←34.62←26.92)،
وDPO بعد SFT **بلا** أفضليات دفاعية يمحو المعرفة الجديدة ويكسر معايرة الامتناب
(0.275–0.95)، والتقاطع الخطي (WiSE-FT) يسقف rb-ling عند نقطة الخدمة (36.54 عند
α=0.5). الوصفة §6 تبني القاعدة الواسعة **أولًا** ثم تعيد المحاذاة **فوقها** —
والترتيب الذي لم يُجرَّب على CPU لأن التدريب الواسع (34,240,402 توكن) يحتاج GPU.

**انحرافان موثقان سبقاهما ويقيَّدان هنا رسميًا:**
1. **نقطة الإحماء** (درس §6.1 الموثق في v1.22.0): من نقطة DPO/الخدمة
   لا من `training_v5b`. التحقق المباشر: `training_v5b/final.pt` نقطة Tiny
   (11.5M معلم، n_embd 256 / 4 طبقات / ctx 256) — تحميلها في معمارية
   `ryko-small` يرفع `RuntimeError: size mismatch` فورًا (مُثبت بالتجربة).
   نقطة الخدمة `artifacts/dpo_round2_small/final.pt` (52,179,456 معلمًا)
   تُحمَّل نظيفة: 110 مصفوفات، missing=0 unexpected=0.
2. **أرضية cosine**: قمة `--lr 1e-4` مع أرضية الكونفج الافتراضية `6e-5`
   (المضبوطة لقمة `6e-4`) تُبقي lr عند 60% من القمة نهاية الجولة — لذا
   يُمرَّر `--min-lr 1e-5` صراحةً (نمط §6.1: أفق cosine كامل 1e-4 ← 1e-5،
   ونمط ×0.1 الثابت في كل كونفجات المشروع).

---

## 1) الأجهزة والجرعة والزمن المتوقع

| المرحلة | العتاد | الحساب | الزمن التقديري (A100 40GB / RTX 4090) |
|---|---|---|---|
| 5أ — التدريب الواسع | GPU واحد ≥24GB | 20,000 خطوة × batch 32 × grad_accum 2 × ctx 512 = **655,360,000 توكن ≈ 19.1 حقبة** على حزمة 34,240,402 توكن | ~2–4.5 ساعة (عند 40–90k توكن/ث) |
| 5ب — بوابة PPL + مسبارات | CPU | مسطرة 148,631 توكن + 52 مسبارًا | ~20 دقيقة |
| 6أ — SFT الحزمة المجمّعة | GPU | 82 خطوة × 2,048 توكن = 167,936 توكن ≈ **3.0 حقبة** من 55,695 توكن | ~5–10 دقائق |
| 6ب — DPO الختامية | GPU | 1,393 زوجًا ÷ batch 2 = **697 خطوة** (epoch واحد) | ~10–15 دقيقة |
| 7 — الرباعي الكامل ×2 (تشغيل مزدوج) | CPU (بروتوكول منشور) | 315 + 52 + 1,300 + 981 حالة | ~40–60 دقيقة |

- **ذاكرة 5أ**: مصفوفة logits عند batch 32 = 32×512×32768 fp32 ≈ **2.15GB
  للنسخة** — تناسب بطاقة 24GB بمعلم 52M؛ عند أي OOM: `--batch-size 8`
  (يبقى grad_accum 2 → 8,192 توكن/خطوة ≈ 4.8 حقبة، زمن ~0.75–1.5 ساعة).
- الدرس المضبوط من CPU (v1.20.0): batch 4 على مضيف 3GB يُحدث OOM بنفس
  السبب (268MB للنسخة) — الحجم متدرج مع الذاكرة لا مع المعمارية.

---

## 2) ما قبل الجلسة على CPU (pre-flight — قسم إجباري)

كل ما يلي **قابل للتنفيذ والتحقق قبل شحن أي وقت GPU**. الأوامر المعلَّمة
بـ✅ جرى تنفيذها فعلاً أثناء تجهيز هذه الوثيقة على نسخة v1.26.0 نظيفة
(commit `1996089`) مع `torch 2.15.0+cpu`، ومخرجاتها المسجلة أدناه هي
**المخرجات المتوقعة حرفيًا** على آلة الجلسة.

### 2.1) البيئة والاعتماديات

```bash
git clone https://github.com/tibatechapp-ship-it/Ryko && cd Ryko
git checkout v1.26.0                      # أو main عند 1996089
git rev-parse --short HEAD                # المتوقع: 1996089

python --version                          # المتوقع: 3.12.x
# GPU: ثبّت بنية CUDA أولًا — مع السقوط الآمن لفهرس CPU عند غياب المطابقة،
# ثم بقية الاعتماديات من requirements.txt الملتزم (tokenizers/numpy/torch/
# fastapi/uvicorn/pydantic/pyarrow/pandas/pytest/httpx/PyYAML/matplotlib):
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cu121 \
    || pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export PYTHONPATH=src                     # قاعدة بيئة دائمة للجلسة كلها
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# المتوقع على آلة الجلسة: ...+cu121 True   (على CPU: +cpu False)
PYTHONPATH=src python -c "import ryko_lm, ryko_lm.training.train, ryko_lm.training.dpo_train, ryko_lm.eval.benchmarks, ryko_lm.eval.ryko1000_eval, ryko_lm.training.pack_bundle; print('imports OK')"   # ✅ imports OK
```

### 2.2) التحقق البايتي من الأصول الملتزمة (كلها ✅ منفَّذة محليًا)

```bash
# 1) نقطة الخدمة — يجب أن تطابق الثابت المثبت بالاختبار:
sha256sum artifacts/dpo_round2_small/final.pt
# ✅ المتوقع: f0361c4a54e01e1a198d2bd00fa2b5be2393d988204dabe7b59916240a52157d

# 2) إيصالات الأصول الداخلة في الجلسة (تحقق دون اتصال):
for r in artifacts/dpo_round2_small/final_receipt.json \
         artifacts/training_v5b/final_receipt.json \
         artifacts/proof_bundle_small/final_receipt.json \
         artifacts/proof_argrel_small/final_receipt.json \
         data/training/bundle_pack/receipt.json \
         data/training/argrel_pack/receipt.json \
         data/knowledge/ryko.receipt.json \
         data/knowledge/ryko1000.receipt.json ; do
    PYTHONPATH=src python -m ryko_lm.provenance verify "$r" | tail -1
done
# ✅ المتوقع: PASSED للثمانية كلها

# 3) الحزمة المجمّعة: مطابقة jsonl ↔ مانيفست ↔ receipt (البايتات):
PYTHONPATH=src python - <<'EOF'
import json, hashlib, pathlib
man = json.load(open('data/training/bundle_pack_manifest.json'))
for k in ('train','val'):
    p = pathlib.Path(man['files'][k]['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == man['files'][k]['sha256'], k
print('bundle jsonl<->manifest sha256 OK',
      man['files']['train']['n'], 'train /', man['files']['val']['n'], 'val')
EOF
# ✅ المتوقع: bundle jsonl<->manifest sha256 OK 740 train / 29 val

# 4) بروتوكولات القياس المجمَّدة (قيم مقفلة باختبارات قائمة):
sha256sum docs/benchmarks/ling_probes.jsonl docs/benchmarks/ryko1000_seed.jsonl \
          docs/benchmarks/ryko_bench_seed.jsonl data/bench/palmx_ic.jsonl \
          data/training/bundle_dpo_pairs_mixed.jsonl
# ✅ المتوقع:
#  a48ec0a2f5bc9571cdb0697448823f9c96517073b9128724f991e50a995493ca  ling_probes.jsonl
#  a5613cb020067a21d1be3bb3ddc3858e5477f9f548915e808f592cbba05f1d25  ryko1000_seed.jsonl
#  9c1849f58437bb47d1c049a3ba29d30dfe543b6218ca8f6a3c90804dc17e3983  ryko_bench_seed.jsonl
#  ad027c9d92f1d0738044a06dffb761eae0820a8d40f8a5bd1e3be8a813ae7859  palmx_ic.jsonl
#  b101dd352bdfaf22c95c744061578f30368f22dac6a4aea81f00d539ec1d980c  bundle_dpo_pairs_mixed.jsonl

# 5) سجل تجارب v1.26.0 (17 ملفًا مفهرسًا) — الاختبار يفشل عند أي انحراف:
PYTHONPATH=src python -m pytest -q tests/test_bundle_pack.py tests/test_guard_provenance.py tests/test_palmx.py
# ✅ المتوقع: exit 0 — 20 حالة خضراء (11 guard + 4 bundle_pack + 5 palmx)

# 6) الرمّاز المعتمد (يدخل في كل إيصالات الحزم):
sha256sum artifacts/tokenizer/ryko-tokenizer.json
# ✅ المتوقع: fe6c68658568b2868d1e0777df70376c9c3184d21c9d114c1c30c6b12d80fce1
```

> ملاحظة على `scripts/verify_all_receipts.py`: مسار الاستيراد/الجذر فيه
> يفترض تخطيط مجلدات مختلفًا (يفشل بـModuleNotFoundError من جذر هذا
> المستودع) — استخدم حلقة `ryko_lm.provenance verify` أعلاه بدلًا منه.
> حلقة شاملة على كل الإيصالات الملتزمة محليًا تعطي **17 PASSED / 5 FAILED
> متوقعًا** — والإخفاقات الخمسة كلها لأصول ثقيلة مقصودة خارج git
> (`data/processed_ling/*.bin`، `processed_v3/v4/v5/v52`، `ryko-wide-full.sqlite`)
> تُعاد بناؤها محليًا (بند 2.3). على آلة الجلسة يجب أن يصير
> `data/processed_ling/receipt.json` PASSED قبل بدء 5أ.

### 2.3) حزمة اللغة (إجباري قبل 5أ — الباينات خارج git)

`data/processed_ling/` الملتزمة تحوي stats + receipt فقط؛ الملفات الثقيلة
(`train.bin` 68,480,804 بايت = 34,240,402 توكن / `val.bin` 297,262 بايت =
148,631 توكن) تُعاد بناؤها حتميًا من المصدر الخام ثم **يُلزم أن يمر الإيصال**:

```bash
make lugha-collect          # الجالب الاستئنافي (89 كتاباً — ملف تقدم واحد)
make ling-pack && make ling-pack-binarize
PYTHONPATH=src python -m ryko_lm.provenance verify data/processed_ling/receipt.json
# الإلزام: PASSED — sha256 train.bin = 2d6b4f2d…e471 ، val.bin = e87b880d…5d41 (داخل الإيصال الملتزم)
```
*(ملاحظتان: (1) `make lugha-build-full` غير مطلوب للجلسة — القياسات كلها
على المخزن الملتزم `ryko.sqlite` والمسبارات الملتزمة، فلا حاجة لإعادة
بناء القاعدة الكاملة. (2) بديل مقبول: نقل ملفات `train.bin`/`val.bin` من
بيئة موثوقة ثم تحقق الإيصال مباشرة — البايتات هي العقد لا طريقة النقل.)*

### 2.4) إعادة بناء الحزمة المجمّعة وحتميتها (✅ منفَّذ — ~40 ثانية)

```bash
rm -rf /tmp/bundle_recheck /tmp/bundle_recheck_pack
PYTHONPATH=src python -m ryko_lm.training.pack_bundle \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --out /tmp/bundle_recheck --pack-out /tmp/bundle_recheck_pack
# المتوقع حرفيًا: rows train=740 val=29 | argrel 90 | ling-sft 102 |
#   weakest 255 | replay 293 (probe-overlap excluded 7) | leakage 0
#   train 740 pairs 55,695 tokens sha256=a650f7ea28bc7a8d… |
#   val 29 pairs 1,701 tokens sha256=c398cbe685348668…
#   [budget] tokens/step=2048 → ~3 epochs ≈ 82 steps (raw 82, capped 120)
```
البايتات المعاد بناؤها تطابق الملتزم (`a650f7ea…`/`c398cbe6…` = إيصال
`bundle-pack`) — ✅ مثبت أثناء التجهيز. ملاحظة: `receipt_sha256` الجديد
يختلف عن الملتزم لأن الإيصال يختم `created_utc` — العقد هو تطابق
**الأصول** لا ختم الإيصال نفسه. 82 خطوة في 6أ مشتقة من هذه الطباعة
(≈3 حقب)، وأي تغيير مستقبلي للحزمة يغيّرها آليًا.

### 2.5) فحص دخول المُعلَّمات (flags) — ✅ منفَّذ بـ --help

`ryko_lm.training.train` (`--config --data-dir --out-dir --total-steps
--batch-size --lr --min-lr --init-from --time-budget --eval-every
--eval-batches --reset --fresh-best`)، `ryko_lm.training.dpo_train`
(`--config --init-from --pairs --tokenizer --out-dir --epochs --batch-size
--beta --lr --min-lr --seed --max-steps --save-every --time-budget`)،
`ryko_lm.eval.benchmarks run` (`--benchmark --checkpoint --tokenizer
--config --store --abstain-threshold --out --device`)،
`ryko_lm.eval.ryko1000_eval` (`--seed --store --checkpoint --tokenizer
--config --mode {base,rag,both,engine,merge} --parts --slice --skip-engine
--out --device --limit`)، `scripts/build_bundle_dpo.py` (`--ling-repeat
--argrel-repeat --r2-mode {ling,full} ...`)، `scripts/soup_checkpoints.py`
(`--a --b --alpha --out`)، `ryko_lm.provenance make/verify` — كلها موجودة.

### 2.6) الاختبار الكامل قبل شحن GPU

```bash
PYTHONPATH=src python -m pytest -q     # الإلزام: 246 passed, 1 skipped (موثق)
df -h .                                # الإلزام: ≥6GB حرّة (الحزمة ~1.1GB + نقاط تدريب ~0.6GB + سجلات)
```

---

## 3) الأوامر الجاهزة بالترتيب (كل أمر نُقض ضد الشيفرة)

انضباط التنفيذ (يعادل الأمامية foreground مع سجل دائم — ممنوع الخلفية):

```bash
LOG=docs/results/gpu_session_$(date +%Y%m%d_%H%M).log
# كل أمر يُنفَّذ هكذا (أمامي بالكامل + سجل يُقرأ أثناء التقدم):
#   PYTHONPATH=src python -m ... 2>&1 | tee -a "$LOG"
```

### 3أ) التدريب الواسع — المرحلة 5 (init من نقطة الخدمة، ليس v5b)

```bash
PYTHONPATH=src python -m ryko_lm.training.train \
    --config configs/models/ryko-small.yaml \
    --data-dir data/processed_ling --out-dir artifacts/training_small_ling \
    --init-from artifacts/dpo_round2_small/final.pt \
    --lr 1e-4 --min-lr 1e-5 --total-steps 20000
# بديل مكافئ بعد إصلاحه: make ling-train-small (أُصلح init-from إلى نقطة الخدمة)
```

- **الاستئناف الدقيق**: نفس الأمر يعاد تشغيله حرفيًا — `last.pt` يُحمَّل
  (model + optimizer + step)، و`rng = seed + step` يضمن تيار نوافذ جديداً
  (درس anti-memorization داخل `train.py`). الخروج `3` = متوقف مؤقتًا،
  `0` = مكتمل. **التقطيع**: أضف `--time-budget 3300` للجلسات القصيرة —
  يتوقف بسجلة عند الحد ويُستأنف بنفس الأمر بلا فقد.
- **أثر الإحماء الموثق**: لحظات AdamW تُعاد وإحماء warmup (500 خطوة من
  الكونفج) يُعاد — مكتوب في سجل التشغيل بذاته (`[init] warm start ...`).
- **سقف الوقت**: عند 19 حقبة على 88 مستندًا لغويًا فإن التحفظ (memorization)
  متوقع ومقصود ضمنياً بالوصفة — لكن ذراع الجرعة النصفية جاهز إن انكسرت
  بوابات عدم التراجع لاحقًا: `--batch-size 8` (≈4.8 حقب) أو
  `--total-steps 12000` (≈11.4 حقبة) — قراره بوابة 5ب ثم 7 لا ذوق.

### 3ب) بوابة المرحلة 5: PPL < 60 على المسطرة + تسجيل المسبارات

```bash
# PPL على مسطرة اللغة (الوحدة الصحيحة: ryko_lm.training.evaluate —
# لا وجود لوحدة ryko_lm.eval.perplexity في المستودع):
PYTHONPATH=src python -m ryko_lm.training.evaluate \
    --checkpoint artifacts/training_small_ling/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --data-dir data/processed_ling --split val \
    --batch-size 8 --max-batches 32 --prompts
# المتوقع مطبوعاً: [ppl] val: <60.00   (خط أساس موثق للمقارنة: 101.34 لشظية CPU 300 خطوة)
# ملاحظة بروتوكولية: قيمة summary.json من 3أ تُحسب على 16 دفعة عشوائية
# بينما أمر evaluate هنا على 32 دفعة من المسطرة نفسها — القيمتان قريبتان
# تقريبياً لا متطابقتان؛ البوابة تحكم على قيمة أمر evaluate.

# تسجيل المسبارات اللغوية على نقطة الأساس الواسع (معلومة قياس — البوابة
# المزدوجة تُفرض في 7 بعد إعادة المحاذاة):
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ling_probes.jsonl \
    --checkpoint artifacts/training_small_ling/final.pt \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.34 \
    --out docs/results/gpu_ling_base
```
*(عتبة المسبارات 0.34 هي عتبة بروتوكول rb-ling المنشور؛ المسبارات 52 حالة
mcq صرفة فلا يُستهلك --store فعلياً — يُمرَّر لمطابقة شكل الأوامر التاريخية.)*

**قرار 5ب**: PPL ≥ 60 ⇒ لا انتقال إلى 6أ — راجع أذرع الجرعة (عدد خطوات/
batch) وأعد 3أ. PPL < 60 ⇒ انتقل.

### 3ج) إعادة المحاذاة فوق الأساس الواسع — SFT الحزمة المجمّعة (v1.26.0 كما هي، بإيصال)

```bash
PYTHONPATH=src python -m ryko_lm.training.train \
    --config configs/models/ryko-small-sft.yaml \
    --data-dir data/training/bundle_pack \
    --out-dir artifacts/gpu_bundle_sft \
    --init-from artifacts/training_small_ling/final.pt \
    --total-steps 82 --lr 2e-5 --min-lr 2e-6
```
(نفس بروتوكول التجربة 2 الرابحة: 82 خطوة ≈ 3 حقب، batch 1 × grad_accum 4
من الكونفج = 2,048 توكن/خطوة. ذراع الجرعة الموثق: 82–120 خطوة.)

### 3د) DPO الختامية بالأفضليات الدفاعية (الملف المختلط الملتزم 1,393)

```bash
PYTHONPATH=src python -m ryko_lm.training.dpo_train \
    --config configs/models/ryko-small-sft.yaml \
    --init-from artifacts/gpu_bundle_sft/final.pt \
    --pairs data/training/bundle_dpo_pairs_mixed.jsonl \
    --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --out-dir artifacts/gpu_bundle_dpo --epochs 1 --lr 1e-5 --beta 0.1 \
    --seed 1337
```
تركيبة الملف (مانيفست `bundle_dpo_manifest.json`، sha256 `b101dd35…`):
أفضليات لغوية 116×3 تكرار + أفضليات حجاج دفاعية 105 (مغلقة الأفعال
الثمانية) + r2 كاملة 940 (باستبعاد 7 متداخلة مع المسبارات) = 1,393 —
نفس التشكيلة التي حفظت حجاج 76.7 عبر DPO في التجربتين 5 و6.

### 3هـ) الرباعي الكامل على المرشح (تشغيل مزدوج — بند 5)

```bash
CP=artifacts/gpu_bundle_dpo/final.pt
# 1) البذرة الداخلية 315 @0.65:
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ryko_bench_seed.jsonl \
    --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.65 \
    --out docs/results/gpu_bench_final
# 2) rb-ling 52 @0.34:
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark docs/benchmarks/ling_probes.jsonl \
    --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --store data/knowledge/ryko.sqlite --abstain-threshold 0.34 \
    --out docs/results/gpu_ling_final
# 3) RYKO-1000 كاملة (base) — 1,000 mcq + 100 امتنان + 200 اقتباس + مسبارات مجسدة:
PYTHONPATH=src python -m ryko_lm.eval.ryko1000_eval \
    --seed docs/benchmarks/ryko1000_seed.jsonl \
    --store data/knowledge/ryko.sqlite \
    --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --mode base --out docs/results/gpu_ryko1000_final
# 4) PalmX-IC الخارجي 981 (mcq صرف — بلا store):
PYTHONPATH=src python -m ryko_lm.eval.benchmarks run \
    --benchmark data/bench/palmx_ic.jsonl \
    --checkpoint $CP --tokenizer artifacts/tokenizer/ryko-tokenizer.json \
    --abstain-threshold 0.65 --out docs/results/palmx_ic_v5
# ثم يُعاد كل أمر من الأربعة مرة ثانية حرفيًا مع تغيير --out فقط
# (النمط /tmp/..._run2 — حتى لا تُطاول الجولة الثانية ملف الجولة الأولى)
# (بند 5 — قاعدة التشغيل المزدوج).
```
*(تسمية `palmx_ic_v5` هي الاسم التسلسلي التالي المتاح بعد v4 — الصيغة
الأقدم في الوثيقة كانت تقفز إلى v6 بلا سبب. بروتوكول العائلات الضعيفة
بشظايا قائمة العمل الموثق — `istidlal`/`mustalahat` عند 100:300 و`tarjih`
عند 500:600 مع `--slice` و`--skip-engine` ثم `--mode merge` — يُستعمل
بدلاً من الجولة الكاملة فقط عند قيود زمن قاسية؛ الجولة الكاملة أصل البروتوكول
وهي التي أنتجت أرقام 33.71/44.60 المنشورة.)*

كل أصل جديد يُختم قبل الانتقال:

```bash
PYTHONPATH=src python -m ryko_lm.provenance make artifacts/training_small_ling/final.pt \
    --kind checkpoint --meta '{"phase":"5 wide linguistics (GPU v1.27.0)","params":52179456,"init_from":"artifacts/dpo_round2_small/final.pt","steps":20000}' \
    --out artifacts/training_small_ling/final_receipt.json
PYTHONPATH=src python -m ryko_lm.provenance make artifacts/gpu_bundle_sft/final.pt \
    --kind checkpoint --meta '{"phase":"6a bundle SFT (GPU v1.27.0)","init_from":"artifacts/training_small_ling/final.pt","steps":82,"pack":"data/training/bundle_pack (receipt bundle-pack)"}' \
    --out artifacts/gpu_bundle_sft/final_receipt.json
PYTHONPATH=src python -m ryko_lm.provenance make artifacts/gpu_bundle_dpo/final.pt \
    --kind checkpoint --meta '{"phase":"6b defensive DPO (GPU v1.27.0)","pairs":"data/training/bundle_dpo_pairs_mixed.jsonl","beta":0.1,"lr":1e-5,"seed":1337}' \
    --out artifacts/gpu_bundle_dpo/final_receipt.json
PYTHONPATH=src python -m ryko_lm.provenance verify <كل إيصال فوق>
```

---

## 4) بوابات القبول — عقد الإصدار (كلها على نقطة واحدة: `gpu_bundle_dpo/final.pt`)

### 4.1) البوابة التوأم (سبب الجلسة — كلاهما معاً أو لا إصدار)

| البوابة | العتبة | بالعدّ الدقيق | المرجع المقاس |
|---|---|---|---|
| rb-ling (52 @0.34) | **≥ 44.23%** | ≥ 23/52 | نقطة الخدمة `ling_probes_round2.json` = 44.23 (23/52) — الجدار أن لا نقطة CPU تجاوزتها مع حجاج ≥70 |
| علاقة-حجاج (من 315 @0.65) | **≥ 70%** | ≥ 21/30 | التجربة 2 و5 و6 = 76.67 (23/30) — هدف الحزمة المجمّعة |

### 4.2) بوابات عدم التراجع (كسر أي واحدة ⇒ لا ترقية — وثّق واحتفظ)

| البوابة | العتبة الصارمة | بالعدّ | التبرير من السجل المقاس (لا مُختلَق) |
|---|---|---|---|
| MCQ داخلي (159 @0.65) | **≥ 37%** | ≥ 59/159 | نقطة الخدمة 39.62 (63/159، `benchmarks_round2.json`)؛ الشريحة 59–63 = ≤4 حالات = ضجيج قياس 159 حالة؛ دون 59 انحدار مادي تحت رقم منشور. الطموح: ≥55.35 (أفضل مقاس — شظية argrel و التجربة 2/5: 55.97) |
| امتنان صائب داخلي | **= 1.00** | 40/40 خارج-المتن | كل نقطة منشورة منذ v1.14 = 1.00؛ التجربة 2 انكسرت إلى 0.70 فاعتُبرت غير قابلة للترقية — البوابة مثبتة بالفشل نفسه |
| امتنان خاطئ (رفض زائد) داخلي | **≤ 0.65** | ≤ 10/16 | نقطة الخدمة 0.5625 (9/16)؛ 0.6875 (11/16) هو المستوى الذي كسر بوابة الامتنان تاريخياً ورُفضت به تشكيلة r2a (ROADMAP v1.23) |
| cite match داخلي | **= 1.0** | 100/100 | 1.0 على كل نقطة منشورة (`benchmarks_round2.json`، `argrel_proof.json`) |
| PalmX-IC (981 @0.65) | **≥ 29.05%** (هدف ≥31.09) | ≥ 285/981 (هدف 305) | 29.05 = رقم نقطة الخدمة المنشور (`palmx_ic_v4.json`) — لا تراجع؛ 31.09 = أفضل خارجي مقاس (v5c، `palmx_ic_v3`) أي استرداد كلفة المحاذاة −2.04 الموثقة |
| RYKO-1000 base MCQ | **≥ 33.71%** (هدف ≥44.60) | ≥ 337/1000 | 33.71 = منشور نقطة الخدمة (`ryko1000_eval_dpo_r2.json`)؛ 44.60 = شظية argrel (`argrel_proof_ryko1000.json`) |
| RYKO-1000 امتنان صائب / رفض زائد | **= 1.00 / ≤ 0.65** | 36/36 / ≤41/64 | 1.00 محفوظ عبر كل النقاط؛ رفض زائد 0.5938 = 38/64 (نقطة الخدمة) ← 0.4062 (شظية argrel) — 42/64=0.65625 فما فوق هو نطاق الرفض التاريخي المرفوض (0.6875) |
| istidlal / mustalahat / tarjih (نفس النقطة، base) | **≥ 50 / ≥ 40 / ≥ 35** | 50/100، 40/100، 35/100 | قياس معالجة v1.26.0 على نقطة الحزمة SFT-480: 60.0/51.0/44.0 (`bundle_sft_r2_480_ryko1000_parts.json`)؛ نقطة الخدمة 23/23/29 — العتبات تطلب الاحتفاظ بأغلب مكسب المعالجة مع هامش ضجيج (الحالة الواحدة = 1%) |
| PPL مسطرة اللغة على النقطة النهائية | **≤ 101.34** | val_loss ≤ ln(101.34)=4.619 | سقف منع النسيان الكارثي للطلاقة اللغوية عبر مراحل المحاذاة: 101.34 = قياس شظية CPU 300 خطوة الموثق (`LINGUISTICS §5`) — البوابة الأصلية PPL<60 تُفرض عند 5ب على الأساس الواسع قبل المحاذاة |
| البناء المزدوج | **مطابقة بايتية** | — | عقد المشروع الدائم (بند 5) |

### 4.3) منطق القرار

- **كل البوابات (التوأم + عدم التراجع) على نقطة واحدة واعية ⇒ ترقية وفق بند 6.**
- التوأم تحقق وبوابة عدم تراجع انكسرت ⇒ **لا ترقية** — توثيق قياس صادق
  في `docs/results/gpu_session/` وإعادة تشكيل من أذرع الضبط (بند 7) —
  منهجية v1.25.0/v1.26.0 الحرفية (لا نقطة تتفوق بلا كسر بوابة منشورة).
- الأرقام تُقاس ولا تُست extrapolate: كل رقم في الجدول له ملف JSON مصدر
  مذكور، والمقارنة تكون على نفس البروتوكول الحرفي (بند 5).

---

## 5) بروتوكول القياس وقاعدة التشغيل المزدوج

1. **البذرة الداخلية**: `docs/benchmarks/ryko_bench_seed.jsonl` — 315 حالة
   (159 mcq + 56 امتنان + 100 اقتباس) بعتبة **0.65** (نقطة تشغيل v1.14.0
   المعايرة) والمخزن الملتزم `data/knowledge/ryko.sqlite` لحالات cite.
   حجاج = `results.mcq.per_domain["علاقة-حجاج"]` (n=30).
2. **rb-ling**: `docs/benchmarks/ling_probes.jsonl` — 52 mcq بعتبة **0.34**،
   ملف مثبت بايتياً (`a48ec0a2…`) ومحتجَز عن كل حزم التدريب.
3. **RYKO-1000**: `ryko1000_eval --mode base` على البذرة المجمَّدة
   (`a5613cb0…`) — 1,300 حالة (1,000 mcq بالعائلات العشر + 100 امتنان + 200
   اقتباس محرك) + المسبارات المجسدة. العائلات الضعيفة من `per_family` مباشرة.
4. **PalmX-IC**: `data/bench/palmx_ic.jsonl` — 981 mcq صرفة (تحقق: صفر
   cite/abstain) بعتبة 0.65. معيار خارجي append-only لم يُمس منذ v1.15.
5. **التشغيل المزدوج (القرار النهائي قبل الترقية)**: كل قياس من الأربعة
   يُنفَّذ **مرتين متتاليتين بنفس الأمر ونفس الجهاز** — والجولة الثانية
   بـ`--out` مختلف (نمط `/tmp/<الاسم>_run2`) حمايةً من الطوالان:
   - `benchmarks run` (القياسات 1 و2 و4): ملفا JSON **متطابقان بايت-ببايت**
     (تقارير المشروع بلا أختام زمنية) — المطلوب `sha256(a)==sha256(b)`.
   - `ryko1000_eval`: التقرير يتضمن `latency_ms` و`ms` لكل صف — هما
     القياسان الوحيدان المسموح اختلافهما؛ **المطلوب**: تطابق كل المقاييس
     (`mcq`/`abstain`/`cite_alignment`/`probes` وبصمات `rows` مطروحاً منها
     `ms`) بين الجولتين. الفحص:

     ```bash
     PYTHONPATH=src python - <<'EOF'
     import json
     a = json.load(open('docs/results/gpu_ryko1000_final.json'))
     b = json.load(open('/tmp/gpu_ryko1000_final_run2.json'))
     def strip(d):
         d = json.loads(json.dumps(d))
         d['base']['latency_ms'] = {}
         for r in d['base'].get('rows', []): r.pop('ms', None)
         return d
     assert strip(a) == strip(b), 'ryko1000 double-run mismatch'
     print('ryko1000 double-run: metrics identical')
     EOF
     ```
   - أي عدم تطابق ⇒ **توقف**: قفزة أجهزة/عدم حتمية — يُحل قبل أي نشر
     (القياسات التاريخية كلها CPU؛ إن نُفذت الجولة على `--device cuda`
     وجب أن تكون الجولتان المتطابقتان على cuda وأمُيّز ذلك في الدفتر).
6. **التوثيق**: أرقام النقطة المرشحة تُنسخ إلى `docs/results/gpu_session/`
   (بنفس نمط `bundle_experiments/`: ملفات JSON + `index.json` ببصمات sha256)
   قبل أي تعديل على بوابات الترقية.

---

## 6) إجراء الترقية إلى نقطة خدمة جديدة (فقط بعد مرور كل البوابات)

هذا **فعل إصدار واعٍ** بعقد مكتوب داخل docstring اختبار الحارس
(`tests/test_guard_provenance.test_serving_default_is_the_promoted_round2_point`)
— الترتيب الإلزامي:

1. **الإيصال**: `final_receipt.json` للنقطة الجديدة (أمر `provenance make`
   في 3هـ) + `PYTHONPATH=src python -m ryko_lm.provenance verify` PASS،
   وsha256 الملف يُسجل في دفتر الجلسة.
2. **القياس الخارجي على النقطة الجديدة نفسها** (بند 5 كامل، تشغيل مزدوج
   مطابق) — لا ترقية بقياسات على نقطة غير المقدَّمة (درس خلل v1.17.0
   المعاكس الذي أصلحته v1.24.0).
3. **تحديث التثبيت**:
   - `tests/test_guard_provenance.py`: `PROMOTED_SERVING_POINT` إلى المسار
     الجديد و`PROMOTED_SHA256` إلى sha256 الجديد (القيم الحالية
     `artifacts/dpo_round2_small/final.pt` / `f0361c4a…52157d` **لا تُمس**
     قبل الترقية).
   - `tests/conftest.py`: `FINAL_MODEL` إلى المسار الجديد.
   - `src/ryko_lm/serving/api.py`: `MODEL_PATH` الافتراضي + سجل الترقيات
     في docstring الوحدة (سطر القاعدة الحالي: `dpo_round2_small` منذ v1.24.0).
   - `docs/ROUTING.md`: جدول نقطة الخدمة (L2) وأرقامها.
   - `CHANGELOG.md`: بند `[1.27.0]` بأرقام الرباعي الكامل المطابقة بايتياً
     لملفات `docs/results/gpu_session/`.
   - `docs/ROADMAP.md`/`docs/EVALUATION.md`: حالة المرحلة 5/الجولة التالية.
4. **الحارس الأخضر**: `PYTHONPATH=src python -m pytest -q` ⇒ 246 passed +
   1 skipped (اختبارات الحارس ترفض بصوت عالٍ أي نقطة غير مُرقّاة/غير
   مستلمة/متبدلة البايتات).
5. **الوسم**: `git tag v1.27.0` بعد المسح السري (`make secret-scan`) وصحة
   الإيصالات — **دون أي push** (سياسة المستودع: الدفع قرار منفصل).

---

## 7) مسارات الفشل والتراجع

1. **فشل بوابة ⇒ لا ترقية** (document-and-hold، منهجية v1.26.0 الحرفية):
   الدفتر يُكمل بأمانة (`docs/results/gpu_session/` + فهرس sha256)، نقطة
   الخدمة والإيصال وثابتا الحارس يبقيان كما هما، ولا وسم. الأصول تبقى
   ملتزمة كنقاط عمل موثقة (نمط `proof_bundle_small`).
2. **أذرع إعادة التشكيل الجاهزة** (بالترتيب الاقتصادي):
   - جرعة SFT: 82 ← 120 خطوة (نطاق موثق من التجارب 1↔2) أو العكس.
   - أذرع الأفضليات: `scripts/build_bundle_dpo.py --ling-repeat {1,3,5}
     --argrel-repeat {1,2} --r2-mode {ling,full}` — **ملاحظة إلزامية**:
     أي ملف أفضليات جديد يُعاد إيصاله ويُعاد فحص التسرب (بوابة المسبارات
     قاتلة داخل الباني) ويُقيَّس قبل الاستعمال (أرقام التجربة 5 = التشكيلة
     المرجعية 116×3+105+940).
   - β/lr للـDPO (الضعف #10 في `docs/EVALUATION.md`): تجربة β أكبر على
     شظايا صغيرة قبل الجولة الكاملة — لا يُجرَّب مباشرة على النقطة النهائية.
   - جرعة الأساس الواسع: `--batch-size 8` (≈4.8 حقب) أو `--total-steps
     12000` إن أظهر القياس تحفظاً/نسياناً زائدًا.
   - التقاطع WiSE-FT (`scripts/soup_checkpoints.py --a X --b Y --alpha 0.5
     --out Z`) ذراع إنقاذ أخير فقط: الجدار المقيّس يسقف rb-ling عند
     36.54 مع انطفاء حجاج (تجربة 7) — لا يُتوقع أن يحقق التوأم، يُستعمل
     لتشخيص فقط.
3. **تراجع ما بعد ترقية خاطئة** (لو اكتُشفت بعد الوسم): ثابتا الحارس
   يعادان للقيم السابقة، `conftest.FINAL_MODEL` و`api.MODEL_PATH` و
   `ROUTING` تُعاد — نقطة الخدمة السابقة ما تزال ملتزمة بإيصالها فالرجوع
   بايتي لا مُخترع، ويُوثق الحادث في CHANGELOG بند الإصلاح.
4. **انهيار بيئة GPU في منتصف الجولة**: كل مرحلة استئنافية بالضبط
   (`last.pt` ذرّية الكتابة) — يُعاد نفس الأمر؛ لا يُحذف `last.pt` إلا
   قراراً واعياً بإعادة مرحلة من الصفر (`--reset`).
5. **فشل PPL<60 عند 5ب**: لا يُدخل الجلسة مرحلة المحاذاة أصلاً — الجلسة
   تُغلق بدفتر تشخيص (جرعة/تقارب) وتُعاد جدولة؛ لا يُختم الأساس الواسع
   كأصل اعتماد.

---

## 8) جدول المخاطر

| الخطر | المؤشر المبكر | الصمّام في العقد |
|---|---|---|
| انجراف تفضيلي / ارتفاع PPL بعد DPO (المقايضة الموثقة: val PPL 165.5←247.6 على الخارجي في v1.23) | PalmX ينزل تحت 29.05؛ PPL المسطرة يتجاوز 101.34 | بوابتا PalmX ≥29.05 وسقف PPL ≤101.34؛ أفضليات دفاعية إجبارية في 3د (التجربتان 5/6: حجاج ثابت 76.7 بلاها ينهر إلى 3.3) |
| نسيان كارثي لسلوك الامتنان/الاقتباس | امتنان صائب <1.00 أو cite <1.0 أو رفض زائد >0.65 | بوابات عدم التراجع الصارمة (4.2) — مثبتة بالفشل التاريخي (تجربة 2: 0.70؛ r2a: 0.6875 مرفوضة) |
| OOM على GPU (logits 2.15GB/نسخة عند batch 32) | OOM في أول خطوات 3أ | `--batch-size 8` (ذراع مضبوط)؛ درس CPU الموثق: المصفوفة تتناسب خطياً مع batch |
| تحفظ زائد على 88 مستنداً (19 حقبة) يقصم العائلات الأخرى | MCQ/istidlal ينزلان في 7 رغم نجاح PPL | أذرع الجرعة (batch 8 / خطوات أقل)؛ بوابات istidlal/mustalahat/tarjih و MCQ تجبر إعادة الضبط |
| تسرب المسبارات في أي حزمة جديدة | فشل باني الحزم ببوابة قاتلة، أو اختبارات `test_bundle_pack` | البوابات القاتلة داخل `pack_bundle`/`build_ling_sft`/`build_bundle_dpo` + إعادة الاختبار قبل أي استعمال (مسبارات `a48ec0a2…` محتجَزة عن كل الحزم) |
| عدم حتمية عابرة للأجهزة (cpu↔cuda) | تشغيل مزدوج غير متطابق | قاعدة التشغيل المزدوج بنفس الجهاز (بند 5.5) — القياسات المنشورة تاريخياً CPU |
| اختلاط نقطة القياس بنقطة الخدمة (خلل v1.17.0 بالاتجاهين) | أرقام دفتر على نقطة غير المقدمة | إجراء الترقية (6): القياس الكامل على النقطة المرشحة نفسها ثم التثبيت الثلاثي (حارس + conftest + api) |
| نفاد قرص/شبكة أثناء جمع lugha | فشل `ling-pack`/الإيصال | بند 2.3 إجباري قبل 5أ: لا تدريب على حزمة غير مُتحقق إيصالها |

---

## 9) ملخص عقد الجلسة (سطر واحد لكل مرحلة)

1. CPU pre-flight (بند 2) كامل وأخضر — 246+1 اختبار، إيصالات، بايتات.
2. 5أ: 20K خطوة واسعة من نقطة الخدمة → إيصال + بوابة PPL<60.
3. 6أ/6ب: الحزمة المجمّعة (82) + DPO دفاعي (697) → إيصالات.
4. 7: الرباعي كامل ×2 على المرشح → جدول 4.1+4.2 كلّه أخضر.
5. ترقية واعية (بند 6) أو document-and-hold (بند 7) — لا طريق ثالث.
