# تقرير بوابة الجودة النهائية — الوكيل 5 (تحضير إصدار v1.27.0)

**التاريخ:** 2026-11-19 · **القاعدة:** main @ `1996089` (v1.26.0) · **المهنة:** بوابة الجودة النهائية (الوكيل 5 من 5)
**النطاق:** جمع مخرجات الوكلاء 1-4، التحقق المتقاطع المستقل، تصحيح المتبقي، الحكم على جاهزية جلسة GPU.

---

## 1) حالة الوكلاء (1-5)

| الوكيل | المخرَج | حالة التحقق عند البوابة |
|---|---|---|
| 1 — مهندس الخطة | `docs/GPU_SESSION_PLAN.md` (517 سطراً) + إصلاح warm-start في `Makefile` | ✅ مُتحقق (§4 و §7 أدناه) |
| 2 — المراجع | 246 passed + 1 skipped؛ توفيق 3 أرقام في `docs/results/*.md` | ✅ أعيد التشغيل مستقلاً: **246 passed, 1 skipped** |
| 3 — المنافسة | `docs/COMPETITIVE.md` (+172) و`docs/ROADMAP_LEADERSHIP.md` (213) | ✅ الأرقام والاستشهادات مُتحققة (§3 و §5) |
| 4 — الاختبار العميق | `docs/qa/FLAW_REPORT.md` (F1 major + F2-F5 minor) | ✅ مُعالَج (§6) — F1 أُصلح، والبقية مفتوحة بمعالجات مقترحة |
| 5 — بوابة الجودة | هذا التقرير + إصلاحان رقميان + إصلاح F1 + اختبار | ✅ هذا المستند |

## 2) التحقق المستقل (البوابة الصارمة)

- **الجناح الكامل قبل إصلاح F1:** `246 passed, 1 skipped in 4:33` (إعادة تشغيل مستقلة جديدة، لا اعتماد على تقارير الوكلاء).
- **الجناح الكامل بعد إصلاح F1:** `247 passed, 1 skipped in 4:36` (الاختبار الزائد = اختبار انحدار F1 الجديد).
- **حالة الشجرة:** مطابقة تماماً للمتوقع — تعديلات: `Makefile`، `docs/COMPETITIVE.md`، `docs/GPU_SESSION_PLAN.md`، `docs/results/{argrel_proof_palmx,argrel_proof_ling,ling_probes_round2}.md`، وإضافات: `docs/ROADMAP_LEADERSHIP.md`، `docs/qa/`، `scripts/qa_temp/` (سكراتش مسموح: `smoke.py`)، ثم إضافات البوابة: `src/ryko_lm/serving/api.py` و `tests/test_serving.py`. لا مفاجآت. لم يُؤدَّ أي commit أو push (تعليمات المهمة).

## 3) تدقيق الأرقام (المصدر: `docs/results/*.json` + الإيصالات + CHANGELOG)

| الرقم في الوثائق الجديدة | المصدر المُتحقق | الحالة |
|---|---|---|
| rb-ling **44.23** (23/52) | `ling_probes_round2.json:12` = 0.4423 | ✅ |
| علاقة-حجاج **76.7 / 76.67** | `argrel_proof.json:20` + `bundle_experiments/*.json` = 0.7667 | ✅ |
| MCQ داخلي **39.62** | `benchmarks_round2.json:12` = 0.3962 | ✅ |
| امتناع صائب **1.00** / خاطئ **0.5625** | `benchmarks_round2.json:39` (+ `abstain_calibration.json:95`) | ✅ |
| PalmX-IC **29.05** / **31.09** | `palmx_ic_v4.json:12` / `palmx_ic_v3.json:12` | ✅ |
| RYKO-1000 MCQ **33.71** | `ryko1000_eval_dpo_r2.json:139` = 0.3371 | ✅ |
| istidlal/mustalahat/tarjih **60/51/44** | `bundle_experiments/bundle_sft_r2_480_ryko1000_parts.json` = 0.6/0.51/0.44 (n=100 لكل عائلة) | ✅ |
| معلمات **52,179,456 / 52.18** | مثبتة بالاختبار (`tests/test_serving.py` و`test_ryko_small_param_contract`) + `params_M: 52.18` في كل نتائج JSON | ✅ |
| توكنات **34,240,402 / 148,631** | `data/processed_ling/binarize_stats.json` + إيصال `train.bin` 68,480,804 بايت / `val.bin` 297,262 بايت | ✅ |
| مقاطع **423,250** | `ryko1000_verification.json:178` ("lugha": 423250) | ✅ |
| PPL مسطرة **≤101.34** | `CHANGELOG.md:367` (val_loss 4.6185 = PPL 101.34) و `LINGUISTICS.md:144` | ✅ |
| أحرف الحزمة **108,831,798** | ❌ الإيصال الحالي `receipt.json` = **108,831,563**، والتوكنات التاريخية 34,240,204 ≠ الإيصال 34,240,402 | **مُصلَح** |

**الإصلاحان الرقمان (تحرير أدنى):** خلط تاريخي في وثيقتين جديدتين نقل رقما سجل v1.21.0 القديم (108,831,798 / 34,240,204) فوق قيم الإيصال الحالي الملتزَم (108,831,563 / 34,240,402):
- `docs/COMPETITIVE.md` (سطر «مقياس البيانات»): صُحح إلى قيم الإيصال مع الإبقاء على السجل التاريخي مذكوراً بين قوسين.
- `docs/ROADMAP_LEADERSHIP.md` (صف v2.0.0-GPU): نفس التصحيح.
- ملاحظة: `CHANGELOG.md:348` و`docs/LINGUISTICS.md:117` و`docs/ROADMAP.md` تحفظ أرقام v1.21.0 **كسجل تاريخي موثَّق بتاريخه** — لم تُمسّ؛ القيمة الحاكمة لجلسة GPU هي الإيصال الملتزَم (وتتحقق آلياً عبر `provenance verify` بعد إعادة البناء).

## 4) قابلية تنفيذ `docs/GPU_SESSION_PLAN.md`

- **الوحدات:** `ryko_lm.training.train` / `training.evaluate` / `eval.benchmarks` / `eval.ryko1000_eval` / `provenance` — كلها موجودة (لا أثر لـ `eval.perplexity` الوهمية).
- **الأعلام:** `--init-from`، `--min-lr` في `train.py`؛ `--ling-repeat/--argrel-repeat/--r2-mode` في `scripts/build_bundle_dpo.py` — موجودة.
- **السكربتات:** `build_bundle_dpo.py`، `soup_checkpoints.py`، `verify_all_receipts.py` — موجودة.
- **أهداف Makefile:** `lugha-collect`، `ling-pack`، `ling-pack-binarize`، `ling-train-small`، `secret-scan` — موجودة؛ و`make -n ling-train-small` يمرّ مع `--init-from artifacts/dpo_round2_small/final.pt` (إصلاح الوكيل 1 مثبت).
- **المصنوعات المُشار إليها:** `artifacts/dpo_round2_small/final.pt` + إيصاله، وإيصالات `training_v5b`/`proof_bundle_small`/`proof_argrel_small`، و`artifacts/tokenizer/ryko-tokenizer.json` — كلها على القرص.

## 5) تدقيق الاستشهادات التنافسية (فتح المصادر حيّاً)

| المصدر | الأرقام المقتبسة | النتيجة |
|---|---|---|
| ورقة PalmX (`arXiv 2509.02550` عبر ar5iv) | 84.22 / 83.82 / 82.52 (منصة Subtask-2) و75.12 (NileChat-3B) | ✅ كلها موجودة في النص |
| بطاقة `inception42/Jais-3-7B` | «PaLM-X 2025-2\*» 91.2، QIAS-1\* 88.5، IslamicEval2025-1B\* 74.9 | ✅ موجودة في البطاقة |
| بطاقة `humain-ai/ALLaM-7B-Instruct-preview` | رخصة Apache-2.0 | ✅ مؤكدة (تاريخ 2025-02-13 من بيانات HF الوصفية، ليس في README — غير حاسم) |

لا شيء يتطلب وسم «غير متحقق» بعد هذا الفحص؛ البنود غير المؤكدة سلفاً موسومة أصلاً في `COMPETITIVE.md`.

## 6) العيوب (من `docs/qa/FLAW_REPORT.md`) — التسريع النهائي

| العيب | الخطورة | القرار عند البوابة |
|---|---|---|
| **F1** صمت EOS الفوري بلا إشارة امتنان | major | **مُصلَح في الكود.** إصلاح إضافي ~6 أسطر في `src/ryko_lm/serving/api.py`: كل رد `/generate` يتضمن الآن `abstained` (bool) و`finish_reason` (`"eos_immediate"` عند إكمال فارغ و`generated_tokens ≤ 1`، وإلا `"stop"`) — حقول مضافة لا تكسر العملاء. تحقق ميداني أولي: بذرة 315 على «ما هي أركان الإسلام؟» و«ما هو تقويم الهجري؟» ⇒ إكمال فارغ + `gen=1` كما وثّق الوكيل 4. اختبار انحدار جديد `test_generate_reports_immediate_eos_abstention` + تأكيد المسار المعاكس في `test_generate_endpoint`؛ `tests/test_serving.py`: **7 passed**، ثم الجناح الكامل **247 passed, 1 skipped**. (`/generate/stream` تُركت كما هي — تحسينها يؤجل عن قصد.) |
| F2 عدم حتمية مسار seed=None | minor | مفتوح — توثيق «بلا بذرة = عينة» وتشجيع تمرير بذرة؛ لا تغيير كود |
| F3 سقف 2000 حرف مقابل ميزانية 512 رمزاً | minor | مفتوح — توثيق السقف ومبرره الرمزي في عقد الـ API؛ التقطيع يؤجل (قرار معماري) |
| F4 انجراف اقتباس مُفبرك في الجيل الحر | minor/جودة | مفتوح — مرشح المعالجة الطبيعي إعادة موازنة DPO بأفضليات دفاعية بعد v1.27.0 |
| F5 تباين تسمية `completion`/`text` | minor/info | مفتوح — ملاحظة تناغم في العقد؛ التوحيد الفعلي يؤجل لعدم كسر العملاء |

## 7) جدول بوابات جلسة GPU v1.27.0 (بعد التحقق المتبادل)

| البوابة | العتبة | خط أساس مُتحقق عند البوابة |
|---|---|---|
| rb-ling (52 @0.34) | ≥ 44.23 (≥23/52) | 44.23 = نقطة الخدمة (`ling_probes_round2.json`) |
| علاقة-حجاج | ≥ 70 | 76.67 (`argrel_proof.json`) |
| MCQ داخلي (159 @0.65) | ≥ 37 | 39.62 = 63/159 (`benchmarks_round2.json`) |
| امتناع صائب / خاطئ | = 1.00 / ≤ 0.65 | 1.00 / 0.5625 |
| توثيق اقتباس | = 1.0 | 1.0 |
| PalmX-IC (981 @0.65) | ≥ 29.05 (هدف ≥31.09) | 29.05 (`palmx_ic_v4.json`)؛ 31.09 (`palmx_ic_v3.json`) |
| RYKO-1000 MCQ | ≥ 33.71 | 33.71 (`ryko1000_eval_dpo_r2.json`) |
| istidlal / mustalahat / tarjih | ≥ 50 / 40 / 35 | 60/51/44 (`bundle_sft_r2_480_ryko1000_parts.json`) |
| PPL مسطرة اللغة | ≤ 101.34 | سقف CHANGELOG v1.25.0 (val_loss 4.6185) |
| حتمية القياس المزدوج | تطابق بايتي | أعاد الوكيل 4 إنتاجها (sha256 متطابق + dict-compare IDENTICAL) |

## 8) الحكم النهائي: هل نبدأ جلسة GPU الآن؟

### **نعم — الجلسة جاهزة، بستة شروط إلزامية:**

1. **إعادة بناء البيانات محلياً قبل أي أمر GPU** (بقعة `data/processed_ling/*.bin` غير ملتزَمة عمداً — والتحقق من الإيصال يفشل محلياً بدونها **بالتصميم**):
   `make lugha-collect && make ling-pack && make ling-pack-binarize` ثم `PYTHONPATH=src python -m ryko_lm.provenance verify data/processed_ling/receipt.json` ⇒ يجب PASS ومطابقة 34,240,402 / 148,631.
2. تنفيذ قسم pre-flight على CPU في الخطة أولاً (الإيصالات الثمانية + جناح الاختبار + sha256 نقطة الخدمة `f0361c4a…52157d`).
3. الالتزام الصارم ببوابات §7 أعلاه عند كل مرحلة (5أ→7) — أي فشل بوابة = توقف وتراجع (rollback) لا تقدّم قسري.
4. لا مساس بـ `artifacts/dpo_round2_small/final.pt` و`final_receipt.json` وثوابت `tests/test_guard_provenance.py` (نقطة التراجع المقدسة).
5. الجلسة على GPU واحد ≥24GB وفق جرعات الخطة (20,000 خطوة ≈ 19.1 حقبة على 34,240,402 توكن) مع الأفضليات الدفاعية الإجبارية في 3د.
6. بعد الجلسة: مسح سري (`make secret-scan`) ثم وسم `git tag v1.27.0` — **ولا commit/push ضمن مهمة البوابة هذه** (كل التعديلات فوق متروكة في شجرة العمل للمراجعة).

### المتبقي ما بعد الجلسة (غير مانع)
- F2-F5 في `docs/qa/FLAW_REPORT.md` (توثيق/تناغم، لا كود حرج).
- أرقام v1.21.0 التاريخية في `CHANGELOG.md`/`LINGUISTICS.md`/`ROADMAP.md` تبقى كسجل مؤرَّخ؛ الحاكم هو إيصال البيانات الملتزَم.
- توسيع `/generate/stream` بحقلَي الامتنان (يوحَّد مع F5 في تحديث عقد لاحق).

---
*بوابة الجودة — الوكيل 5. كل ما ورد أعلاه مُتحقق آلياً على الشجرة الحالية بتاريخ 2026-11-19. الجناح النهائي: **247 passed, 1 skipped**.*
