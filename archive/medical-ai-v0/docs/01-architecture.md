# الوثيقة الهندسية التنفيذية — MEDICAL-AI CORE v0

**المستودع التنظيمي:** Ryko | **التاريخ:** 12 نوفمبر 2026 | **ترتيب البناء:** انظر [BUILD_ORDER.md](BUILD_ORDER.md)

---

## 1. قرارات معمارية ملزمة (غير قابلة للنقاش داخل v0)

1. **Monorepo** بوحدات مستقلة، **عقود أولاً**: لا كود محرك قبل اعتماد مخططه.
2. **المنسّق آلة حالات صريحة**، لا وكلاء أحرار. كل انتقال مسجّل في التدقيق.
3. **السلامة كود حتمي فوق الجميع**: مصدر واحد للقواعد يولّد Python + Dart، مع اختبارات تكافؤ إلزامية.
4. **كل الحسابات بكود**: فروق، تحويل وحدات (UCUM)، اتجاهات. LLM يفهم ويصوغ فقط.
5. **الثقة = معايرة/مقاييس استخراج**، لا تقييم ذاتي من LLM.
6. **محتوى المستندات بيانات غير موثوقة** — لا تنفيذ تعليمات منه (مقاومة الحقن).
7. **تأكيد بشري إلزامي** لكل قيمة مستخرجة قبل دخولها السجل المؤكد.
8. **Provenance لكل datum**: مستند/صفحة/إحداثيات/إصدار محرك/حالة تأكيد.
9. **Feature Flags تنظيمية**: `pharma.interactions` مطفأ حتى توقيع ترخيص KB؛ `devices.*` و`clinician_mode` مطفأة (v1).
10. **لا PHI في السجلات/التتبع** — تنقيح إلزامي + اختبار تسريب في CI.

## 2. هيكل المستودع

```
Ryko/
├── docs/
│   ├── 00-intended-use.md          # المستند 0 (يوقّع قبل W1)
│   ├── 01-architecture.md          # هذه الوثيقة
│   ├── BUILD_ORDER.md              # ترتيب البناء (8 موجات)
│   ├── REPO_AUDIT.md               # ناتج بروتوكول فحص المستودع W0
│   └── adr/                        # قرارات معمارية مرقمة
├── contracts/v1/                   # JSON Schema 2020-12 — المصدر الوحيد للحقيقة
│   ├── engine_result.schema.json
│   ├── triage.schema.json
│   ├── lab_extraction.schema.json
│   ├── interaction_check.schema.json
│   └── pre_visit_report.schema.json
├── safety/
│   ├── rules/*.yaml                # قواعد الخطر + القيم الحرجة (موقّعة سريرياً ⚠)
│   ├── codegen/                    # مولّد Python + Dart + فحص التكافؤ
│   └── golden/triage_500.jsonl     # 500 حالة، 100 طارئة محكّمة
├── services/
│   └── safety/rules_generated.py   # ناتج المولّد (بوابة الخطر — الخادم)
├── mobile/medai_app/lib/safety/    # ناتج المولّد (بوابة الخطر المحلية — Dart)
├── evals/                          # مشغّل البوابات + فاحص العتبات + التقارير
├── scripts/                        # مولّد المجموعة الذهبية
├── tests/                          # اختبارات الوحدة والعقود
├── alembic/                        # هجرات قاعدة البيانات
├── infra/                          # Docker, Terraform, Keycloak, CI/CD (W2+)
└── Makefile                        # make safety-gates / evals / parity / audit
```

## 3. العقود — الغلاف الموحد لكل محرك

كل محرك يعيد `EngineResult` (مقتطف الحقول الملزمة):

```json
{
  "engine": "lab|pharma|records|reports",
  "engine_version": "semver", "rules_version": "semver|null",
  "status": "ok | abstain | error",
  "data": { },
  "evidence": [{"source_document_id":"uuid","page":2,"bbox":[0,0,100,20],"quote":"..."}],
  "confidence": {"method":"ocr_field_score|kb_exact_match|rule","value":0.97},
  "abstain_reason": "unmapped_unit | low_ocr | conflicting_values | null",
  "limitations": ["قيم غير مؤكدة مستثناة"],
  "safety_flags": ["critical_value:potassium"]
}
```

قواعد العقد: `abstain` حالة نجاح وليست فشلاً؛ لا يجوز `status:ok` مع `evidence` فارغة في المحركات السريرية؛ كل رقم في `data` يجب أن يطابق قيمة في قاعدة البيانات (يفحصه CI بعينات).

**عقد بوابة الخطر** (يعمل على الجهاز بلا إنترنت):

```json
// Input
{"symptoms":["chest_pain","dyspnea"],"duration_min":20,"age_band":"45-60",
 "confirmed_critical_labs":[],"locale":"ar-SA"}
// Output
{"level":"EMERGENCY","rule_id":"RF-001","rules_version":"1.0.0",
 "action":"call_ambulance","emergency_number":"997",
 "message_key":"safety.emergency.chest_pain"}
```

جدول أرقام الطوارئ قابل للتهيئة حسب الدولة/المنطقة (السعودية 997، الإمارات 998، مصر 123…) ⚠ يُراجع قانونياً.

## 4. آلة حالات المنسّق

```
RECEIVED → SAFETY_GATE ─(EMERGENCY)→ EMERGENCY_RESPONSE [نهائي فوري]
              │
              ▼
         CLASSIFY → COLLECT (أسئلة ناقصة، جولتان كحد أقصى)
              → ROUTE → ENGINE_RUN (متوازٍ، مهلة لكل محرك 30s)
              → VERIFY (حتمي: وحدات/مدى/تعارض/اكتمال أدلة)
              → COMPOSE (قالب + صياغة LLM مقيدة)
              → SAFETY_REVIEW (تمرير القواعد على المخرَج النهائي)
              → DELIVER | ABSTAIN
```

- الطوارئ تقاطع أي حالة وتُنهي المسار فوراً — البوابة تعمل قبل أي استدعاء LLM وبموازاته.
- كل انتقال: سطر تدقيق (من/إلى/سبب/إصدارات). مهلة إجمالية للمهمة 120s ⇒ `ABSTAIN` مع سبب.

## 5. محرك السلامة — مصدر واحد، منصتان

```yaml
# safety/rules/RF-001_chest_pain_complex.yaml   ⚠ يتطلب توقيعاً سريرياً
id: RF-001
when:
  all: [symptom.chest_pain]
  any: [symptom.dyspnea, symptom.diaphoresis, symptom.radiating_pain, duration_min >= 15]
then: {level: EMERGENCY, action: call_ambulance}
```

- عتبات القيم الحرجة الافتراضية (بوتاسيوم ≤2.5/≥6.5، غلوكوز ≤54 mg/dL، Hb ≤7 g/dL، صفيحات ≤20×10⁹/L، INR ≥5، صوديوم ≤120/≥160) **كلها ⚠ معلّقة على التوقيع السريري** وتُخزّن بإصدار موقّع.
- الصيغة التنفيذية الآلية المعتمدة في W1 (قابلة للتوليد والفحص) في `safety/rules/*.yaml` — قيم حرجة: `CV-K-01` (بوتاسيوم ≥6.5 mmol/L من تحاليل مؤكدة فقط).
- `make parity`: الحمولة الموحدة للقواعد مطابقة بايت-ببايت بين Python وDart + مجموعة ذهبية + 1000 حالة عشوائية (الجزء السلوكي يتطلب Dart SDK — انظر `safety/codegen/parity_check.py`). أي انحراف = فشل بناء.

## 6. قاعدة البيانات (PostgreSQL 16 + pgvector؛ مقتطف الجداول الحرجة)

```sql
CREATE TABLE lab_result (
  id uuid PRIMARY KEY, patient_id uuid NOT NULL,
  document_id uuid, extraction_run_id uuid,
  loinc_code text,                    -- 718-7 Hb, 2823-3 K+, 2345-7 Glucose...
  original_name text,                 -- الاسم كما ظهر (عربي/إنجليزي)
  value_num numeric, value_text text,
  unit_ucum text,                     -- g/dL, mmol/L, 10*3/uL, %
  original_unit text,
  ref_low numeric, ref_high numeric, ref_unit_ucum text,
  effective_at timestamptz, collected_at timestamptz,
  confirmation_status text CHECK (confirmation_status IN
    ('extracted','confirmed','corrected','rejected')),
  confirmed_by uuid, confirmed_at timestamptz,
  page_no int, bbox jsonb, provenance_id uuid
);
CREATE INDEX ON lab_result (patient_id, loinc_code, effective_at)
  WHERE confirmation_status IN ('confirmed','corrected');

CREATE TABLE loinc_map (              -- حوكمة: لا خريطة تدخل بلا موافقة
  local_name text, lang text, lab_vendor text, loinc_code text,
  approved_by uuid, approved_at timestamptz, PRIMARY KEY(local_name,lang,lab_vendor));

CREATE TABLE unit_map (original_unit text PRIMARY KEY, ucum_code text,
  factor numeric, approved_by uuid);  -- التحويل حتمي عبر UCUM فقط

CREATE TABLE medication_statement (id uuid PRIMARY KEY, patient_id uuid,
  atc_code text,                      -- B01AA03 warfarin, C10AA01 simvastatin
  rxnorm_code text, brand_local_id uuid, name_text text,
  dose_text text, frequency_text text,  -- كما أبلغ المريض، بلا تدقيق جرعة
  status text, source text);

CREATE TABLE interaction_check (id uuid PRIMARY KEY, patient_id uuid,
  kb_vendor text, kb_version text, inputs_hash text,
  result_json jsonb, checked_at timestamptz);

CREATE TABLE audit_log (seq bigserial, ts timestamptz, actor uuid,
  action text, resource text, payload_hash text, prev_hash text);
  -- append-only + سلسلة تجزئة؛ إرساء يومي خارجي
```

جداول مكملة: `patient, consent, document, extraction_run, allergy, drug_catalog (brand↔ATC↔SFDA reg), clinical_report, safety_event, provenance`.

**التعيين إلى FHIR R4:** lab_result→Observation (بعد التأكيد فقط)، document→DocumentReference، medication_statement→MedicationStatement، تقرير الزيارة→Composition، وكل provenance→Provenance. الأنظمة: `http://loinc.org`، `http://unitsofmeasure.org`، `http://www.whocc.no/atc`.

**المحلي (SQLCipher):** نسخة قراءة فقط من المؤكد + قواعد الخطر + أرقام الطوارئ. الحذف (PDPL) عبر Crypto-shredding: مفتاح بيانات لكل مريض يُتلف عند طلب الحذف.

## 7. واجهات API (FastAPI، `/api/v1`، OIDC، Idempotency-Key على POST، أخطاء RFC 9457)

| Endpoint | الوظيفة | ملاحظات |
|---|---|---|
| `POST /documents` | رفع ملف (multipart) | 202 + job؛ فحص نوع/حجم/برمجيات خبيثة |
| `GET /documents/{id}/extraction` | نتائج الاستخراج الموسومة | كل حقل بثقة + bbox |
| `POST /documents/{id}/confirmations` | تأكيد/تصحيح/رفض جملة | الذراع البشرية الإلزامية |
| `GET /labs?loinc=&from=&to=` | القيم المؤكدة فقط افتراضياً | `status=extracted` يتطلب flag عرض |
| `GET /labs/{loinc}/trend` | اتجاه حتمي | وحدات موحّدة UCUM |
| `POST/GET/PATCH /medications`, `/allergies` | قائمة المريض | مصدر=patient موثّق |
| `POST /interaction-checks` | فحص KB | **409 إذا العلم مطفأ (بلا ترخيص)** |
| `POST /triage` | مرآة خادم لبوابة الجهاز | الجهاز يعمل offline أولاً |
| `POST /reports/pre-visit` → `GET /reports/{id}` | Composition→PDF | وسم «ليس تشخيصاً» + ملحق provenance |
| `POST /assistant/intake` | حوار أعراض منظم | LLM مقيد بمخطط، جولتان كحد أقصى |
| `POST /data-export`, `POST /data-deletion` | حقوق PDPL | تدقيق كامل |

## 8. حواجز LLM (طبقة `services/llm`)

1. **منع الأرقام**: أي رمز رقمي في نص LLM المولّد ⇒ رفض وإعادة توليد (regex gate)؛ الأرقام تُحقن من DB في القالب.
2. **تقييد المخرجات** بمخططات JSON؛ لا نص حر في مسارات القرار.
3. **تأريض**: الشرح من `content/` المراجعة طبياً + بيانات المريض المؤكدة فقط؛ اقتباس مصدر إلزامي.
4. **مقاومة الحقن**: محتوى المستندات في قناة بيانات معزولة؛ لا استدعاء أدوات مصدره نص مستند؛ مجموعة اختبار 100 مستند خبيث.
5. **سجل نماذج**: كل إصدار + مجموعة تقييم + تاريخ اعتماد؛ تغيير النموذج = تشغيل البوابات كاملة (نمط PCCP حتى قبل الاعتماد التنظيمي [[16]]).

## 9. بوابات القبول الرقمية (CI يمنع الإصدار عند أي إخفاق)

| البوابة | العتبة | مجموعة الاختبار |
|---|---|---|
| حساسية الطوارئ | **≥99%** | 100 حالة طارئة محكّمة ضمن 500 |
| نوعية الفرز (منع فرط التوجيه) | Specificity **≥85%** | الـ400 غير الطارئة |
| ألم صدر منفرد | 0 طمأنة (قاعدة صلبة) | فحص شامل |
| القيم الحرجة | استرجاع **100%** | مجموعة قيم حرجة ذهبية |
| استخراج حقول التحاليل | **≥99.5%** field-level | 300 تقرير (عربي/إنجليزي/مصور، ≥10 تنسيقات مختبرات) |
| تكافؤ القواعد Python↔Dart | **100%** | ذهبية + 1000 عشوائية |
| التداخلات الكبرى/الموانع | استرجاع **100%** | 200 زوج من KB المرخّص (بعد التفعيل) |
| زمن بوابة الخطر | **p95 <1s** offline | جهاز مرجعي متوسط (4GB RAM) |
| استخراج 5 صفحات | p95 <60s | مجموعة الأداء |
| حقن المستندات | 0 تنفيذ تعليمات | 100 مستند خبيث |
| PHI في السجلات | 0 تسريب | فحص آلي |
| الأمن | OWASP ASVS L2 + MASVS L2؛ 0 حرج/عالٍ في اختبار اختراق | قبل الإطلاق |
| صحة الامتناع | يُقاس ويُعرض لكل إصدار | حالات فشل تحقق مصطنعة |

**المنفذ في هذا المستودع الآن:** حساسية الطوارئ، نوعية الفرز، ألم صدر منفرد، القيم الحرجة (CV-K-01)، تكافؤ الحمولة Python↔Dart — عبر `make safety-gates && make parity`. بقية البوابات تُفعّل مع موجاتها (W3+).

## 10. بروتوكول فحص المستودع قبل أي تعديل (W0)

```bash
git log --oneline -20 && git status          # حالة الفرع والنظافة
tree -L 3 -I 'node_modules|.git'             # البنية الفعلية
cloc . && pip-audit && flutter pub outdated  # الحجم والاعتماديات
gitleaks detect                              # أسرار مسربة
pytest --collect-only -q                     # جرد الاختبارات الموجودة
```

- قرار لكل وحدة قائمة: **تبنّي / إعادة هيكلة / حجر** (كود بلا اختبارات = حجر افتراضياً).
- **pharma-sdl**: مرشح لنواة PHARMA — يُفحص بنفس البروتوكول؛ ما ينجح منه يُرحّل كـ adapter، والباقي يُعاد بناؤه على العقود.
- الناتج: `docs/REPO_AUDIT.md` موقّع، يُرفق بالوثيقة قبل W1.

## 11. سجل المخاطر والقرارات المفتوحة

| الخطر | الاحتمال/الأثر | التخفيف |
|---|---|---|
| تأخر ترخيص KB | متوسط/عالٍ | الإطلاق بدونه: PHARMA = تنظيم قائمة فقط، والتداخلات ميزة لاحقة بالعلم |
| ضعف OCR العربي على تقارير مصورة | عالٍ/عالٍ | bake-off ملزم + تأكيد بشري + امتناع عند الثقة المنخفضة |
| انزياح عتبات المختبرات | عالٍ/متوسط | ref_range من المستند نفسه، لا جداول عامة؛ خريطة وحدات معتمدة |
| تمدد النطاق | عالٍ/قاتل | المستند 0 + مراجعة كل ميزة مقابل 0.3 في كل sprint |
| حادث سلامة في بيتا | منخفض/قاتل | وقف بالعلم ≤24h + سجل حوادث + تأمين مسؤولية قبل البيتا |

**قرارات تُحسم في W0:** سوق الإطلاق الأول، مزود KB، مزود OCR، استضافة (سيادة البيانات السعودية إن لزم)، هوية القائد السريري. راجع [adr/ADR-0001-market-kb.md](adr/ADR-0001-market-kb.md).
